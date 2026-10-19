#include "dsp_demo/retrieval_rnn_descent_core.h"

#include <errno.h>
#include <float.h>
#include <limits.h>
#include <stdio.h>
#include <stddef.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define RNN_GRAPH_CACHE_MAGIC 0x524E4E47u
#define RNN_GRAPH_CACHE_DISTANCE_SOURCE_FLOAT 0

typedef struct RetrievalRnnGraphCacheHeader {
    uint32_t magic;
    uint32_t version;
    int32_t num_vectors;
    int32_t dim;
    int32_t build_s;
    int32_t build_r;
    int32_t build_t1;
    int32_t build_t2;
    uint32_t seed;
    int32_t distance_source;
    int32_t layout_name_len;
    uint64_t degrees_offset;
    uint64_t neighbors_offset;
    uint64_t neighbor_dists_offset;
    uint64_t file_size;
} RetrievalRnnGraphCacheHeader;

typedef struct RetrievalRnnInsert {
    int src;
    int dst;
    float dist;
} RetrievalRnnInsert;

typedef struct RnnDistanceRuntime {
    const float *base;
    int dim;
    const RetrievalRnnBuildRuntime *runtime;
    RetrievalRnnBuildStats *stats;
    pthread_mutex_t *distance_mu;
} RnnDistanceRuntime;

typedef struct RnnRangeWorker {
    RetrievalRnnGraph *graph;
    RnnDistanceRuntime dist_rt;
    RetrievalRnnBuildStats stats;
    RetrievalRnnInsert *pending;
    int32_t *out_graph;
    int pending_count;
    int pending_cap;
    int s;
    int top_k;
    int start;
    int end;
    uint32_t seed;
    int ret;
} RnnRangeWorker;

static uint64_t rnn_now_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
}

static size_t rnn_slot(const RetrievalRnnGraph *graph, int src, int pos) {
    return (size_t)src * (size_t)graph->max_degree + (size_t)pos;
}

static uint32_t rnn_lcg_next(uint32_t x) {
    return x * 1664525u + 1013904223u;
}

static uint32_t rnn_seed_for_node(uint32_t seed, int node) {
    uint32_t x = seed ^ (uint32_t)(node + 1) * 0x9e3779b9u;
    x = rnn_lcg_next(x);
    x ^= x >> 16;
    return rnn_lcg_next(x);
}

static int rnn_edge_is_better(float new_dist, int new_id,
                              float old_dist, int old_id) {
    if (new_dist < old_dist) {
        return 1;
    }
    if (new_dist > old_dist) {
        return 0;
    }
    return new_id < old_id;
}

static int rnn_graph_alloc_storage(RetrievalRnnGraph *graph,
                                   int num_vectors,
                                   int dim,
                                   int max_degree,
                                   int allocate_flags) {
    size_t edge_cap;
    size_t i;

    if (!graph || num_vectors <= 0 || dim <= 0 || max_degree <= 0 ||
        (size_t)num_vectors > SIZE_MAX / (size_t)max_degree) {
        return -1;
    }

    memset(graph, 0, sizeof(*graph));
    graph->num_vectors = num_vectors;
    graph->dim = dim;
    graph->max_degree = max_degree;

    edge_cap = (size_t)num_vectors * (size_t)max_degree;
    graph->degrees = (int *)calloc((size_t)num_vectors, sizeof(int));
    graph->neighbors = (int *)malloc(edge_cap * sizeof(int));
    graph->neighbor_dists = (float *)malloc(edge_cap * sizeof(float));
    if (allocate_flags) {
        graph->is_new = (uint8_t *)calloc(edge_cap, sizeof(uint8_t));
    }
    if (!graph->degrees || !graph->neighbors ||
        !graph->neighbor_dists || (allocate_flags && !graph->is_new)) {
        retrieval_rnn_graph_free(graph);
        return -1;
    }

    for (i = 0; i < edge_cap; i++) {
        graph->neighbors[i] = -1;
        graph->neighbor_dists[i] = FLT_MAX;
    }

    return 0;
}

void retrieval_rnn_build_stats_reset(RetrievalRnnBuildStats *stats) {
    if (stats) {
        memset(stats, 0, sizeof(*stats));
    }
}

static void rnn_stats_accumulate(RetrievalRnnBuildStats *dst,
                                 const RetrievalRnnBuildStats *src) {
    if (!dst || !src) {
        return;
    }

    dst->init_us += src->init_us;
    dst->update_us += src->update_us;
    dst->reverse_us += src->reverse_us;
    dst->export_us += src->export_us;
    dst->distance_calls += src->distance_calls;
    dst->distance_batches += src->distance_batches;
    dst->distance_pairs_requested += src->distance_pairs_requested;
    dst->distance_pairs_cpu += src->distance_pairs_cpu;
    dst->distance_pairs_dsp += src->distance_pairs_dsp;
    dst->distance_rpc_us += src->distance_rpc_us;
    dst->distance_fallbacks += src->distance_fallbacks;
    if (src->pending_max > dst->pending_max) {
        dst->pending_max = src->pending_max;
    }
    dst->changed_edges += src->changed_edges;
    dst->edge_insertions += src->edge_insertions;
    dst->edge_replacements += src->edge_replacements;
    dst->duplicate_rejects += src->duplicate_rejects;
    dst->self_rejects += src->self_rejects;
}

int retrieval_rnn_graph_alloc(RetrievalRnnGraph *graph,
                              int num_vectors,
                              int dim,
                              int max_degree) {
    return rnn_graph_alloc_storage(graph, num_vectors, dim, max_degree, 1);
}

void retrieval_rnn_graph_free(RetrievalRnnGraph *graph) {
    if (!graph) {
        return;
    }

    free(graph->degrees);
    free(graph->neighbors);
    free(graph->neighbor_dists);
    free(graph->is_new);
    memset(graph, 0, sizeof(*graph));
}

int retrieval_rnn_graph_alloc_without_flags(RetrievalRnnGraph *graph,
    int num_vectors, int dim, int max_degree) {
    return rnn_graph_alloc_storage(graph, num_vectors, dim, max_degree, 0);
}

float retrieval_rnn_l2_distance_cpu(const float *base,
                                    int dim,
                                    int a,
                                    int b) {
    const float *va;
    const float *vb;
    float acc = 0.0f;
    int d;

    if (!base || dim <= 0 || a < 0 || b < 0) {
        return FLT_MAX;
    }

    va = base + (size_t)a * (size_t)dim;
    vb = base + (size_t)b * (size_t)dim;
    for (d = 0; d < dim; d++) {
        float diff = va[d] - vb[d];
        acc += diff * diff;
    }
    return acc;
}

static int rnn_cpu_distance_batch(void *ctx,
                                  const int32_t *ids_a,
                                  const int32_t *ids_b,
                                  int pair_count,
                                  float *out_dists) {
    RnnDistanceRuntime *rt = (RnnDistanceRuntime *)ctx;
    int i;

    if (!rt || !rt->base || !ids_a || !ids_b || !out_dists ||
        pair_count <= 0) {
        return -1;
    }

    for (i = 0; i < pair_count; i++) {
        out_dists[i] = retrieval_rnn_l2_distance_cpu(rt->base,
                                                     rt->dim,
                                                     ids_a[i],
                                                     ids_b[i]);
    }
    return 0;
}

static int rnn_distance_batch(RnnDistanceRuntime *rt,
                              const int32_t *ids_a,
                              const int32_t *ids_b,
                              int pair_count,
                              float *out_dists) {
    RetrievalRnnDistanceBatchFn fn;
    void *ctx;

    if (!rt || !ids_a || !ids_b || !out_dists || pair_count <= 0) {
        return -1;
    }

    fn = (rt->runtime && rt->runtime->distance_batch)
        ? rt->runtime->distance_batch
        : rnn_cpu_distance_batch;
    ctx = (rt->runtime && rt->runtime->distance_batch)
        ? rt->runtime->distance_ctx
        : rt;

    if (rt->stats) {
        rt->stats->distance_batches++;
        rt->stats->distance_pairs_requested += (uint64_t)pair_count;
        rt->stats->distance_calls += (uint64_t)pair_count;
        if (rt->runtime && rt->runtime->distance_batch) {
            rt->stats->distance_pairs_dsp += (uint64_t)pair_count;
        } else {
            rt->stats->distance_pairs_cpu += (uint64_t)pair_count;
        }
    }

    if (rt->distance_mu && rt->runtime && rt->runtime->distance_batch) {
        int rc;
        pthread_mutex_lock(rt->distance_mu);
        rc = fn(ctx, ids_a, ids_b, pair_count, out_dists);
        pthread_mutex_unlock(rt->distance_mu);
        return rc;
    }

    return fn(ctx, ids_a, ids_b, pair_count, out_dists);
}

static int rnn_distance_batch_capacity(const RnnDistanceRuntime *rt,
                                       int fallback) {
    if (rt && rt->runtime && rt->runtime->batch_size > 0) {
        return rt->runtime->batch_size;
    }
    return (fallback > 0) ? fallback : 1;
}

int retrieval_rnn_graph_has_edge(const RetrievalRnnGraph *graph,
                                 int src,
                                 int dst) {
    int i;

    if (!graph || src < 0 || src >= graph->num_vectors ||
        dst < 0 || dst >= graph->num_vectors) {
        return 0;
    }

    for (i = 0; i < graph->degrees[src]; i++) {
        if (graph->neighbors[rnn_slot(graph, src, i)] == dst) {
            return 1;
        }
    }

    return 0;
}

static int rnn_graph_insert_edge_with_distance(RetrievalRnnGraph *graph,
                                               int src,
                                               int dst,
                                               float dist,
                                               uint8_t mark_new,
                                               RetrievalRnnBuildStats *stats) {
    int degree;
    int farthest_pos = -1;
    int i;

    if (!graph || src < 0 || src >= graph->num_vectors ||
        dst < 0 || dst >= graph->num_vectors || dist < 0.0f) {
        return -1;
    }
    if (src == dst) {
        if (stats) {
            stats->self_rejects++;
        }
        return 0;
    }
    if (retrieval_rnn_graph_has_edge(graph, src, dst)) {
        if (stats) {
            stats->duplicate_rejects++;
        }
        return 0;
    }

    degree = graph->degrees[src];
    if (degree < graph->max_degree) {
        size_t slot = rnn_slot(graph, src, degree);
        graph->neighbors[slot] = dst;
        graph->neighbor_dists[slot] = dist;
        if (graph->is_new) {
            graph->is_new[slot] = mark_new ? 1u : 0u;
        }
        graph->degrees[src] = degree + 1;
        if (stats) {
            stats->edge_insertions++;
        }
        return 1;
    }

    for (i = 0; i < degree; i++) {
        size_t slot = rnn_slot(graph, src, i);
        if (farthest_pos < 0 ||
            rnn_edge_is_better(graph->neighbor_dists[rnn_slot(graph, src, farthest_pos)],
                               graph->neighbors[rnn_slot(graph, src, farthest_pos)],
                               graph->neighbor_dists[slot],
                               graph->neighbors[slot])) {
            farthest_pos = i;
        }
    }

    if (farthest_pos >= 0) {
        size_t slot = rnn_slot(graph, src, farthest_pos);
        if (rnn_edge_is_better(dist, dst,
                               graph->neighbor_dists[slot],
                               graph->neighbors[slot])) {
            graph->neighbors[slot] = dst;
            graph->neighbor_dists[slot] = dist;
            if (graph->is_new) {
                graph->is_new[slot] = mark_new ? 1u : 0u;
            }
            if (stats) {
                stats->edge_replacements++;
            }
            return 1;
        }
    }

    return 0;
}

int retrieval_rnn_graph_insert_edge(RetrievalRnnGraph *graph,
                                    const float *base,
                                    int src,
                                    int dst,
                                    uint8_t mark_new,
                                    RetrievalRnnBuildStats *stats) {
    float dist;

    if (!graph || !base || src < 0 || src >= graph->num_vectors ||
        dst < 0 || dst >= graph->num_vectors) {
        return -1;
    }
    if (src == dst) {
        if (stats) {
            stats->self_rejects++;
        }
        return 0;
    }
    if (retrieval_rnn_graph_has_edge(graph, src, dst)) {
        if (stats) {
            stats->duplicate_rejects++;
        }
        return 0;
    }

    dist = retrieval_rnn_l2_distance_cpu(base, graph->dim, src, dst);
    if (stats) {
        stats->distance_calls++;
    }
    return rnn_graph_insert_edge_with_distance(graph, src, dst, dist,
                                               mark_new, stats);
}

static void rnn_row_remove_at(RetrievalRnnGraph *graph, int src, int pos) {
    int degree = graph->degrees[src];
    int i;

    if (!graph || src < 0 || src >= graph->num_vectors ||
        pos < 0 || pos >= degree) {
        return;
    }

    for (i = pos + 1; i < degree; i++) {
        size_t dst_slot = rnn_slot(graph, src, i - 1);
        size_t src_slot = rnn_slot(graph, src, i);
        graph->neighbors[dst_slot] = graph->neighbors[src_slot];
        graph->neighbor_dists[dst_slot] = graph->neighbor_dists[src_slot];
        if (graph->is_new) {
            graph->is_new[dst_slot] = graph->is_new[src_slot];
        }
    }

    degree--;
    graph->degrees[src] = degree;
    graph->neighbors[rnn_slot(graph, src, degree)] = -1;
    graph->neighbor_dists[rnn_slot(graph, src, degree)] = FLT_MAX;
    if (graph->is_new) {
        graph->is_new[rnn_slot(graph, src, degree)] = 0u;
    }
}

void retrieval_rnn_graph_sort_row(RetrievalRnnGraph *graph, int src) {
    int degree;
    int i;

    if (!graph || src < 0 || src >= graph->num_vectors) {
        return;
    }

    degree = graph->degrees[src];
    for (i = 1; i < degree; i++) {
        int id = graph->neighbors[rnn_slot(graph, src, i)];
        float dist = graph->neighbor_dists[rnn_slot(graph, src, i)];
        uint8_t flag = graph->is_new ? graph->is_new[rnn_slot(graph, src, i)] : 0u;
        int j = i - 1;

        while (j >= 0 &&
               rnn_edge_is_better(dist, id,
                                  graph->neighbor_dists[rnn_slot(graph, src, j)],
                                  graph->neighbors[rnn_slot(graph, src, j)])) {
            size_t to = rnn_slot(graph, src, j + 1);
            size_t from = rnn_slot(graph, src, j);
            graph->neighbors[to] = graph->neighbors[from];
            graph->neighbor_dists[to] = graph->neighbor_dists[from];
            if (graph->is_new) {
                graph->is_new[to] = graph->is_new[from];
            }
            j--;
        }

        graph->neighbors[rnn_slot(graph, src, j + 1)] = id;
        graph->neighbor_dists[rnn_slot(graph, src, j + 1)] = dist;
        if (graph->is_new) {
            graph->is_new[rnn_slot(graph, src, j + 1)] = flag;
        }
    }
}

static void *rnn_export_topk_worker_fn(void *arg) {
    RnnRangeWorker *worker = (RnnRangeWorker *)arg;
    int u;

    worker->ret = 0;
    for (u = worker->start; u < worker->end; u++) {
        int j;
        retrieval_rnn_graph_sort_row(worker->graph, u);
        for (j = 0; j < worker->top_k; j++) {
            int value = -1;
            if (j < worker->graph->degrees[u]) {
                value = worker->graph->neighbors[rnn_slot(worker->graph, u, j)];
            }
            worker->out_graph[(size_t)u * (size_t)worker->top_k + (size_t)j] =
                (int32_t)value;
        }
    }
    return NULL;
}

int retrieval_rnn_export_topk_threads(RetrievalRnnGraph *graph,
                                      int top_k,
                                      int threads,
                                      int32_t *out_graph) {
    RnnRangeWorker *workers = NULL;
    pthread_t *tids = NULL;
    int launched = 0;
    int rc = 0;
    int i;

    if (!graph || !out_graph || top_k <= 0 || threads <= 0) {
        return -1;
    }

    if (threads > graph->num_vectors) {
        threads = graph->num_vectors;
    }
    if (threads <= 1 || graph->num_vectors < 2) {
        RnnRangeWorker worker;
        memset(&worker, 0, sizeof(worker));
        worker.graph = graph;
        worker.out_graph = out_graph;
        worker.top_k = top_k;
        worker.start = 0;
        worker.end = graph->num_vectors;
        rnn_export_topk_worker_fn(&worker);
        return worker.ret;
    }

    workers = (RnnRangeWorker *)calloc((size_t)threads, sizeof(*workers));
    tids = (pthread_t *)calloc((size_t)threads, sizeof(*tids));
    if (!workers || !tids) {
        free(workers);
        free(tids);
        return -1;
    }

    for (i = 0; i < threads; i++) {
        workers[i].graph = graph;
        workers[i].out_graph = out_graph;
        workers[i].top_k = top_k;
        workers[i].start = (int)(((int64_t)graph->num_vectors * i) / threads);
        workers[i].end = (int)(((int64_t)graph->num_vectors * (i + 1)) / threads);
        if (pthread_create(&tids[i], NULL,
                           rnn_export_topk_worker_fn,
                           &workers[i]) != 0) {
            workers[i].ret = -1;
            break;
        }
        launched++;
    }

    for (i = 0; i < launched; i++) {
        pthread_join(tids[i], NULL);
    }

    if (launched != threads) {
        rc = -1;
    }
    for (i = 0; i < threads; i++) {
        if (workers[i].ret != 0) {
            rc = -1;
        }
    }

    free(workers);
    free(tids);
    return rc;
}

int retrieval_rnn_export_topk(RetrievalRnnGraph *graph,
                              int top_k,
                              int32_t *out_graph) {
    return retrieval_rnn_export_topk_threads(graph, top_k, 1, out_graph);
}

static int rnn_random_graph_range(RnnDistanceRuntime *dist_rt,
                                  RetrievalRnnGraph *graph,
                                  int s,
                                  uint32_t seed,
                                  int start,
                                  int end,
                                  RetrievalRnnBuildStats *stats) {
    int32_t *batch_src = NULL;
    int32_t *batch_dst = NULL;
    float *batch_dist = NULL;
    int *row_candidates = NULL;
    int batch_count = 0;
    int batch_cap;
    int u;

    if (!dist_rt || !graph || s <= 0 || graph->num_vectors <= 1) {
        return -1;
    }
    if (s > graph->max_degree) {
        s = graph->max_degree;
    }
    if (s > graph->num_vectors - 1) {
        s = graph->num_vectors - 1;
    }

    if (start < 0) {
        start = 0;
    }
    if (end > graph->num_vectors) {
        end = graph->num_vectors;
    }

    batch_cap = rnn_distance_batch_capacity(dist_rt, s);
    if (batch_cap < s) {
        batch_cap = s;
    }
    batch_src = (int32_t *)malloc((size_t)batch_cap * sizeof(int32_t));
    batch_dst = (int32_t *)malloc((size_t)batch_cap * sizeof(int32_t));
    batch_dist = (float *)malloc((size_t)batch_cap * sizeof(float));
    row_candidates = (int *)malloc((size_t)s * sizeof(int));
    if (!batch_src || !batch_dst || !batch_dist || !row_candidates) {
        free(batch_src);
        free(batch_dst);
        free(batch_dist);
        free(row_candidates);
        return -1;
    }

    for (u = start; u < end; u++) {
        uint32_t state = rnn_seed_for_node(seed, u);
        uint64_t attempts = 0;
        uint64_t attempt_limit = (uint64_t)(uint32_t)graph->num_vectors *
                                 (uint64_t)(uint32_t)graph->num_vectors;
        int row_count = 0;

        while (row_count < s && attempts < attempt_limit) {
            int v;
            int dup = 0;
            int i;

            state = rnn_lcg_next(state);
            v = (int)(state % (uint32_t)graph->num_vectors);
            if (u != v) {
                for (i = 0; i < row_count; i++) {
                    if (row_candidates[i] == v) {
                        dup = 1;
                        break;
                    }
                }
                if (!dup && !retrieval_rnn_graph_has_edge(graph, u, v)) {
                    row_candidates[row_count++] = v;
                } else if (stats) {
                    stats->duplicate_rejects++;
                }
            } else if (stats) {
                stats->self_rejects++;
            }
            attempts++;
        }
        if (row_count < s) {
            int v;
            for (v = 0; v < graph->num_vectors && row_count < s; v++) {
                int dup = 0;
                int i;

                if (u == v) {
                    if (stats) {
                        stats->self_rejects++;
                    }
                    continue;
                }
                for (i = 0; i < row_count; i++) {
                    if (row_candidates[i] == v) {
                        dup = 1;
                        break;
                    }
                }
                if (!dup && !retrieval_rnn_graph_has_edge(graph, u, v)) {
                    row_candidates[row_count++] = v;
                } else if (stats) {
                    stats->duplicate_rejects++;
                }
            }
        }

        if (batch_count + row_count > batch_cap && batch_count > 0) {
            int i;
            if (rnn_distance_batch(dist_rt,
                                   batch_src,
                                   batch_dst,
                                   batch_count,
                                   batch_dist) != 0) {
                free(batch_src);
                free(batch_dst);
                free(batch_dist);
                free(row_candidates);
                return -1;
            }
            for (i = 0; i < batch_count; i++) {
                rnn_graph_insert_edge_with_distance(graph,
                                                    (int)batch_src[i],
                                                    (int)batch_dst[i],
                                                    batch_dist[i],
                                                    1u,
                                                    stats);
            }
            batch_count = 0;
        }

        {
            int i;
            for (i = 0; i < row_count; i++) {
                batch_src[batch_count] = (int32_t)u;
                batch_dst[batch_count] = (int32_t)row_candidates[i];
                batch_count++;
            }
        }
    }

    if (batch_count > 0) {
        int i;
        if (rnn_distance_batch(dist_rt,
                               batch_src,
                               batch_dst,
                               batch_count,
                               batch_dist) != 0) {
            free(batch_src);
            free(batch_dst);
            free(batch_dist);
            free(row_candidates);
            return -1;
        }
        for (i = 0; i < batch_count; i++) {
            rnn_graph_insert_edge_with_distance(graph,
                                                (int)batch_src[i],
                                                (int)batch_dst[i],
                                                batch_dist[i],
                                                1u,
                                                stats);
        }
    }

    free(batch_src);
    free(batch_dst);
    free(batch_dist);
    free(row_candidates);
    return 0;
}

static void *rnn_random_graph_worker_fn(void *arg) {
    RnnRangeWorker *worker = (RnnRangeWorker *)arg;

    worker->ret = rnn_random_graph_range(&worker->dist_rt,
                                         worker->graph,
                                         worker->s,
                                         worker->seed,
                                         worker->start,
                                         worker->end,
                                         &worker->stats);
    return NULL;
}

static int rnn_random_graph(RnnDistanceRuntime *dist_rt,
                            RetrievalRnnGraph *graph,
                            int s,
                            uint32_t seed,
                            int threads,
                            RetrievalRnnBuildStats *stats) {
    RnnRangeWorker *workers = NULL;
    pthread_t *tids = NULL;
    int launched = 0;
    int rc = 0;
    int i;

    if (!dist_rt || !graph || threads <= 0) {
        return -1;
    }

    if (threads <= 1 || graph->num_vectors < 2) {
        return rnn_random_graph_range(dist_rt,
                                      graph,
                                      s,
                                      seed,
                                      0,
                                      graph->num_vectors,
                                      stats);
    }

    workers = (RnnRangeWorker *)calloc((size_t)threads, sizeof(*workers));
    tids = (pthread_t *)calloc((size_t)threads, sizeof(*tids));
    if (!workers || !tids) {
        free(workers);
        free(tids);
        return -1;
    }

    for (i = 0; i < threads; i++) {
        workers[i].graph = graph;
        workers[i].dist_rt = *dist_rt;
        workers[i].dist_rt.stats = &workers[i].stats;
        workers[i].s = s;
        workers[i].seed = seed;
        workers[i].start = (int)(((int64_t)graph->num_vectors * i) / threads);
        workers[i].end = (int)(((int64_t)graph->num_vectors * (i + 1)) / threads);
        if (pthread_create(&tids[i], NULL,
                           rnn_random_graph_worker_fn,
                           &workers[i]) != 0) {
            workers[i].ret = -1;
            break;
        }
        launched++;
    }

    for (i = 0; i < launched; i++) {
        pthread_join(tids[i], NULL);
    }

    if (launched != threads) {
        rc = -1;
    }
    for (i = 0; i < threads; i++) {
        if (workers[i].ret != 0) {
            rc = -1;
        }
        rnn_stats_accumulate(stats, &workers[i].stats);
    }

    free(workers);
    free(tids);
    return rc;
}

static int rnn_insert_list_append(RetrievalRnnInsert **items,
                                  int *count,
                                  int *cap,
                                  int src,
                                  int dst,
                                  float dist) {
    RetrievalRnnInsert *new_items;
    int new_cap;

    if (!items || !count || !cap) {
        return -1;
    }

    if (*count >= *cap) {
        new_cap = (*cap > 0) ? (*cap * 2) : 64;
        new_items = (RetrievalRnnInsert *)realloc(
            *items, (size_t)new_cap * sizeof(RetrievalRnnInsert));
        if (!new_items) {
            return -1;
        }
        *items = new_items;
        *cap = new_cap;
    }

    (*items)[*count].src = src;
    (*items)[*count].dst = dst;
    (*items)[*count].dist = dist;
    (*count)++;
    return 0;
}

static int rnn_insert_compare(const void *a, const void *b) {
    const RetrievalRnnInsert *ia = (const RetrievalRnnInsert *)a;
    const RetrievalRnnInsert *ib = (const RetrievalRnnInsert *)b;

    if (ia->src != ib->src) {
        return (ia->src < ib->src) ? -1 : 1;
    }
    if (ia->dst != ib->dst) {
        return (ia->dst < ib->dst) ? -1 : 1;
    }
    if (ia->dist < ib->dist) {
        return -1;
    }
    if (ia->dist > ib->dist) {
        return 1;
    }
    return 0;
}

static int rnn_apply_insertions(RetrievalRnnGraph *graph,
                                RetrievalRnnInsert *items,
                                int count,
                                RetrievalRnnBuildStats *stats) {
    int i;
    int changed = 0;

    if (count > 1) {
        qsort(items, (size_t)count, sizeof(items[0]), rnn_insert_compare);
    }

    for (i = 0; i < count; i++) {
        int r = rnn_graph_insert_edge_with_distance(graph,
                                                    items[i].src,
                                                    items[i].dst,
                                                    items[i].dist,
                                                    1u,
                                                    stats);
        if (r < 0) {
            return -1;
        }
        if (r > 0) {
            changed++;
        }
    }
    return changed;
}

static int rnn_update_neighbors_range(RnnDistanceRuntime *dist_rt,
                                      RetrievalRnnGraph *graph,
                                      int start,
                                      int end,
                                      RetrievalRnnInsert **pending_out,
                                      int *pending_count_out,
                                      RetrievalRnnBuildStats *stats) {
    RetrievalRnnInsert *pending = NULL;
    int32_t *batch_a = NULL;
    int32_t *batch_b = NULL;
    float *batch_dist = NULL;
    int pending_count = 0;
    int pending_cap = 0;
    int u;

    if (!dist_rt || !graph || !pending_out || !pending_count_out) {
        return -1;
    }

    if (start < 0) {
        start = 0;
    }
    if (end > graph->num_vectors) {
        end = graph->num_vectors;
    }

    batch_a = (int32_t *)malloc((size_t)graph->max_degree * sizeof(int32_t));
    batch_b = (int32_t *)malloc((size_t)graph->max_degree * sizeof(int32_t));
    batch_dist = (float *)malloc((size_t)graph->max_degree * sizeof(float));
    if (!batch_a || !batch_b || !batch_dist) {
        free(batch_a);
        free(batch_b);
        free(batch_dist);
        return -1;
    }

    for (u = start; u < end; u++) {
        int i = 0;
        retrieval_rnn_graph_sort_row(graph, u);
        while (i < graph->degrees[u]) {
            int v = graph->neighbors[rnn_slot(graph, u, i)];
            float dist_uv = graph->neighbor_dists[rnn_slot(graph, u, i)];
            uint8_t flag_v = graph->is_new[rnn_slot(graph, u, i)];
            int remove = 0;
            int w_pos;
            int batch_count = 0;

            for (w_pos = 0; w_pos < i; w_pos++) {
                int w = graph->neighbors[rnn_slot(graph, u, w_pos)];
                uint8_t flag_w = graph->is_new[rnn_slot(graph, u, w_pos)];

                if (!flag_v && !flag_w) {
                    continue;
                }
                batch_a[batch_count] = (int32_t)v;
                batch_b[batch_count] = (int32_t)w;
                batch_count++;
            }

            if (batch_count > 0 &&
                rnn_distance_batch(dist_rt,
                                   batch_a,
                                   batch_b,
                                   batch_count,
                                   batch_dist) != 0) {
                free(batch_a);
                free(batch_b);
                free(batch_dist);
                free(pending);
                return -1;
            }

            for (w_pos = 0; w_pos < batch_count; w_pos++) {
                int w = (int)batch_b[w_pos];
                float dist_vw = batch_dist[w_pos];

                if (dist_uv >= dist_vw) {
                    if (rnn_insert_list_append(&pending, &pending_count,
                                               &pending_cap, w, v,
                                               dist_vw) != 0) {
                        free(batch_a);
                        free(batch_b);
                        free(batch_dist);
                        free(pending);
                        return -1;
                    }
                    remove = 1;
                    break;
                }
            }

            if (remove) {
                rnn_row_remove_at(graph, u, i);
            } else {
                i++;
            }
        }

        for (i = 0; i < graph->degrees[u]; i++) {
            graph->is_new[rnn_slot(graph, u, i)] = 0u;
        }
    }

    free(batch_a);
    free(batch_b);
    free(batch_dist);

    if (stats && (uint64_t)pending_count > stats->pending_max) {
        stats->pending_max = (uint64_t)pending_count;
    }
    *pending_out = pending;
    *pending_count_out = pending_count;
    return 0;
}

static void *rnn_update_neighbors_worker_fn(void *arg) {
    RnnRangeWorker *worker = (RnnRangeWorker *)arg;

    worker->ret = rnn_update_neighbors_range(&worker->dist_rt,
                                             worker->graph,
                                             worker->start,
                                             worker->end,
                                             &worker->pending,
                                             &worker->pending_count,
                                             &worker->stats);
    return NULL;
}

static int rnn_concat_worker_pending(RnnRangeWorker *workers,
                                     int worker_count,
                                     RetrievalRnnInsert **pending_out,
                                     int *pending_count_out) {
    RetrievalRnnInsert *pending = NULL;
    int total = 0;
    int offset = 0;
    int i;

    if (!workers || worker_count <= 0 || !pending_out || !pending_count_out) {
        return -1;
    }

    for (i = 0; i < worker_count; i++) {
        if (workers[i].pending_count < 0 ||
            workers[i].pending_count > INT_MAX - total) {
            return -1;
        }
        total += workers[i].pending_count;
    }

    if (total > 0) {
        pending = (RetrievalRnnInsert *)malloc(
            (size_t)total * sizeof(RetrievalRnnInsert));
        if (!pending) {
            return -1;
        }
        for (i = 0; i < worker_count; i++) {
            if (workers[i].pending_count > 0) {
                memcpy(pending + offset,
                       workers[i].pending,
                       (size_t)workers[i].pending_count *
                           sizeof(RetrievalRnnInsert));
                offset += workers[i].pending_count;
            }
        }
    }

    *pending_out = pending;
    *pending_count_out = total;
    return 0;
}

static int rnn_update_neighbors_once(RnnDistanceRuntime *dist_rt,
                                     RetrievalRnnGraph *graph,
                                     int threads,
                                     RetrievalRnnBuildStats *stats) {
    RetrievalRnnInsert *pending = NULL;
    int pending_count = 0;
    int rc;

    if (!dist_rt || !graph || threads <= 0) {
        return -1;
    }

    if (threads <= 1 || graph->num_vectors < 2) {
        rc = rnn_update_neighbors_range(dist_rt,
                                        graph,
                                        0,
                                        graph->num_vectors,
                                        &pending,
                                        &pending_count,
                                        stats);
        if (rc != 0) {
            free(pending);
            return -1;
        }
    } else {
        RnnRangeWorker *workers = NULL;
        pthread_t *tids = NULL;
        int launched = 0;
        int i;

        workers = (RnnRangeWorker *)calloc((size_t)threads, sizeof(*workers));
        tids = (pthread_t *)calloc((size_t)threads, sizeof(*tids));
        if (!workers || !tids) {
            free(workers);
            free(tids);
            return -1;
        }

        for (i = 0; i < threads; i++) {
            workers[i].graph = graph;
            workers[i].dist_rt = *dist_rt;
            workers[i].dist_rt.stats = &workers[i].stats;
            workers[i].start = (int)(((int64_t)graph->num_vectors * i) / threads);
            workers[i].end = (int)(((int64_t)graph->num_vectors * (i + 1)) / threads);
            if (pthread_create(&tids[i], NULL,
                               rnn_update_neighbors_worker_fn,
                               &workers[i]) != 0) {
                workers[i].ret = -1;
                break;
            }
            launched++;
        }

        for (i = 0; i < launched; i++) {
            pthread_join(tids[i], NULL);
        }

        rc = (launched == threads) ? 0 : -1;
        for (i = 0; i < threads; i++) {
            if (workers[i].ret != 0) {
                rc = -1;
            }
            rnn_stats_accumulate(stats, &workers[i].stats);
        }
        if (rc == 0 &&
            rnn_concat_worker_pending(workers, threads,
                                      &pending, &pending_count) != 0) {
            rc = -1;
        }
        for (i = 0; i < threads; i++) {
            free(workers[i].pending);
        }
        free(workers);
        free(tids);
        if (rc != 0) {
            free(pending);
            return -1;
        }
    }

    if (stats && (uint64_t)pending_count > stats->pending_max) {
        stats->pending_max = (uint64_t)pending_count;
    }
    rc = rnn_apply_insertions(graph, pending, pending_count, stats);
    if (stats && rc > 0) {
        stats->changed_edges += (uint64_t)rc;
    }
    free(pending);
    return rc;
}

static int rnn_collect_reverse_edges_range(RetrievalRnnGraph *graph,
                                           int start,
                                           int end,
                                           RetrievalRnnInsert **pending_out,
                                           int *pending_count_out,
                                           RetrievalRnnBuildStats *stats) {
    RetrievalRnnInsert *pending = NULL;
    int pending_count = 0;
    int pending_cap = 0;
    int u;

    if (!graph || !pending_out || !pending_count_out) {
        return -1;
    }

    if (start < 0) {
        start = 0;
    }
    if (end > graph->num_vectors) {
        end = graph->num_vectors;
    }

    for (u = start; u < end; u++) {
        int i;
        for (i = 0; i < graph->degrees[u]; i++) {
            int v = graph->neighbors[rnn_slot(graph, u, i)];
            float dist = graph->neighbor_dists[rnn_slot(graph, u, i)];
            if (rnn_insert_list_append(&pending, &pending_count,
                                       &pending_cap, v, u, dist) != 0) {
                free(pending);
                return -1;
            }
        }
    }

    if (stats && (uint64_t)pending_count > stats->pending_max) {
        stats->pending_max = (uint64_t)pending_count;
    }
    *pending_out = pending;
    *pending_count_out = pending_count;
    return 0;
}

static void *rnn_collect_reverse_edges_worker_fn(void *arg) {
    RnnRangeWorker *worker = (RnnRangeWorker *)arg;

    worker->ret = rnn_collect_reverse_edges_range(worker->graph,
                                                  worker->start,
                                                  worker->end,
                                                  &worker->pending,
                                                  &worker->pending_count,
                                                  &worker->stats);
    return NULL;
}

static int rnn_graph_add_reverse_edges(RetrievalRnnGraph *graph,
                                       int threads,
                                       RetrievalRnnBuildStats *stats) {
    RetrievalRnnInsert *pending = NULL;
    int pending_count = 0;
    int rc;

    if (!graph || threads <= 0) {
        return -1;
    }

    if (threads <= 1 || graph->num_vectors < 2) {
        rc = rnn_collect_reverse_edges_range(graph,
                                             0,
                                             graph->num_vectors,
                                             &pending,
                                             &pending_count,
                                             stats);
        if (rc != 0) {
            free(pending);
            return -1;
        }
    } else {
        RnnRangeWorker *workers = NULL;
        pthread_t *tids = NULL;
        int launched = 0;
        int i;

        workers = (RnnRangeWorker *)calloc((size_t)threads, sizeof(*workers));
        tids = (pthread_t *)calloc((size_t)threads, sizeof(*tids));
        if (!workers || !tids) {
            free(workers);
            free(tids);
            return -1;
        }

        for (i = 0; i < threads; i++) {
            workers[i].graph = graph;
            workers[i].start = (int)(((int64_t)graph->num_vectors * i) / threads);
            workers[i].end = (int)(((int64_t)graph->num_vectors * (i + 1)) / threads);
            if (pthread_create(&tids[i], NULL,
                               rnn_collect_reverse_edges_worker_fn,
                               &workers[i]) != 0) {
                workers[i].ret = -1;
                break;
            }
            launched++;
        }

        for (i = 0; i < launched; i++) {
            pthread_join(tids[i], NULL);
        }

        rc = (launched == threads) ? 0 : -1;
        for (i = 0; i < threads; i++) {
            if (workers[i].ret != 0) {
                rc = -1;
            }
            rnn_stats_accumulate(stats, &workers[i].stats);
        }
        if (rc == 0 &&
            rnn_concat_worker_pending(workers, threads,
                                      &pending, &pending_count) != 0) {
            rc = -1;
        }
        for (i = 0; i < threads; i++) {
            free(workers[i].pending);
        }
        free(workers);
        free(tids);
        if (rc != 0) {
            free(pending);
            return -1;
        }
    }

    rc = rnn_apply_insertions(graph, pending, pending_count, stats);
    if (stats) {
        if ((uint64_t)pending_count > stats->pending_max) {
            stats->pending_max = (uint64_t)pending_count;
        }
        if (rc > 0) {
            stats->changed_edges += (uint64_t)rc;
        }
    }
    free(pending);
    return rc;
}

int retrieval_rnn_graph_add_reverse_edges_cpu(RetrievalRnnGraph *graph,
                                              RetrievalRnnBuildStats *stats) {
    return rnn_graph_add_reverse_edges(graph, 1, stats);
}

static void rnn_fill_stats(const RetrievalRnnGraph *graph,
                           RetrievalRnnBuildStats *stats) {
    int u;
    int sum_degree = 0;
    int max_degree = 0;
    int isolated = 0;

    if (!graph || !stats) {
        return;
    }

    for (u = 0; u < graph->num_vectors; u++) {
        int degree = graph->degrees[u];
        sum_degree += degree;
        if (degree > max_degree) {
            max_degree = degree;
        }
        if (degree == 0) {
            isolated++;
        }
    }

    stats->max_degree = max_degree;
    stats->isolated_nodes = isolated;
    stats->avg_degree_x100 = (graph->num_vectors > 0)
        ? (int)(((int64_t)sum_degree * 100) / graph->num_vectors)
        : 0;
}

static void rnn_print_progress(const RetrievalRnnGraph *graph,
                               const RetrievalRnnBuildStats *stats,
                               int t1_idx,
                               int t1_total,
                               int t2_idx,
                               int t2_total) {
    RetrievalRnnBuildStats degree_stats;

    retrieval_rnn_build_stats_reset(&degree_stats);
    rnn_fill_stats(graph, &degree_stats);
    printf("  RNN build progress: t1=%d/%d t2=%d/%d pairs=%llu batches=%llu changed=%llu avg_degree=%.2f max_degree=%d isolated=%d\n",
           t1_idx + 1,
           t1_total,
           t2_idx + 1,
           t2_total,
           stats ? (unsigned long long)stats->distance_pairs_requested : 0ULL,
           stats ? (unsigned long long)stats->distance_batches : 0ULL,
           stats ? (unsigned long long)stats->changed_edges : 0ULL,
           (double)degree_stats.avg_degree_x100 / 100.0,
           degree_stats.max_degree,
           degree_stats.isolated_nodes);
    fflush(stdout);
}

static void *rnn_sort_rows_worker_fn(void *arg) {
    RnnRangeWorker *worker = (RnnRangeWorker *)arg;
    int u;

    worker->ret = 0;
    for (u = worker->start; u < worker->end; u++) {
        retrieval_rnn_graph_sort_row(worker->graph, u);
    }
    return NULL;
}

static int rnn_sort_rows(RetrievalRnnGraph *graph, int threads) {
    RnnRangeWorker *workers = NULL;
    pthread_t *tids = NULL;
    int launched = 0;
    int rc = 0;
    int i;

    if (!graph || threads <= 0) {
        return -1;
    }
    if (threads <= 1 || graph->num_vectors < 2) {
        for (i = 0; i < graph->num_vectors; i++) {
            retrieval_rnn_graph_sort_row(graph, i);
        }
        return 0;
    }

    workers = (RnnRangeWorker *)calloc((size_t)threads, sizeof(*workers));
    tids = (pthread_t *)calloc((size_t)threads, sizeof(*tids));
    if (!workers || !tids) {
        free(workers);
        free(tids);
        return -1;
    }

    for (i = 0; i < threads; i++) {
        workers[i].graph = graph;
        workers[i].start = (int)(((int64_t)graph->num_vectors * i) / threads);
        workers[i].end = (int)(((int64_t)graph->num_vectors * (i + 1)) / threads);
        if (pthread_create(&tids[i], NULL,
                           rnn_sort_rows_worker_fn,
                           &workers[i]) != 0) {
            workers[i].ret = -1;
            break;
        }
        launched++;
    }

    for (i = 0; i < launched; i++) {
        pthread_join(tids[i], NULL);
    }

    if (launched != threads) {
        rc = -1;
    }
    for (i = 0; i < threads; i++) {
        if (workers[i].ret != 0) {
            rc = -1;
        }
    }

    free(workers);
    free(tids);
    return rc;
}

int retrieval_rnn_build_graph_ex(const float *base,
                                 int num_vectors,
                                 int dim,
                                 const RetrievalRnnBuildConfig *cfg,
                                 const RetrievalRnnBuildRuntime *runtime,
                                 RetrievalRnnGraph *out_graph,
                                 RetrievalRnnBuildStats *stats) {
    RetrievalRnnGraph graph;
    RnnDistanceRuntime dist_rt;
    pthread_mutex_t distance_mu;
    uint64_t t0;
    int t1;
    int r;
    int threads = 1;
    int have_distance_mu = 0;
    int rc = -1;

    if (!base || !cfg || !out_graph || num_vectors <= 1 || dim <= 0 ||
        cfg->s <= 0 || cfg->r <= 0 || cfg->t1 <= 0 || cfg->t2 <= 0 ||
        cfg->s > cfg->r) {
        return -1;
    }
    if (runtime) {
        if (runtime->threads <= 0 ||
            runtime->batch_size < 0 ||
            runtime->min_batch < 0 ||
            runtime->progress_every_t2 < 0 ||
            runtime->min_batch > runtime->batch_size) {
            return -1;
        }
        threads = runtime->threads;
    }
    if (threads > num_vectors) {
        threads = num_vectors;
    }
    if (threads < 1) {
        threads = 1;
    }

    if (stats) {
        retrieval_rnn_build_stats_reset(stats);
        stats->threads_used = threads;
        stats->distance_provider =
            (runtime && runtime->distance_batch) ? 1 : 0;
    }
    memset(out_graph, 0, sizeof(*out_graph));
    memset(&graph, 0, sizeof(graph));
    memset(&dist_rt, 0, sizeof(dist_rt));
    dist_rt.base = base;
    dist_rt.dim = dim;
    dist_rt.runtime = runtime;
    dist_rt.stats = stats;
    if (runtime && runtime->distance_batch) {
        if (pthread_mutex_init(&distance_mu, NULL) != 0) {
            return -1;
        }
        have_distance_mu = 1;
        dist_rt.distance_mu = &distance_mu;
    }

    if (retrieval_rnn_graph_alloc(&graph, num_vectors, dim, cfg->r) != 0) {
        goto done;
    }

    t0 = rnn_now_us();
    if (rnn_random_graph(&dist_rt, &graph, cfg->s, cfg->seed,
                         threads, stats) != 0) {
        goto done;
    }
    if (stats) {
        stats->init_us += rnn_now_us() - t0;
    }

    for (t1 = 0; t1 < cfg->t1; t1++) {
        int t2;
        t0 = rnn_now_us();
        for (t2 = 0; t2 < cfg->t2; t2++) {
            if (rnn_update_neighbors_once(&dist_rt, &graph,
                                          threads, stats) < 0) {
                goto done;
            }
            if (runtime && runtime->progress_every_t2) {
                rnn_print_progress(&graph, stats, t1, cfg->t1, t2, cfg->t2);
            }
        }
        if (stats) {
            stats->update_us += rnn_now_us() - t0;
        }

        if (t1 + 1 < cfg->t1) {
            t0 = rnn_now_us();
            r = rnn_graph_add_reverse_edges(&graph, threads, stats);
            if (r < 0) {
                goto done;
            }
            if (stats) {
                stats->reverse_us += rnn_now_us() - t0;
            }
        }
    }

    if (rnn_sort_rows(&graph, threads) != 0) {
        goto done;
    }
    rnn_fill_stats(&graph, stats);

    *out_graph = graph;
    memset(&graph, 0, sizeof(graph));
    rc = 0;

done:
    retrieval_rnn_graph_free(&graph);
    if (have_distance_mu) {
        pthread_mutex_destroy(&distance_mu);
    }
    return rc;
}

int retrieval_rnn_build_graph_cpu(const float *base,
                                  int num_vectors,
                                  int dim,
                                  const RetrievalRnnBuildConfig *cfg,
                                  RetrievalRnnGraph *out_graph,
                                  RetrievalRnnBuildStats *stats) {
    RetrievalRnnBuildRuntime runtime;

    memset(&runtime, 0, sizeof(runtime));
    runtime.threads = 1;
    runtime.batch_size = 1;
    runtime.min_batch = 1;

    return retrieval_rnn_build_graph_ex(base,
                                        num_vectors,
                                        dim,
                                        cfg,
                                        &runtime,
                                        out_graph,
                                        stats);
}

static int rnn_write_exact(FILE *fp, const void *ptr, size_t size) {
    if (size == 0) {
        return 0;
    }
    return (fwrite(ptr, 1, size, fp) == size) ? 0 : -1;
}

static int rnn_read_exact(FILE *fp, void *ptr, size_t size) {
    if (size == 0) {
        return 0;
    }
    return (fread(ptr, 1, size, fp) == size) ? 0 : -1;
}

static int rnn_cache_compute_layout(const char *layout_name,
                                    int num_vectors,
                                    int max_degree,
                                    uint64_t *layout_len_out,
                                    uint64_t *degrees_offset_out,
                                    uint64_t *neighbors_offset_out,
                                    uint64_t *neighbor_dists_offset_out,
                                    uint64_t *file_size_out) {
    uint64_t layout_len;
    uint64_t edge_count;
    uint64_t offset;

    if (!layout_name || num_vectors <= 0 || max_degree <= 0 ||
        sizeof(int) != sizeof(int32_t)) {
        return -1;
    }

    layout_len = (uint64_t)strlen(layout_name);
    if (layout_len > (uint64_t)INT32_MAX) {
        return -1;
    }

    edge_count = (uint64_t)(uint32_t)num_vectors * (uint64_t)(uint32_t)max_degree;
    if (edge_count / (uint64_t)(uint32_t)max_degree !=
        (uint64_t)(uint32_t)num_vectors) {
        return -1;
    }

    offset = (uint64_t)sizeof(RetrievalRnnGraphCacheHeader) + layout_len;
    *degrees_offset_out = offset;
    offset += (uint64_t)(uint32_t)num_vectors * (uint64_t)sizeof(int);
    *neighbors_offset_out = offset;
    offset += edge_count * (uint64_t)sizeof(int);
    *neighbor_dists_offset_out = offset;
    offset += edge_count * (uint64_t)sizeof(float);

    *layout_len_out = layout_len;
    *file_size_out = offset;
    return 0;
}

static int rnn_cache_header_matches(const RetrievalRnnGraphCacheHeader *hdr,
                                    const char *layout_name,
                                    int num_vectors,
                                    int dim,
                                    const RetrievalRnnBuildConfig *cfg,
                                    uint32_t cache_version) {
    uint64_t layout_len;
    uint64_t degrees_offset;
    uint64_t neighbors_offset;
    uint64_t neighbor_dists_offset;
    uint64_t file_size;

    if (!hdr || !cfg ||
        rnn_cache_compute_layout(layout_name, num_vectors, cfg->r,
                                 &layout_len, &degrees_offset,
                                 &neighbors_offset, &neighbor_dists_offset,
                                 &file_size) != 0) {
        return 0;
    }

    return hdr->magic == RNN_GRAPH_CACHE_MAGIC &&
           hdr->version == cache_version &&
           hdr->num_vectors == num_vectors &&
           hdr->dim == dim &&
           hdr->build_s == cfg->s &&
           hdr->build_r == cfg->r &&
           hdr->build_t1 == cfg->t1 &&
           hdr->build_t2 == cfg->t2 &&
           hdr->seed == cfg->seed &&
           hdr->distance_source == RNN_GRAPH_CACHE_DISTANCE_SOURCE_FLOAT &&
           hdr->layout_name_len == (int32_t)layout_len &&
           hdr->degrees_offset == degrees_offset &&
           hdr->neighbors_offset == neighbors_offset &&
           hdr->neighbor_dists_offset == neighbor_dists_offset &&
           hdr->file_size == file_size;
}

static int rnn_validate_loaded_graph(const RetrievalRnnGraph *graph) {
    int u;

    if (!graph || !graph->degrees || !graph->neighbors ||
        !graph->neighbor_dists || graph->num_vectors <= 0 ||
        graph->dim <= 0 || graph->max_degree <= 0) {
        return -1;
    }

    for (u = 0; u < graph->num_vectors; u++) {
        int i;
        int degree = graph->degrees[u];
        if (degree < 0 || degree > graph->max_degree) {
            return -1;
        }
        for (i = 0; i < degree; i++) {
            int v = graph->neighbors[rnn_slot(graph, u, i)];
            float dist = graph->neighbor_dists[rnn_slot(graph, u, i)];
            if (v < 0 || v >= graph->num_vectors || v == u || dist < 0.0f) {
                return -1;
            }
        }
    }

    return 0;
}

int retrieval_rnn_graph_save_cache(const char *path,
                                   const char *layout_name,
                                   const RetrievalRnnGraph *graph,
                                   const RetrievalRnnBuildConfig *cfg,
                                   uint32_t cache_version) {
    RetrievalRnnGraphCacheHeader hdr;
    uint64_t layout_len;
    uint64_t degrees_offset;
    uint64_t neighbors_offset;
    uint64_t neighbor_dists_offset;
    uint64_t file_size;
    uint64_t edge_count;
    char *tmp_path;
    size_t tmp_len;
    FILE *fp;
    int saved_errno = 0;
    int rc = -1;

    if (!path || !layout_name || !graph || !cfg ||
        !graph->degrees || !graph->neighbors || !graph->neighbor_dists ||
        graph->num_vectors <= 0 || graph->dim <= 0 || graph->max_degree <= 0 ||
        cfg->s <= 0 || cfg->r <= 0 || cfg->t1 <= 0 || cfg->t2 <= 0 ||
        graph->max_degree != cfg->r ||
        rnn_validate_loaded_graph(graph) != 0 ||
        rnn_cache_compute_layout(layout_name, graph->num_vectors, graph->max_degree,
                                 &layout_len, &degrees_offset,
                                 &neighbors_offset, &neighbor_dists_offset,
                                 &file_size) != 0) {
        return -1;
    }

    tmp_len = strlen(path) + 5;
    tmp_path = (char *)malloc(tmp_len);
    if (!tmp_path) {
        return -1;
    }
    snprintf(tmp_path, tmp_len, "%s.tmp", path);

    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = RNN_GRAPH_CACHE_MAGIC;
    hdr.version = cache_version;
    hdr.num_vectors = graph->num_vectors;
    hdr.dim = graph->dim;
    hdr.build_s = cfg->s;
    hdr.build_r = cfg->r;
    hdr.build_t1 = cfg->t1;
    hdr.build_t2 = cfg->t2;
    hdr.seed = cfg->seed;
    hdr.distance_source = RNN_GRAPH_CACHE_DISTANCE_SOURCE_FLOAT;
    hdr.layout_name_len = (int32_t)layout_len;
    hdr.degrees_offset = degrees_offset;
    hdr.neighbors_offset = neighbors_offset;
    hdr.neighbor_dists_offset = neighbor_dists_offset;
    hdr.file_size = file_size;

    edge_count = (uint64_t)(uint32_t)graph->num_vectors *
                 (uint64_t)(uint32_t)graph->max_degree;
    fp = fopen(tmp_path, "wb");
    if (!fp) {
        saved_errno = errno;
        fprintf(stderr,
                "  ERROR: graph cache fopen failed: tmp=%s errno=%d (%s)\n",
                tmp_path,
                saved_errno,
                strerror(saved_errno));
        free(tmp_path);
        return -1;
    }

    if (rnn_write_exact(fp, &hdr, sizeof(hdr)) == 0 &&
        rnn_write_exact(fp, layout_name, (size_t)layout_len) == 0 &&
        rnn_write_exact(fp, graph->degrees,
                        (size_t)graph->num_vectors * sizeof(int)) == 0 &&
        rnn_write_exact(fp, graph->neighbors,
                        (size_t)edge_count * sizeof(int)) == 0 &&
        rnn_write_exact(fp, graph->neighbor_dists,
                        (size_t)edge_count * sizeof(float)) == 0 &&
        fflush(fp) == 0) {
        rc = 0;
    }
    if (rc != 0) {
        saved_errno = errno;
        fprintf(stderr,
                "  ERROR: graph cache write failed: tmp=%s expected=%llu bytes errno=%d (%s) ferror=%d\n",
                tmp_path,
                (unsigned long long)file_size,
                saved_errno,
                strerror(saved_errno),
                ferror(fp));
    }

    if (fclose(fp) != 0) {
        saved_errno = errno;
        fprintf(stderr,
                "  ERROR: graph cache fclose failed: tmp=%s errno=%d (%s)\n",
                tmp_path,
                saved_errno,
                strerror(saved_errno));
        rc = -1;
    }
    if (rc == 0 && rename(tmp_path, path) != 0) {
        saved_errno = errno;
        fprintf(stderr,
                "  ERROR: graph cache rename failed: tmp=%s path=%s errno=%d (%s)\n",
                tmp_path,
                path,
                saved_errno,
                strerror(saved_errno));
        rc = -1;
    }
    if (rc != 0) {
        remove(tmp_path);
    }

    free(tmp_path);
    return rc;
}

int retrieval_rnn_graph_load_cache(const char *path,
                                   const char *layout_name,
                                   int num_vectors,
                                   int dim,
                                   const RetrievalRnnBuildConfig *cfg,
                                   uint32_t cache_version,
                                   RetrievalRnnGraph *out_graph) {
    RetrievalRnnGraphCacheHeader hdr;
    RetrievalRnnGraph graph;
    uint64_t layout_len;
    uint64_t degrees_offset;
    uint64_t neighbors_offset;
    uint64_t neighbor_dists_offset;
    uint64_t file_size;
    uint64_t edge_count;
    char *cached_layout = NULL;
    FILE *fp = NULL;
    int rc = -1;

    if (!path || !layout_name || !cfg || !out_graph ||
        num_vectors <= 0 || dim <= 0 ||
        cfg->s <= 0 || cfg->r <= 0 || cfg->t1 <= 0 || cfg->t2 <= 0 ||
        cfg->s > cfg->r ||
        rnn_cache_compute_layout(layout_name, num_vectors, cfg->r,
                                 &layout_len, &degrees_offset,
                                 &neighbors_offset, &neighbor_dists_offset,
                                 &file_size) != 0) {
        return -1;
    }

    memset(&graph, 0, sizeof(graph));
    fp = fopen(path, "rb");
    if (!fp) {
        return -1;
    }

    if (rnn_read_exact(fp, &hdr, sizeof(hdr)) != 0 ||
        !rnn_cache_header_matches(&hdr, layout_name, num_vectors, dim,
                                  cfg, cache_version)) {
        fclose(fp);
        return -1;
    }

    cached_layout = (char *)malloc((size_t)layout_len + 1u);
    if (!cached_layout) {
        fclose(fp);
        return -1;
    }
    if (rnn_read_exact(fp, cached_layout, (size_t)layout_len) != 0) {
        goto done;
    }
    cached_layout[layout_len] = '\0';
    if (memcmp(cached_layout, layout_name, (size_t)layout_len) != 0) {
        goto done;
    }

    if (rnn_graph_alloc_storage(&graph, num_vectors, dim, cfg->r, 0) != 0) {
        goto done;
    }

    edge_count = (uint64_t)(uint32_t)num_vectors * (uint64_t)(uint32_t)cfg->r;
    if (rnn_read_exact(fp, graph.degrees,
                       (size_t)num_vectors * sizeof(int)) != 0 ||
        rnn_read_exact(fp, graph.neighbors,
                       (size_t)edge_count * sizeof(int)) != 0 ||
        rnn_read_exact(fp, graph.neighbor_dists,
                       (size_t)edge_count * sizeof(float)) != 0 ||
        fgetc(fp) != EOF ||
        rnn_validate_loaded_graph(&graph) != 0) {
        goto done;
    }

    *out_graph = graph;
    memset(&graph, 0, sizeof(graph));
    rc = 0;

done:
    retrieval_rnn_graph_free(&graph);
    free(cached_layout);
    fclose(fp);
    return rc;
}
