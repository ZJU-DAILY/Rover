#include "dsp_demo/retrieval_search_build_core.h"

#include <float.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <errno.h>
#include <math.h>


typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    uint64_t issued, completed, total, boundary;
    int active, failed;
    const RetrievalSearchBuildRuntime *runtime;
    uint64_t rest_us, events;
} SearchBuildGate;

static uint64_t sb_monotonic_us(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000u + t.tv_nsec / 1000u;
}
static void sb_gate_abort(SearchBuildGate *g) {
    if (!g) return;
    pthread_mutex_lock(&g->mu); g->failed=1;
    pthread_cond_broadcast(&g->cv); pthread_mutex_unlock(&g->mu);
}
static int sb_gate_enter(SearchBuildGate *g) {
    if (!g) return 0;
    pthread_mutex_lock(&g->mu);
    while (!g->failed && g->issued >= g->boundary)
        pthread_cond_wait(&g->cv, &g->mu);
    if (g->failed) { pthread_mutex_unlock(&g->mu); return -1; }
    ++g->issued; ++g->active; pthread_mutex_unlock(&g->mu); return 0;
}
static int sb_gate_finish(SearchBuildGate *g) {
    if (!g) return 0;
    pthread_mutex_lock(&g->mu); --g->active; ++g->completed;
    if (!g->failed && !g->active && g->completed==g->boundary && g->completed<g->total) {
        const RetrievalSearchBuildRuntime *rt=g->runtime;
        uint64_t start=0,end=0;
        struct timespec req={rt->cooldown_us/1000000u,(rt->cooldown_us%1000000u)*1000u};
        if(rt->cooldown_quiesce && rt->cooldown_quiesce(rt->cooldown_quiesce_ctx,1)) g->failed=1;
        if(!g->failed) {
            int rc;start=sb_monotonic_us();
            do { rc=nanosleep(&req,&req); } while(rc && errno==EINTR);
            end=sb_monotonic_us();
            if(rc) g->failed=1;
        }
        if(rt->cooldown_quiesce && rt->cooldown_quiesce(rt->cooldown_quiesce_ctx,0)) g->failed=1;
        if(!g->failed) {
            g->rest_us+=end-start;++g->events;
            if(rt->cooldown_event)rt->cooldown_event(rt->cooldown_ctx,g->completed,start,end);
            g->boundary+=rt->cooldown_every_queries;
        }
        pthread_cond_broadcast(&g->cv);
    }
    int ret=g->failed ? -1 : 0;
    pthread_mutex_unlock(&g->mu);return ret;
}

static float sb_distance(const RetrievalSearchBuildRuntime *rt,
                         const float *base,int dim,int32_t a,int32_t b) {
    float d;
    if (!rt || !rt->distance_batch) return retrieval_rnn_l2_distance_cpu(base,dim,a,b);
    if (rt->distance_batch(rt->distance_ctx,&a,&b,1,&d) || !isfinite(d)) return NAN;
    return d;
}

typedef struct SearchBuildWorker {
    const float *base;
    int num_vectors;
    int dim;
    const RetrievalSearchBuildConfig *cfg;
    const RetrievalSearchBuildRuntime *runtime;
    const RetrievalRnnGraph *old_graph;
    RetrievalRnnGraph *new_graph;
    RetrievalSearchBuildStats stats;
    SearchBuildGate *gate;
    int start;
    int end;
    int ret;
    int old_scores_valid;
    int *failed;
} SearchBuildWorker;

typedef struct SearchBuildPoolItem {
    int32_t id;
    float dist;
    uint8_t expanded;
} SearchBuildPoolItem;

static uint64_t sb_now_us(void) { return sb_monotonic_us(); }

static size_t sb_slot(const RetrievalRnnGraph *graph, int src, int pos) {
    return (size_t)src * (size_t)graph->max_degree + (size_t)pos;
}

static uint32_t sb_lcg_next(uint32_t x) {
    return x * 1664525u + 1013904223u;
}

static uint32_t sb_seed_for_node(uint32_t seed, int node) {
    uint32_t x = seed ^ (uint32_t)(node + 1) * 0x9e3779b9u;
    x = sb_lcg_next(x);
    x ^= x >> 16;
    return sb_lcg_next(x);
}

static int sb_candidate_compare(const void *a, const void *b) {
    const RetrievalSearchBuildCandidate *ca =
        (const RetrievalSearchBuildCandidate *)a;
    const RetrievalSearchBuildCandidate *cb =
        (const RetrievalSearchBuildCandidate *)b;
    if (ca->exact_dist < cb->exact_dist) {
        return -1;
    }
    if (ca->exact_dist > cb->exact_dist) {
        return 1;
    }
    if (ca->id != cb->id) {
        return (ca->id < cb->id) ? -1 : 1;
    }
    return 0;
}

static int sb_pool_item_better(float new_dist,
                               int new_id,
                               float old_dist,
                               int old_id) {
    if (new_dist < old_dist) {
        return 1;
    }
    if (new_dist > old_dist) {
        return 0;
    }
    return new_id < old_id;
}

static void sb_stats_accumulate(RetrievalSearchBuildStats *dst,
                                const RetrievalSearchBuildStats *src) {
    if (!dst || !src) {
        return;
    }
    dst->init_us += src->init_us;
    dst->search_us += src->search_us;
    dst->prune_us += src->prune_us;
    dst->refine_us += src->refine_us;
    dst->export_us += src->export_us;
    dst->searched_queries += src->searched_queries;
    dst->candidates_seen += src->candidates_seen;
    dst->candidates_kept += src->candidates_kept;
    dst->exact_distance_pairs += src->exact_distance_pairs;
    dst->duplicate_rejects += src->duplicate_rejects;
    dst->self_rejects += src->self_rejects;
    dst->rows_filled += src->rows_filled;
    dst->fill_edges += src->fill_edges;
    dst->scores_reused += src->scores_reused;
    dst->scores_computed += src->scores_computed;
    dst->score_us += src->score_us;
    dst->row_vector_loads += src->row_vector_loads;
    dst->row_read_us += src->row_read_us;
    dst->pair_distance_us += src->pair_distance_us;
    dst->reverse_rows += src->reverse_rows;
    dst->reverse_edges_seen += src->reverse_edges_seen;
    dst->reverse_candidates_dropped += src->reverse_candidates_dropped;
}

void retrieval_search_build_stats_reset(RetrievalSearchBuildStats *stats) {
    if (stats) {
        memset(stats, 0, sizeof(*stats));
    }
}

static int sb_validate(const float *base,
                       int num_vectors,
                       int dim,
                       const RetrievalSearchBuildConfig *cfg,
                       const RetrievalSearchBuildRuntime *runtime,
                       RetrievalRnnGraph *out_graph) {
    if (!cfg || !runtime || (!base && (!runtime->distance_batch || runtime->search==retrieval_search_build_exact_provider || runtime->search==retrieval_search_build_graph_provider)) || !out_graph ||
        (runtime->cooldown_every_queries && runtime->search_pair) ||
        num_vectors <= 1 || dim <= 0 ||
        cfg->r <= 0 || cfg->r >= num_vectors ||
        cfg->iterations <= 0 ||
        cfg->candidate_count < cfg->r ||
        cfg->candidate_count <= 0 ||
        cfg->entry_points <= 0 ||
        cfg->expand_count <= 0 ||
        cfg->pool_size <= 0 ||
        cfg->pool_size < cfg->candidate_count ||
        cfg->batch_align <= 0 ||
        (cfg->r % cfg->batch_align) != 0 ||
        cfg->prune_mode < 0 || cfg->prune_mode > 1 ||
        cfg->candidate_count > INT_MAX - cfg->r ||
        runtime->reverse_candidate_count < 0 ||
        runtime->reverse_candidate_count > cfg->candidate_count ||
        (runtime->reverse_candidate_count && !runtime->score_candidates) ||
        (runtime->score_candidates && runtime->search_pair) ||
        (runtime->publish_row && (!runtime->score_candidates || !runtime->prepare_iteration)) ||
        (runtime->row_vector_buffer && !base && !runtime->read_vector) ||
        runtime->threads <= 0 ||
        (!runtime->search && !runtime->search_pair)) {
        return -1;
    }
    return 0;
}

static void sb_fill_graph_stats(const RetrievalRnnGraph *graph,
                                RetrievalSearchBuildStats *stats) {
    int u;
    int sum = 0;
    int min_degree = INT_MAX;
    int max_degree = 0;
    int isolated = 0;

    if (!graph || !stats) {
        return;
    }
    for (u = 0; u < graph->num_vectors; u++) {
        int degree = graph->degrees[u];
        sum += degree;
        if (degree < min_degree) {
            min_degree = degree;
        }
        if (degree > max_degree) {
            max_degree = degree;
        }
        if (degree == 0) {
            isolated++;
        }
    }
    stats->avg_degree_x100 = (graph->num_vectors > 0)
        ? (int)(((int64_t)sum * 100) / graph->num_vectors)
        : 0;
    stats->min_degree = (min_degree == INT_MAX) ? 0 : min_degree;
    stats->max_degree = max_degree;
    stats->isolated_nodes = isolated;
}

static void sb_sort_row(RetrievalRnnGraph *graph, int src) {
    retrieval_rnn_graph_sort_row(graph, src);
}

static int sb_row_contains(const RetrievalRnnGraph *graph, int src, int dst) {
    int i;
    for (i = 0; i < graph->degrees[src]; i++) {
        if (graph->neighbors[sb_slot(graph, src, i)] == dst) {
            return 1;
        }
    }
    return 0;
}

static int sb_set_row_edge(RetrievalRnnGraph *graph,
                           int src,
                           int pos,
                           int dst,
                           float dist) {
    size_t slot;
    if (!graph || src < 0 || src >= graph->num_vectors ||
        pos < 0 || pos >= graph->max_degree ||
        dst < 0 || dst >= graph->num_vectors || src == dst) {
        return -1;
    }
    slot = sb_slot(graph, src, pos);
    graph->neighbors[slot] = dst;
    graph->neighbor_dists[slot] = dist;
    if (graph->is_new) {
        graph->is_new[slot] = 1u;
    }
    if (graph->degrees[src] <= pos) {
        graph->degrees[src] = pos + 1;
    }
    return 0;
}

static int sb_candidate_add(RetrievalSearchBuildCandidate *items,
                            int cap,
                            int *count,
                            int id,
                            float search_dist,
                            RetrievalSearchBuildStats *stats) {
    int i;
    if (!items || !count || cap <= 0 || id < 0) {
        return -1;
    }
    for (i = 0; i < *count; i++) {
        if (items[i].id == id) {
            if (stats) {
                stats->duplicate_rejects++;
            }
            if (search_dist < items[i].search_dist) {
                items[i].search_dist = search_dist;
            }
            return 0;
        }
    }
    if (*count >= cap) {
        return 0;
    }
    items[*count].id = id;
    items[*count].search_dist = search_dist;
    items[*count].exact_dist = FLT_MAX;
    (*count)++;
    return 1;
}

int retrieval_search_build_exact_provider(
    void *ctx,
    const float *base,
    int num_vectors,
    int dim,
    const RetrievalSearchBuildConfig *cfg,
    const RetrievalRnnGraph *graph_snapshot,
    int query_id,
    int candidate_cap,
    RetrievalSearchBuildCandidate *out_candidates,
    int *out_count) {
    RetrievalSearchBuildCandidate *all = NULL;
    int count = 0;
    int i;

    (void)ctx;
    (void)cfg;
    (void)graph_snapshot;
    if (!base || num_vectors <= 1 || dim <= 0 ||
        query_id < 0 || query_id >= num_vectors ||
        candidate_cap <= 0 || !out_candidates || !out_count) {
        return -1;
    }

    all = (RetrievalSearchBuildCandidate *)malloc(
        (size_t)(num_vectors - 1) * sizeof(*all));
    if (!all) {
        return -1;
    }
    for (i = 0; i < num_vectors; i++) {
        if (i == query_id) {
            continue;
        }
        all[count].id = (int32_t)i;
        all[count].search_dist =
            retrieval_rnn_l2_distance_cpu(base, dim, query_id, i);
        all[count].exact_dist = all[count].search_dist;
        count++;
    }
    qsort(all, (size_t)count, sizeof(*all), sb_candidate_compare);
    if (count > candidate_cap) {
        count = candidate_cap;
    }
    memcpy(out_candidates, all, (size_t)count * sizeof(*out_candidates));
    *out_count = count;
    free(all);
    return 0;
}

static int sb_next_power_of_two_size(size_t need, size_t *out) {
    size_t cap = 1u;

    if (!out || need == 0u) {
        return -1;
    }
    while (cap < need) {
        if (cap > ((size_t)-1) / 2u) {
            return -1;
        }
        cap <<= 1u;
    }
    *out = cap;
    return 0;
}

static uint32_t sb_hash_u32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static int sb_visited_test_or_add(int *table, size_t table_cap, int id) {
    size_t mask;
    size_t pos;
    size_t probes;

    if (!table || table_cap == 0u || (table_cap & (table_cap - 1u)) != 0u ||
        id < 0) {
        return -1;
    }
    mask = table_cap - 1u;
    pos = (size_t)sb_hash_u32((uint32_t)id) & mask;
    for (probes = 0u; probes < table_cap; probes++) {
        int value = table[pos];
        if (value == id) {
            return 1;
        }
        if (value < 0) {
            table[pos] = id;
            return 0;
        }
        pos = (pos + 1u) & mask;
    }
    return -1;
}

static int sb_pool_insert(SearchBuildPoolItem *pool,
                          int *pool_size,
                          int pool_cap,
                          int id,
                          float dist) {
    int pos;

    if (!pool || !pool_size || pool_cap <= 0 || id < 0) {
        return -1;
    }
    if (*pool_size >= pool_cap &&
        !sb_pool_item_better(dist, id,
                             pool[pool_cap - 1].dist,
                             pool[pool_cap - 1].id)) {
        return 0;
    }

    if (*pool_size < pool_cap) {
        pos = *pool_size;
        (*pool_size)++;
    } else {
        pos = pool_cap - 1;
    }

    while (pos > 0 &&
           sb_pool_item_better(dist, id, pool[pos - 1].dist,
                               pool[pos - 1].id)) {
        pool[pos] = pool[pos - 1];
        pos--;
    }

    pool[pos].id = (int32_t)id;
    pool[pos].dist = dist;
    pool[pos].expanded = 0u;
    return 1;
}

static int sb_graph_provider_add_entry(const float *base,
                                       int num_vectors,
                                       int dim,
                                       int query_id,
                                       int entry_id,
                                       SearchBuildPoolItem *pool,
                                       int *pool_size,
                                       int pool_cap,
                                       int *visited,
                                       size_t visited_cap) {
    float dist;
    int seen;

    if (entry_id < 0 || entry_id >= num_vectors || entry_id == query_id) {
        return 0;
    }
    seen = sb_visited_test_or_add(visited, visited_cap, entry_id);
    if (seen != 0) {
        return (seen > 0) ? 0 : -1;
    }
    dist = retrieval_rnn_l2_distance_cpu(base, dim, query_id, entry_id);
    return (sb_pool_insert(pool, pool_size, pool_cap, entry_id, dist) < 0)
        ? -1
        : 0;
}

int retrieval_search_build_graph_provider(
    void *ctx,
    const float *base,
    int num_vectors,
    int dim,
    const RetrievalSearchBuildConfig *cfg,
    const RetrievalRnnGraph *graph_snapshot,
    int query_id,
    int candidate_cap,
    RetrievalSearchBuildCandidate *out_candidates,
    int *out_count) {
    SearchBuildPoolItem *pool = NULL;
    int *visited = NULL;
    uint32_t state;
    size_t visit_budget;
    size_t visited_cap;
    int pool_size = 0;
    int entries_added = 0;
    int expand_limit;
    int expanded = 0;
    int attempts = 0;
    int rc = -1;
    int i;

    (void)ctx;
    if (!base || !cfg || !graph_snapshot || !out_candidates || !out_count ||
        num_vectors <= 1 || dim <= 0 ||
        graph_snapshot->num_vectors != num_vectors ||
        graph_snapshot->max_degree <= 0 ||
        query_id < 0 || query_id >= num_vectors ||
        candidate_cap <= 0 ||
        cfg->entry_points <= 0 ||
        cfg->expand_count <= 0 ||
        cfg->pool_size <= 0) {
        return -1;
    }

    expand_limit = cfg->expand_count;
    if (expand_limit > graph_snapshot->max_degree) {
        expand_limit = graph_snapshot->max_degree;
    }
    if (expand_limit <= 0) {
        return -1;
    }

    if ((size_t)cfg->pool_size >
        (((size_t)-1) / (size_t)expand_limit)) {
        return -1;
    }
    visit_budget = (size_t)cfg->pool_size * (size_t)expand_limit +
                   (size_t)cfg->entry_points + 2u;
    if (visit_budget < (size_t)cfg->pool_size ||
        visit_budget > ((size_t)-1) / 2u) {
        return -1;
    }
    if (visit_budget < 16u) {
        visit_budget = 16u;
    }
    if (sb_next_power_of_two_size(visit_budget * 2u, &visited_cap) != 0) {
        return -1;
    }
    if (visited_cap > (size_t)INT_MAX) {
        return -1;
    }

    pool = (SearchBuildPoolItem *)malloc((size_t)cfg->pool_size *
                                         sizeof(*pool));
    visited = (int *)malloc(visited_cap * sizeof(*visited));
    if (!pool || !visited) {
        goto done;
    }
    for (i = 0; i < (int)visited_cap; i++) {
        visited[i] = -1;
    }
    if (sb_visited_test_or_add(visited, visited_cap, query_id) < 0) {
        goto done;
    }

    state = sb_seed_for_node(cfg->seed ^ 0x6d2b79f5u, query_id);
    while (entries_added < cfg->entry_points &&
           attempts < num_vectors * 4) {
        int entry_id;
        state = sb_lcg_next(state);
        entry_id = (int)(state % (uint32_t)num_vectors);
        if (entry_id != query_id) {
            int before = pool_size;
            if (sb_graph_provider_add_entry(base,
                                            num_vectors,
                                            dim,
                                            query_id,
                                            entry_id,
                                            pool,
                                            &pool_size,
                                            cfg->pool_size,
                                            visited,
                                            visited_cap) != 0) {
                goto done;
            }
            if (pool_size > before) {
                entries_added++;
            }
        }
        attempts++;
    }
    for (i = 0; entries_added < cfg->entry_points && i < num_vectors; i++) {
        int before = pool_size;
        if (i == query_id) {
            continue;
        }
        if (sb_graph_provider_add_entry(base,
                                        num_vectors,
                                        dim,
                                        query_id,
                                        i,
                                        pool,
                                        &pool_size,
                                        cfg->pool_size,
                                        visited,
                                        visited_cap) != 0) {
            goto done;
        }
        if (pool_size > before) {
            entries_added++;
        }
    }

    while (expanded < cfg->pool_size) {
        int current_pos = -1;
        int current_id;
        int degree;
        int limit;
        int j;

        for (i = 0; i < pool_size; i++) {
            if (!pool[i].expanded) {
                current_pos = i;
                break;
            }
        }
        if (current_pos < 0) {
            break;
        }

        pool[current_pos].expanded = 1u;
        current_id = pool[current_pos].id;
        expanded++;
        if (current_id < 0 || current_id >= num_vectors) {
            goto done;
        }

        degree = graph_snapshot->degrees[current_id];
        if (degree < 0 || degree > graph_snapshot->max_degree) {
            goto done;
        }
        limit = degree;
        if (limit > expand_limit) {
            limit = expand_limit;
        }
        for (j = 0; j < limit; j++) {
            size_t slot = sb_slot(graph_snapshot, current_id, j);
            int neighbor_id = graph_snapshot->neighbors[slot];
            int seen;
            float dist;

            if (neighbor_id < 0 || neighbor_id >= num_vectors) {
                goto done;
            }
            if (neighbor_id == query_id) {
                continue;
            }
            seen = sb_visited_test_or_add(visited, visited_cap, neighbor_id);
            if (seen < 0) {
                goto done;
            }
            if (seen > 0) {
                continue;
            }
            dist = retrieval_rnn_l2_distance_cpu(base, dim, query_id,
                                                 neighbor_id);
            if (sb_pool_insert(pool,
                               &pool_size,
                               cfg->pool_size,
                               neighbor_id,
                               dist) < 0) {
                goto done;
            }
        }
    }

    if (pool_size > candidate_cap) {
        pool_size = candidate_cap;
    }
    for (i = 0; i < pool_size; i++) {
        out_candidates[i].id = pool[i].id;
        out_candidates[i].search_dist = pool[i].dist;
        out_candidates[i].exact_dist = pool[i].dist;
    }
    *out_count = pool_size;
    rc = 0;

done:
    free(pool);
    free(visited);
    return rc;
}

static int sb_random_init_graph(const float *base,
                                int num_vectors,
                                int dim,
                                const RetrievalSearchBuildConfig *cfg,
                                const RetrievalSearchBuildRuntime *runtime,
                                RetrievalRnnGraph *graph,
                                RetrievalSearchBuildStats *stats) {
    RetrievalSearchBuildCandidate *row = NULL;
    int u;

    row = (RetrievalSearchBuildCandidate *)malloc(
        (size_t)cfg->r * sizeof(*row));
    if (!row) {
        return -1;
    }

    for (u = 0; u < num_vectors; u++) {
        uint32_t state = sb_seed_for_node(cfg->seed, u);
        int row_count = 0;
        uint64_t attempts = 0;
        uint64_t attempt_limit = (uint64_t)(uint32_t)num_vectors *
                                 (uint64_t)(uint32_t)num_vectors;
        while (row_count < cfg->r && attempts < attempt_limit) {
            int dup = 0;
            int v;
            int i;
            state = sb_lcg_next(state);
            v = (int)(state % (uint32_t)num_vectors);
            if (v == u) {
                if (stats) {
                    stats->self_rejects++;
                }
                attempts++;
                continue;
            }
            for (i = 0; i < row_count; i++) {
                if (row[i].id == v) {
                    dup = 1;
                    break;
                }
            }
            if (!dup) {
                row[row_count].id = (int32_t)v;
                row[row_count].search_dist = 0.0f;
                row[row_count].exact_dist =
                    sb_distance(runtime, base, dim, u, v);
                if (!isfinite(row[row_count].exact_dist)) { free(row); return -1; }
                row_count++;
                if (stats) {
                    stats->exact_distance_pairs++;
                }
            } else if (stats) {
                stats->duplicate_rejects++;
            }
            attempts++;
        }
        if (row_count < cfg->r) {
            int v;
            for (v = 0; v < num_vectors && row_count < cfg->r; v++) {
                int dup = 0;
                int i;
                if (v == u) {
                    continue;
                }
                for (i = 0; i < row_count; i++) {
                    if (row[i].id == v) {
                        dup = 1;
                        break;
                    }
                }
                if (!dup) {
                    row[row_count].id = (int32_t)v;
                    row[row_count].search_dist = 0.0f;
                    row[row_count].exact_dist =
                        sb_distance(runtime, base, dim, u, v);
                    if (!isfinite(row[row_count].exact_dist)) { free(row); return -1; }
                row_count++;
                    if (stats) {
                        stats->exact_distance_pairs++;
                        stats->fill_edges++;
                    }
                }
            }
        }
        qsort(row, (size_t)row_count, sizeof(*row), sb_candidate_compare);
        for (int i = 0; i < row_count; i++) {
            if (sb_set_row_edge(graph, u, i, row[i].id,
                                row[i].exact_dist) != 0) {
                free(row);
                return -1;
            }
        }
        sb_sort_row(graph, u);
    }

    free(row);
    return 0;
}

static int sb_fill_missing_edges(const float *base,
                                 int dim,
                                 const RetrievalSearchBuildConfig *cfg,
                                const RetrievalSearchBuildRuntime *runtime,
                                 RetrievalRnnGraph *graph,
                                 int u,
                                 RetrievalSearchBuildStats *stats) {
    uint32_t state = sb_seed_for_node(cfg->seed ^ 0xa5a55a5au, u);
    uint64_t attempts = 0;
    uint64_t attempt_limit = (uint64_t)(uint32_t)graph->num_vectors *
                             (uint64_t)(uint32_t)graph->num_vectors;
    int filled_any = 0;

    while (graph->degrees[u] < cfg->r && attempts < attempt_limit) {
        int v;
        state = sb_lcg_next(state);
        v = (int)(state % (uint32_t)graph->num_vectors);
        if (v != u && !sb_row_contains(graph, u, v)) {
            float dist = sb_distance(runtime, base, dim, u, v);
            if (!isfinite(dist) || sb_set_row_edge(graph, u, graph->degrees[u], v, dist) != 0) {
                return -1;
            }
            if (stats) {
                stats->fill_edges++;
                stats->exact_distance_pairs++;
            }
            filled_any = 1;
        }
        attempts++;
    }
    if (graph->degrees[u] < cfg->r) {
        int v;
        for (v = 0; v < graph->num_vectors && graph->degrees[u] < cfg->r; v++) {
            float dist;
            if (v == u || sb_row_contains(graph, u, v)) {
                continue;
            }
            dist = sb_distance(runtime, base, dim, u, v);
            if (!isfinite(dist) || sb_set_row_edge(graph, u, graph->degrees[u], v, dist) != 0) {
                return -1;
            }
            if (stats) {
                stats->fill_edges++;
                stats->exact_distance_pairs++;
            }
            filled_any = 1;
        }
    }
    if (filled_any && stats) {
        stats->rows_filled++;
    }
    return (graph->degrees[u] == cfg->r) ? 0 : -1;
}

static int sb_write_pruned_row(const float *base,
                               int num_vectors,
                               int dim,
                               const RetrievalSearchBuildConfig *cfg,
                                const RetrievalSearchBuildRuntime *runtime,
                               const RetrievalRnnGraph *old_graph,
                               RetrievalRnnGraph *new_graph,
                               int u,
                               RetrievalSearchBuildCandidate *provider_items,
                               int provider_count,
                               RetrievalSearchBuildStats *stats) {
    int cap = cfg->candidate_count + cfg->r;
    RetrievalSearchBuildCandidate *items = NULL;
    int count = 0;
    int i;

    items = (RetrievalSearchBuildCandidate *)malloc(
        (size_t)cap * sizeof(*items));
    if (!items) {
        return -1;
    }

    for (i = 0; i < provider_count; i++) {
        int id = provider_items[i].id;
        if (id == u) {
            if (stats) {
                stats->self_rejects++;
            }
            continue;
        }
        if (id < 0 || id >= num_vectors) {
            continue;
        }
        if (sb_candidate_add(items, cap, &count, id,
                             provider_items[i].search_dist,
                             stats) < 0) {
            free(items);
            return -1;
        }
    }
    if (old_graph) {
        for (i = 0; i < old_graph->degrees[u]; i++) {
            size_t slot = sb_slot(old_graph, u, i);
            int id = old_graph->neighbors[slot];
            if (id >= 0 && id < num_vectors && id != u) {
                if (sb_candidate_add(items, cap, &count, id,
                                     old_graph->neighbor_dists[slot],
                                     stats) < 0) {
                    free(items);
                    return -1;
                }
            }
        }
    }

    for (i = 0; i < count; i++) {
        items[i].exact_dist = sb_distance(runtime, base, dim, u,
                                                            items[i].id);
        if (!isfinite(items[i].exact_dist)) { free(items); return -1; }
        if (stats) {
            stats->exact_distance_pairs++;
        }
    }
    if (stats) {
        stats->candidates_seen += (uint64_t)count;
    }

    qsort(items, (size_t)count, sizeof(*items), sb_candidate_compare);

    if (cfg->prune_mode == 0) {
        int keep = (count < cfg->r) ? count : cfg->r;
        for (i = 0; i < keep; i++) {
            if (sb_set_row_edge(new_graph, u, i, items[i].id,
                                items[i].exact_dist) != 0) {
                free(items);
                return -1;
            }
        }
    } else {
        for (i = 0; i < count && new_graph->degrees[u] < cfg->r; i++) {
            int occluded = 0;
            int j;
            for (j = 0; j < new_graph->degrees[u]; j++) {
                int accepted = new_graph->neighbors[sb_slot(new_graph, u, j)];
                float pc = sb_distance(runtime, base, dim,
                                                         accepted,
                                                         items[i].id);
                if (stats) {
                    stats->exact_distance_pairs++;
                }
                if (!isfinite(pc)) { free(items); return -1; }
                if (pc < items[i].exact_dist) {
                    occluded = 1;
                    break;
                }
            }
            if (!occluded) {
                if (sb_set_row_edge(new_graph, u, new_graph->degrees[u],
                                    items[i].id, items[i].exact_dist) != 0) {
                    free(items);
                    return -1;
                }
            }
        }
        if (new_graph->degrees[u] < cfg->r) {
            for (i = 0; i < count && new_graph->degrees[u] < cfg->r; i++) {
                if (!sb_row_contains(new_graph, u, items[i].id)) {
                    if (sb_set_row_edge(new_graph, u, new_graph->degrees[u],
                                        items[i].id,
                                        items[i].exact_dist) != 0) {
                        free(items);
                        return -1;
                    }
                    if (stats) {
                        stats->fill_edges++;
                    }
                }
            }
        }
    }

    if (new_graph->degrees[u] < cfg->r &&
        sb_fill_missing_edges(base, dim, cfg, runtime, new_graph, u, stats) != 0) {
        free(items);
        return -1;
    }

    sb_sort_row(new_graph, u);
    if (stats) {
        stats->candidates_kept += (uint64_t)new_graph->degrees[u];
    }
    free(items);
    return 0;
}

static int sb_build_rows_serial(const float *base,
                                int num_vectors,
                                int dim,
                                const RetrievalSearchBuildConfig *cfg,
                                const RetrievalSearchBuildRuntime *runtime,
                                SearchBuildGate *gate,
                                const RetrievalRnnGraph *old_graph,
                                RetrievalRnnGraph *new_graph,
                                int start,
                                int end,
                                RetrievalSearchBuildStats *stats) {
    RetrievalSearchBuildCandidate *provider_items = NULL;
    RetrievalSearchBuildCandidate *provider_items_b = NULL;
    int range = end - start;
    int log_enabled = 0;
    int log_step = 0;
    int next_log = 0;
    uint64_t log_t0 = 0;
    int u;

    provider_items = (RetrievalSearchBuildCandidate *)malloc(
        (size_t)cfg->candidate_count * sizeof(*provider_items));
    if (!provider_items) {
        return -1;
    }
    if (runtime->search_pair) {
        provider_items_b = (RetrievalSearchBuildCandidate *)malloc(
            (size_t)cfg->candidate_count * sizeof(*provider_items_b));
        if (!provider_items_b) {
            free(provider_items);
            return -1;
        }
    }

    if (runtime->progress_every_iter && range >= 10000) {
        log_enabled = 1;
        log_step = range / 20;
        if (log_step < 1) {
            log_step = 1;
        }
        next_log = start + log_step;
        log_t0 = sb_now_us();
        printf("  search-build worker rows: [%d,%d) start\n", start, end);
        fflush(stdout);
    }

    for (u = start; u < end; ) {
        int provider_count = 0;
        uint64_t t0 = sb_now_us();
        if (runtime->search_pair) {
            int v = u + 1;
            int have_second = (v < end);
            int provider_count_b = 0;
            if (!have_second) {
                v = u;
            }
            if (runtime->search_pair(runtime->search_ctx,
                                     base,
                                     num_vectors,
                                     dim,
                                     cfg,
                                     old_graph,
                                     u,
                                     v,
                                     cfg->candidate_count,
                                     provider_items,
                                     &provider_count,
                                     provider_items_b,
                                     &provider_count_b) != 0) {
                free(provider_items_b);
                free(provider_items);
                return -1;
            }
            if (stats) {
                stats->search_us += sb_now_us() - t0;
                stats->searched_queries += have_second ? 2u : 1u;
            }
            if (provider_count < 0 ||
                provider_count > cfg->candidate_count ||
                provider_count_b < 0 ||
                provider_count_b > cfg->candidate_count) {
                free(provider_items_b);
                free(provider_items);
                return -1;
            }
            t0 = sb_now_us();
            if (sb_write_pruned_row(base,
                                    num_vectors,
                                    dim,
                                    cfg,
                                runtime,
                                    old_graph,
                                    new_graph,
                                    u,
                                    provider_items,
                                    provider_count,
                                    stats) != 0) {
                free(provider_items_b);
                free(provider_items);
                return -1;
            }
            if (have_second &&
                sb_write_pruned_row(base,
                                    num_vectors,
                                    dim,
                                    cfg,
                                runtime,
                                    old_graph,
                                    new_graph,
                                    v,
                                    provider_items_b,
                                    provider_count_b,
                                    stats) != 0) {
                free(provider_items_b);
                free(provider_items);
                return -1;
            }
            if (stats) {
                stats->prune_us += sb_now_us() - t0;
            }
            u += have_second ? 2 : 1;
            if (log_enabled && (u >= next_log || u >= end)) {
                int done = u - start;
                if (done > range) {
                    done = range;
                }
                printf("  search-build worker rows: [%d,%d) %d/%d (%d%%), elapsed=%llu ms\n",
                       start,
                       end,
                       done,
                       range,
                       (int)(((int64_t)done * 100) / range),
                       (unsigned long long)(sb_now_us() - log_t0) / 1000ULL);
                fflush(stdout);
                while (next_log <= u) {
                    next_log += log_step;
                }
            }
            continue;
        }
        if (sb_gate_enter(gate)) { free(provider_items); return -1; }
        t0 = sb_now_us();
        if (runtime->search(runtime->search_ctx,
                            base,
                            num_vectors,
                            dim,
                            cfg,
                            old_graph,
                            u,
                            cfg->candidate_count,
                            provider_items,
                            &provider_count) != 0) {
            free(provider_items);
            return -1;
        }
        if (stats) {
            stats->search_us += sb_now_us() - t0;
            stats->searched_queries++;
        }
        if (provider_count < 0 || provider_count > cfg->candidate_count) {
            free(provider_items);
            return -1;
        }
        t0 = sb_now_us();
        if (sb_write_pruned_row(base,
                                num_vectors,
                                dim,
                                cfg,
                                runtime,
                                old_graph,
                                new_graph,
                                u,
                                provider_items,
                                provider_count,
                                stats) != 0) {
            free(provider_items);
            return -1;
        }
        if (stats) {
            stats->prune_us += sb_now_us() - t0;
        }
        if (sb_gate_finish(gate)) { free(provider_items_b); free(provider_items); return -1; }
        u++;
        if (log_enabled && (u >= next_log || u >= end)) {
            int done = u - start;
            if (done > range) {
                done = range;
            }
            printf("  search-build worker rows: [%d,%d) %d/%d (%d%%), elapsed=%llu ms\n",
                   start,
                   end,
                   done,
                   range,
                   (int)(((int64_t)done * 100) / range),
                   (unsigned long long)(sb_now_us() - log_t0) / 1000ULL);
            fflush(stdout);
            if (stats) {
                printf("  search-build worker cost: rows=[%d,%d) queries=%llu search_with_lock_wait_ms=%llu prune_ms=%llu exact_pairs=%llu\n",
                       start,end,(unsigned long long)stats->searched_queries,
                       (unsigned long long)stats->search_us/1000ULL,
                       (unsigned long long)stats->prune_us/1000ULL,
                       (unsigned long long)stats->exact_distance_pairs);
                fflush(stdout);
            }
            while (next_log <= u) {
                next_log += log_step;
            }
        }
    }

    free(provider_items_b);
    free(provider_items);
    return 0;
}

#include "retrieval_search_build_rng.c"

static void *sb_worker_fn(void *arg) {
    SearchBuildWorker *w = (SearchBuildWorker *)arg;
    if (w->runtime->score_candidates) {
        w->ret = sb_revised_rows(w);
        if (w->ret < 0) { __atomic_store_n(w->failed,1,__ATOMIC_RELEASE); sb_gate_abort(w->gate); }
        return NULL;
    }
    w->ret = sb_build_rows_serial(w->base,
                                  w->num_vectors,
                                  w->dim,
                                  w->cfg,
                                  w->runtime,
                                  w->gate,
                                  w->old_graph,
                                  w->new_graph,
                                  w->start,
                                  w->end,
                                  &w->stats);
    if (w->ret) sb_gate_abort(w->gate);
    return NULL;
}

static int sb_build_iteration(const float *base,
                              int num_vectors,
                              int dim,
                              const RetrievalSearchBuildConfig *cfg,
                              const RetrievalSearchBuildRuntime *runtime,
                               SearchBuildGate *gate,
                              const RetrievalRnnGraph *old_graph,
                              RetrievalRnnGraph *new_graph,
                              int threads, int old_scores_valid,
                              RetrievalSearchBuildStats *stats) {
    SearchBuildWorker *workers = NULL;
    pthread_t *tids = NULL;
    int launched = 0;
    int failed = 0;
    int rc = 0;
    int i;

    if (threads <= 1 || num_vectors < 2) {
        if (runtime->score_candidates) {
            SearchBuildWorker w={0};
            w.base=base;w.num_vectors=num_vectors;w.dim=dim;w.cfg=cfg;w.runtime=runtime;
            w.gate=gate;w.old_graph=old_graph;w.new_graph=new_graph;w.end=num_vectors;
            w.old_scores_valid=old_scores_valid;w.failed=&failed;
            int ret=sb_revised_rows(&w);sb_stats_accumulate(stats,&w.stats);return ret;
        }
        return sb_build_rows_serial(base, num_vectors, dim, cfg, runtime, gate,
                                    old_graph, new_graph, 0, num_vectors,
                                    stats);
    }

    workers = (SearchBuildWorker *)calloc((size_t)threads, sizeof(*workers));
    tids = (pthread_t *)calloc((size_t)threads, sizeof(*tids));
    if (!workers || !tids) {
        free(workers);
        free(tids);
        return -1;
    }

    for (i = 0; i < threads; i++) {
        workers[i].old_scores_valid = old_scores_valid;
        workers[i].failed = &failed;
        workers[i].base = base;
        workers[i].num_vectors = num_vectors;
        workers[i].dim = dim;
        workers[i].cfg = cfg;
        workers[i].runtime = runtime;
        workers[i].gate = gate;
        workers[i].old_graph = old_graph;
        workers[i].new_graph = new_graph;
        workers[i].start = (int)(((int64_t)num_vectors * i) / threads);
        workers[i].end = (int)(((int64_t)num_vectors * (i + 1)) / threads);
        if (pthread_create(&tids[i], NULL, sb_worker_fn,
                           &workers[i]) != 0) {
            workers[i].ret = -1;
            __atomic_store_n(&failed,1,__ATOMIC_RELEASE);
            sb_gate_abort(gate);
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
        sb_stats_accumulate(stats, &workers[i].stats);
    }

    free(workers);
    free(tids);
    return rc;
}

int retrieval_search_build_graph_ex(
    const float *base,
    int num_vectors,
    int dim,
    const RetrievalSearchBuildConfig *cfg,
    const RetrievalSearchBuildRuntime *runtime,
    RetrievalRnnGraph *out_graph,
    RetrievalSearchBuildStats *stats) {
    RetrievalRnnGraph graph;
    SearchBuildGate gate={0};
    SearchBuildGate *cooling=NULL;
    int threads;
    int iter;
    uint64_t t0;
    int rc = -1;

    if (stats) {
        retrieval_search_build_stats_reset(stats);
    }
    if (sb_validate(base, num_vectors, dim, cfg, runtime, out_graph) != 0) {
        return -1;
    }
    if (runtime->cooldown_every_queries) {
        pthread_mutex_init(&gate.mu,NULL); pthread_cond_init(&gate.cv,NULL);
        gate.runtime=runtime; gate.total=(uint64_t)num_vectors*cfg->iterations;
        gate.boundary=runtime->cooldown_every_queries; cooling=&gate;
    }
    memset(out_graph, 0, sizeof(*out_graph));
    memset(&graph, 0, sizeof(graph));
    threads = runtime->threads;
    if (threads > num_vectors) {
        threads = num_vectors;
    }
    if (threads < 1) {
        threads = 1;
    }
    if (stats) {
        stats->threads_used = threads;
    }

    if (runtime->progress_every_iter) {
        printf("  search-build begin: n=%d dim=%d R=%d iter=%d cand=%d threads=%d provider=%s\n",
               num_vectors,
               dim,
               cfg->r,
               cfg->iterations,
               cfg->candidate_count,
               threads,
               runtime->search_pair ? "pair" : "single");
        fflush(stdout);
    }

    if (retrieval_rnn_graph_alloc_without_flags(&graph, num_vectors, dim, cfg->r) != 0) {
        goto done;
    }

    t0 = sb_now_us();
    if (runtime->progress_every_iter) {
        printf("  search-build init: random graph start (rows=%d, R=%d)\n",
               num_vectors,
               cfg->r);
        fflush(stdout);
    }
    if (sb_random_init_graph(base, num_vectors, dim, cfg, runtime, &graph, stats) != 0) {
        goto done;
    }
    if (stats) {
        stats->init_us += sb_now_us() - t0;
    }
    if (runtime->progress_every_iter) {
        printf("  search-build init: random graph done time=%llu ms exact_pairs=%llu fill=%llu\n",
               (unsigned long long)(sb_now_us() - t0) / 1000ULL,
               stats ? (unsigned long long)stats->exact_distance_pairs : 0ULL,
               stats ? (unsigned long long)stats->fill_edges : 0ULL);
        fflush(stdout);
    }

    for (iter = 0; iter < cfg->iterations; iter++) {
        RetrievalRnnGraph next_graph;
        if (runtime->prepare_iteration) {
            t0 = sb_now_us();
            if (runtime->progress_every_iter) {
                printf("  search-build iter %d/%d: bind graph to provider start\n",
                       iter + 1,
                       cfg->iterations);
                fflush(stdout);
            }
            if (runtime->prepare_iteration(runtime->iteration_ctx,
                                           base,
                                           num_vectors,
                                           dim,
                                           cfg,
                                           &graph,
                                           iter) != 0) {
                goto done;
            }
            if (stats) {
                stats->refine_us += sb_now_us() - t0;
            }
            if (runtime->progress_every_iter) {
                printf("  search-build iter %d/%d: provider graph load done time=%llu ms\n",
                       iter + 1,
                       cfg->iterations,
                       (unsigned long long)(sb_now_us() - t0) / 1000ULL);
                fflush(stdout);
            }
        }
        memset(&next_graph, 0, sizeof(next_graph));
        if (retrieval_rnn_graph_alloc_without_flags(&next_graph, num_vectors, dim,
                                      cfg->r) != 0) {
            goto done;
        }
        t0 = sb_now_us();
        if (runtime->progress_every_iter) {
            printf("  search-build iter %d/%d: build rows start\n",
                   iter + 1,
                   cfg->iterations);
            fflush(stdout);
        }
        if (sb_build_iteration(base, num_vectors, dim, cfg, runtime, cooling,
                               &graph, &next_graph, threads, iter > 0, stats) != 0) {
            retrieval_rnn_graph_free(&next_graph);
            goto done;
        }
        retrieval_rnn_graph_free(&graph);
        graph = next_graph;
        if (runtime->reverse_candidate_count) {
            if (runtime->progress_every_iter) { printf("  search-build reverse: iter=%d/%d begin\n",iter+1,cfg->iterations);fflush(stdout); }
            if (sb_reverse_phase(base,num_vectors,dim,cfg,runtime,&graph,threads,stats)) goto done;
            if (runtime->progress_every_iter) { printf("  search-build reverse: iter=%d/%d done\n",iter+1,cfg->iterations);fflush(stdout); }
        }
        if (stats) {
            stats->iterations_done = iter + 1;
        }
        if (runtime->progress_every_iter) {
            RetrievalSearchBuildStats degree_stats;
            printf("  search-build iter %d/%d: build rows done time=%llu ms\n",
                   iter + 1,
                   cfg->iterations,
                   (unsigned long long)(sb_now_us() - t0) / 1000ULL);
            fflush(stdout);
            retrieval_search_build_stats_reset(&degree_stats);
            sb_fill_graph_stats(&graph, &degree_stats);
            printf("  search-build progress: iter=%d/%d queries=%llu candidates=%llu kept=%llu avg_degree=%.2f min=%d max=%d isolated=%d\n",
                   iter + 1,
                   cfg->iterations,
                   stats ? (unsigned long long)stats->searched_queries : 0ULL,
                   stats ? (unsigned long long)stats->candidates_seen : 0ULL,
                   stats ? (unsigned long long)stats->candidates_kept : 0ULL,
                   (double)degree_stats.avg_degree_x100 / 100.0,
                   degree_stats.min_degree,
                   degree_stats.max_degree,
                   degree_stats.isolated_nodes);
            fflush(stdout);
        }
    }

    sb_fill_graph_stats(&graph, stats);
    *out_graph = graph;
    memset(&graph, 0, sizeof(graph));
    rc = 0;

done:
    if (cooling) {
        if (stats) { stats->cooldown_us=gate.rest_us; stats->cooldown_events=gate.events; }
        pthread_cond_destroy(&gate.cv); pthread_mutex_destroy(&gate.mu);
    }
    retrieval_rnn_graph_free(&graph);
    return rc;
}
