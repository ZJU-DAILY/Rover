#include "build/rover_insert_build_core.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct SearchBuildWorker {
    const float *base;
    int num_vectors;
    int dim;
    const RoverInsertBuildConfig *cfg;
    const RoverInsertBuildRuntime *runtime;
    const RetrievalRnnGraph *old_graph;
    RetrievalRnnGraph *new_graph;
    RoverInsertBuildStats stats;
    int start;
    int end;
    int ret;
} SearchBuildWorker;

typedef struct SearchBuildPoolItem {
    int32_t id;
    float dist;
    uint8_t expanded;
} SearchBuildPoolItem;

static uint64_t sb_now_us(void) {
    struct timespec tv;
    clock_gettime(CLOCK_MONOTONIC, &tv);
    return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_nsec / 1000ULL;
}

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
    const RoverInsertBuildCandidate *ca =
        (const RoverInsertBuildCandidate *)a;
    const RoverInsertBuildCandidate *cb =
        (const RoverInsertBuildCandidate *)b;
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

static void sb_stats_accumulate(RoverInsertBuildStats *dst,
                                const RoverInsertBuildStats *src) {
    if (!dst || !src) {
        return;
    }
    dst->init_us += src->init_us;
    dst->search_us += src->search_us;
    dst->prune_us += src->prune_us;
    dst->refine_us += src->refine_us;
    dst->export_us += src->export_us;
    dst->insert_us += src->insert_us;
    dst->backlink_us += src->backlink_us;
    dst->searched_queries += src->searched_queries;
    dst->candidates_seen += src->candidates_seen;
    dst->candidates_kept += src->candidates_kept;
    dst->exact_distance_pairs += src->exact_distance_pairs;
    dst->duplicate_rejects += src->duplicate_rejects;
    dst->self_rejects += src->self_rejects;
    dst->rows_filled += src->rows_filled;
    dst->fill_edges += src->fill_edges;
    dst->inserted_vectors += src->inserted_vectors;
    dst->backlink_rows_considered += src->backlink_rows_considered;
    dst->backlink_rows_changed += src->backlink_rows_changed;
    dst->provider_failures += src->provider_failures;
}

void rover_insert_build_stats_reset(RoverInsertBuildStats *stats) {
    if (stats) {
        memset(stats, 0, sizeof(*stats));
    }
}

static int sb_validate(const float *base,
                       int num_vectors,
                       int dim,
                       const RoverInsertBuildConfig *cfg,
                       const RoverInsertBuildRuntime *runtime,
                       RetrievalRnnGraph *out_graph) {
    if (!base || !cfg || !runtime || !out_graph ||
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
        runtime->threads <= 0 ||
        (!runtime->search && !runtime->search_pair)) {
        return -1;
    }
    return 0;
}

static void sb_fill_graph_stats(const RetrievalRnnGraph *graph,
                                RoverInsertBuildStats *stats) {
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

static void sb_clear_row(RetrievalRnnGraph *graph, int src) {
    int pos;

    if (!graph || src < 0 || src >= graph->num_vectors) {
        return;
    }
    for (pos = 0; pos < graph->max_degree; pos++) {
        size_t slot = sb_slot(graph, src, pos);
        graph->neighbors[slot] = -1;
        graph->neighbor_dists[slot] = FLT_MAX;
        if (graph->is_new) {
            graph->is_new[slot] = 0u;
        }
    }
    graph->degrees[src] = 0;
}

static int sb_candidate_add(RoverInsertBuildCandidate *items,
                            int cap,
                            int *count,
                            int id,
                            float search_dist,
                            RoverInsertBuildStats *stats) {
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

int rover_insert_build_exact_provider(
    void *ctx,
    const float *base,
    int num_vectors,
    int dim,
    const RoverInsertBuildConfig *cfg,
    const RetrievalRnnGraph *graph_snapshot,
    int query_id,
    int candidate_cap,
    RoverInsertBuildCandidate *out_candidates,
    int *out_count) {
    RoverInsertBuildCandidate *all = NULL;
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

    all = (RoverInsertBuildCandidate *)malloc(
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

int rover_insert_build_graph_provider(
    void *ctx,
    const float *base,
    int num_vectors,
    int dim,
    const RoverInsertBuildConfig *cfg,
    const RetrievalRnnGraph *graph_snapshot,
    int query_id,
    int candidate_cap,
    RoverInsertBuildCandidate *out_candidates,
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
                                const RoverInsertBuildConfig *cfg,
                                RetrievalRnnGraph *graph,
                                RoverInsertBuildStats *stats) {
    RoverInsertBuildCandidate *row = NULL;
    int u;

    row = (RoverInsertBuildCandidate *)malloc(
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
                    retrieval_rnn_l2_distance_cpu(base, dim, u, v);
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
                        retrieval_rnn_l2_distance_cpu(base, dim, u, v);
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


static float sb_vector_l2(const float *a,const float *b,int dim) {
    float sum=0.0f;
    for(int d=0;d<dim;d++){float diff=a[d]-b[d];sum+=diff*diff;}
    return sum;
}
static int sb_read_distance(const float *base,int dim,int a,int b,
    const RoverInsertBuildRuntime *runtime,float *out) {
    if(runtime && runtime->read_vector) {
        float va[dim],vb[dim];
        if(runtime->read_vector(runtime->vector_ctx,a,va) ||
           runtime->read_vector(runtime->vector_ctx,b,vb))return -1;
        *out=sb_vector_l2(va,vb,dim);
    } else *out=retrieval_rnn_l2_distance_cpu(base,dim,a,b);
    return 0;
}

static int sb_fill_missing_edges(const float *base,
                                 int dim,
                                 const RoverInsertBuildConfig *cfg,
                                 const RoverInsertBuildRuntime *runtime,
                                 RetrievalRnnGraph *graph,
                                 int u,
                                 RoverInsertBuildStats *stats) {
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
            float dist;
            if(sb_read_distance(base,dim,u,v,runtime,&dist))return -1;
            if (sb_set_row_edge(graph, u, graph->degrees[u], v, dist) != 0) {
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
            if(sb_read_distance(base,dim,u,v,runtime,&dist))return -1;
            if (sb_set_row_edge(graph, u, graph->degrees[u], v, dist) != 0) {
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
                               const RoverInsertBuildConfig *cfg,
                                 const RoverInsertBuildRuntime *runtime,
                               const RetrievalRnnGraph *old_graph,
                               RetrievalRnnGraph *new_graph,
                               int u,
                               RoverInsertBuildCandidate *provider_items,
                               int provider_count,
                               RoverInsertBuildStats *stats) {
    int cap = cfg->candidate_count + cfg->r;
    RoverInsertBuildCandidate local_items[256], *items = local_items;
    int count = 0;
    int i;
    int streamed=runtime && runtime->read_vector;

    if(cap<=0 || dim<=0 || (streamed &&
       (size_t)(cap+1)*(size_t)dim*sizeof(float)>256u*1024u))return -1;
    float row_vectors[streamed?(size_t)(cap+1)*(size_t)dim:1];
    int selected_slots[streamed?cfg->r:1];
    if(streamed && runtime->read_vector(runtime->vector_ctx,u,row_vectors))return -1;

    if (cap > 256) items = (RoverInsertBuildCandidate *)malloc(
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
            if (items != local_items) free(items);
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
                    if (items != local_items) free(items);
                    return -1;
                }
            }
        }
    }

    for (i = 0; i < count; i++) {
        if(streamed) {
            items[i].vector_slot=i+1;
            float *v=row_vectors+(size_t)(i+1)*dim;
            if(runtime->read_vector(runtime->vector_ctx,items[i].id,v)) {
                if(items!=local_items)free(items);
                return -1;
            }
            items[i].exact_dist=sb_vector_l2(row_vectors,v,dim);
        } else items[i].exact_dist = retrieval_rnn_l2_distance_cpu(base, dim, u,
                                                            items[i].id);
        if (stats) {
            stats->exact_distance_pairs++;
        }
    }
    if (stats) {
        stats->candidates_seen += (uint64_t)count;
    }

    qsort(items, (size_t)count, sizeof(*items), sb_candidate_compare);
    sb_clear_row(new_graph, u);

    if (cfg->prune_mode == 0) {
        int keep = (count < cfg->r) ? count : cfg->r;
        for (i = 0; i < keep; i++) {
            if (sb_set_row_edge(new_graph, u, i, items[i].id,
                                items[i].exact_dist) != 0) {
                if (items != local_items) free(items);
                return -1;
            }
        }
    } else {
        for (i = 0; i < count && new_graph->degrees[u] < cfg->r; i++) {
            int occluded = 0;
            int j;
            for (j = 0; j < new_graph->degrees[u]; j++) {
                int accepted = new_graph->neighbors[sb_slot(new_graph, u, j)];
                float pc = streamed ? sb_vector_l2(
                    row_vectors+(size_t)selected_slots[j]*dim,
                    row_vectors+(size_t)items[i].vector_slot*dim,dim) :
                    retrieval_rnn_l2_distance_cpu(base,dim,accepted,items[i].id);
                if (stats) {
                    stats->exact_distance_pairs++;
                }
                if (pc < items[i].exact_dist) {
                    occluded = 1;
                    break;
                }
            }
            if (!occluded) {
                if(streamed)selected_slots[new_graph->degrees[u]]=items[i].vector_slot;
                if (sb_set_row_edge(new_graph, u, new_graph->degrees[u],
                                    items[i].id, items[i].exact_dist) != 0) {
                    if (items != local_items) free(items);
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
                        if (items != local_items) free(items);
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
        if (items != local_items) free(items);
        return -1;
    }

    sb_sort_row(new_graph, u);
    if (stats) {
        stats->candidates_kept += (uint64_t)new_graph->degrees[u];
    }
    if (items != local_items) free(items);
    return 0;
}

static int sb_build_rows_serial(const float *base,
                                int num_vectors,
                                int dim,
                                const RoverInsertBuildConfig *cfg,
                                const RoverInsertBuildRuntime *runtime,
                                const RetrievalRnnGraph *old_graph,
                                RetrievalRnnGraph *new_graph,
                                int start,
                                int end,
                                RoverInsertBuildStats *stats) {
    RoverInsertBuildCandidate *provider_items = NULL;
    RoverInsertBuildCandidate *provider_items_b = NULL;
    int range = end - start;
    int log_enabled = 0;
    int log_step = 0;
    int next_log = 0;
    uint64_t log_t0 = 0;
    int u;

    provider_items = (RoverInsertBuildCandidate *)malloc(
        (size_t)cfg->candidate_count * sizeof(*provider_items));
    if (!provider_items) {
        return -1;
    }
    if (runtime->search_pair) {
        provider_items_b = (RoverInsertBuildCandidate *)malloc(
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
            while (next_log <= u) {
                next_log += log_step;
            }
        }
    }

    free(provider_items_b);
    free(provider_items);
    return 0;
}

static void *sb_worker_fn(void *arg) {
    SearchBuildWorker *w = (SearchBuildWorker *)arg;
    w->ret = sb_build_rows_serial(w->base,
                                  w->num_vectors,
                                  w->dim,
                                  w->cfg,
                                  w->runtime,
                                  w->old_graph,
                                  w->new_graph,
                                  w->start,
                                  w->end,
                                  &w->stats);
    return NULL;
}

static int sb_build_iteration(const float *base,
                              int num_vectors,
                              int dim,
                              const RoverInsertBuildConfig *cfg,
                              const RoverInsertBuildRuntime *runtime,
                              const RetrievalRnnGraph *old_graph,
                              RetrievalRnnGraph *new_graph,
                              int threads,
                              RoverInsertBuildStats *stats) {
    SearchBuildWorker *workers = NULL;
    pthread_t *tids = NULL;
    int launched = 0;
    int rc = 0;
    int i;

    if (threads <= 1 || num_vectors < 2) {
        return sb_build_rows_serial(base, num_vectors, dim, cfg, runtime,
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
        workers[i].base = base;
        workers[i].num_vectors = num_vectors;
        workers[i].dim = dim;
        workers[i].cfg = cfg;
        workers[i].runtime = runtime;
        workers[i].old_graph = old_graph;
        workers[i].new_graph = new_graph;
        workers[i].start = (int)(((int64_t)num_vectors * i) / threads);
        workers[i].end = (int)(((int64_t)num_vectors * (i + 1)) / threads);
        if (pthread_create(&tids[i], NULL, sb_worker_fn,
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
        sb_stats_accumulate(stats, &workers[i].stats);
    }

    free(workers);
    free(tids);
    return rc;
}

int rover_insert_build_graph_ex(
    const float *base,
    int num_vectors,
    int dim,
    const RoverInsertBuildConfig *cfg,
    const RoverInsertBuildRuntime *runtime,
    RetrievalRnnGraph *out_graph,
    RoverInsertBuildStats *stats) {
    RetrievalRnnGraph graph;
    int threads;
    int iter;
    uint64_t t0;
    int rc = -1;

    if (stats) {
        rover_insert_build_stats_reset(stats);
    }
    if (sb_validate(base, num_vectors, dim, cfg, runtime, out_graph) != 0) {
        return -1;
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

    if (retrieval_rnn_graph_alloc(&graph, num_vectors, dim, cfg->r) != 0) {
        goto done;
    }

    t0 = sb_now_us();
    if (runtime->progress_every_iter) {
        printf("  search-build init: random graph start (rows=%d, R=%d)\n",
               num_vectors,
               cfg->r);
        fflush(stdout);
    }
    if (sb_random_init_graph(base, num_vectors, dim, cfg, &graph, stats) != 0) {
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
                printf("  search-build iter %d/%d: load frozen graph to provider start\n",
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
        if (retrieval_rnn_graph_alloc(&next_graph, num_vectors, dim,
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
        if (sb_build_iteration(base, num_vectors, dim, cfg, runtime,
                               &graph, &next_graph, threads, stats) != 0) {
            retrieval_rnn_graph_free(&next_graph);
            goto done;
        }
        retrieval_rnn_graph_free(&graph);
        graph = next_graph;
        if (stats) {
            stats->iterations_done = iter + 1;
        }
        if (runtime->progress_every_iter) {
            RoverInsertBuildStats degree_stats;
            printf("  search-build iter %d/%d: build rows done time=%llu ms\n",
                   iter + 1,
                   cfg->iterations,
                   (unsigned long long)(sb_now_us() - t0) / 1000ULL);
            fflush(stdout);
            rover_insert_build_stats_reset(&degree_stats);
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
    retrieval_rnn_graph_free(&graph);
    return rc;
}

static int sb_validate_parent_graph(const RetrievalRnnGraph *graph,
                                    int parent_num_vectors,
                                    int dim,
                                    int max_degree) {
    int u;

    if (!graph || graph->num_vectors != parent_num_vectors ||
        graph->dim != dim || graph->max_degree != max_degree ||
        !graph->degrees || !graph->neighbors || !graph->neighbor_dists) {
        return -1;
    }
    for (u = 0; u < parent_num_vectors; u++) {
        int pos;
        int degree = graph->degrees[u];
        if (degree < 0 || degree > max_degree) {
            return -1;
        }
        for (pos = 0; pos < degree; pos++) {
            size_t slot = sb_slot(graph, u, pos);
            int neighbor = graph->neighbors[slot];
            int previous;
            if (neighbor < 0 || neighbor >= parent_num_vectors ||
                neighbor == u || !isfinite(graph->neighbor_dists[slot])) {
                return -1;
            }
            for (previous = 0; previous < pos; previous++) {
                if (graph->neighbors[sb_slot(graph, u, previous)] == neighbor) {
                    return -1;
                }
            }
        }
    }
    return 0;
}

static int sb_validate_insert(const float *base,
                              int parent_num_vectors,
                              int insert_count,
                              int dim,
                              const RoverInsertBuildConfig *cfg,
                              const RoverInsertBuildRuntime *runtime,
                              const RetrievalRnnGraph *parent_graph,
                              RetrievalRnnGraph *out_child_graph) {
    if ((!base && (!runtime || !runtime->read_vector)) || !cfg || !runtime || !parent_graph || !out_child_graph ||
        parent_num_vectors <= 1 || insert_count <= 0 || dim <= 0 ||
        parent_num_vectors > INT_MAX - insert_count ||
        cfg->r <= 0 || cfg->r >= parent_num_vectors ||
        cfg->candidate_count < cfg->r || cfg->candidate_count <= 0 ||
        cfg->entry_points <= 0 || cfg->expand_count <= 0 ||
        cfg->pool_size < cfg->candidate_count ||
        cfg->batch_align <= 0 || (cfg->r % cfg->batch_align) != 0 ||
        cfg->prune_mode != 1 || runtime->threads <= 0 ||
        (!runtime->search && !runtime->search_pair)) {
        return -1;
    }
    return sb_validate_parent_graph(parent_graph,
                                    parent_num_vectors,
                                    dim,
                                    cfg->r);
}

static int sb_clone_parent_graph(const RetrievalRnnGraph *parent_graph,
                                 int child_num_vectors,
                                 RetrievalRnnGraph *child_graph) {
    size_t parent_edges;

    if (retrieval_rnn_graph_alloc_without_flags(child_graph,
                                          child_num_vectors,
                                          parent_graph->dim,
                                          parent_graph->max_degree) != 0) {
        return -1;
    }
    parent_edges = (size_t)parent_graph->num_vectors *
                   (size_t)parent_graph->max_degree;
    memcpy(child_graph->degrees,
           parent_graph->degrees,
           (size_t)parent_graph->num_vectors * sizeof(*child_graph->degrees));
    memcpy(child_graph->neighbors,
           parent_graph->neighbors,
           parent_edges * sizeof(*child_graph->neighbors));
    memcpy(child_graph->neighbor_dists,
           parent_graph->neighbor_dists,
           parent_edges * sizeof(*child_graph->neighbor_dists));
    if (parent_graph->is_new && child_graph->is_new) {
        memcpy(child_graph->is_new,
               parent_graph->is_new,
               parent_edges * sizeof(*child_graph->is_new));
    }
    return 0;
}

static int sb_search_insert_candidates(
    const float *base,
    int num_vectors,
    int dim,
    const RoverInsertBuildConfig *cfg,
    const RoverInsertBuildRuntime *runtime,
    const RetrievalRnnGraph *graph,
    int query_id,
    RoverInsertBuildCandidate *items,
    RoverInsertBuildCandidate *pair_items,
    int *out_count,
    RoverInsertBuildStats *stats) {
    uint64_t t0 = sb_now_us();
    int count = 0;
    int rc;

    if (runtime->search) {
        rc = runtime->search(runtime->search_ctx,
                             base,
                             num_vectors,
                             dim,
                             cfg,
                             graph,
                             query_id,
                             cfg->candidate_count,
                             items,
                             &count);
    } else {
        int ignored_count = 0;
        rc = runtime->search_pair(runtime->search_ctx,
                                  base,
                                  num_vectors,
                                  dim,
                                  cfg,
                                  graph,
                                  query_id,
                                  query_id,
                                  cfg->candidate_count,
                                  items,
                                  &count,
                                  pair_items,
                                  &ignored_count);
    }
    if (stats) {
        stats->search_us += sb_now_us() - t0;
        stats->searched_queries++;
    }
    if (rc != 0 || count < 0 || count > cfg->candidate_count) {
        if (stats) {
            stats->provider_failures++;
        }
        return -1;
    }
    *out_count = count;
    return 0;
}

static int sb_row_ids_equal(const RetrievalRnnGraph *graph,
                            int src,
                            const int *old_ids,
                            int old_degree) {
    int pos;

    if (graph->degrees[src] != old_degree) {
        return 0;
    }
    for (pos = 0; pos < old_degree; pos++) {
        if (graph->neighbors[sb_slot(graph, src, pos)] != old_ids[pos]) {
            return 0;
        }
    }
    return 1;
}

#include "retrieval_insert_backlinks.c"

static int sb_insert_one_inplace(
    const float *base,
    int logical_num_vectors,
    int insert_index,
    int dim,
    const RoverInsertBuildConfig *cfg,
    const RoverInsertBuildRuntime *runtime,
    SbBacklinkPool *backlinks,
    RetrievalRnnGraph *graph,
    RoverInsertBuildCandidate *provider_items,
    RoverInsertBuildCandidate *pair_items,
    int *forward_ids,
    int *old_row_ids,
    RoverInsertBuildStats *stats) {
    RoverInsertBuildCandidate backlink_item;
    uint64_t t0;
    int new_id = logical_num_vectors - 1;
    int provider_count = 0;
    int forward_degree;
    int pos;

    graph->num_vectors = logical_num_vectors;
    if (runtime->prepare_iteration) {
        t0 = sb_now_us();
        if (runtime->prepare_iteration(runtime->iteration_ctx,
                                       base,
                                       logical_num_vectors,
                                       dim,
                                       cfg,
                                       graph,
                                       insert_index) != 0) {
            if (stats) {
                stats->provider_failures++;
            }
            return -1;
        }
        if (stats) {
            stats->refine_us += sb_now_us() - t0;
        }
    }
    if (sb_search_insert_candidates(base,
                                    logical_num_vectors,
                                    dim,
                                    cfg,
                                    runtime,
                                    graph,
                                    new_id,
                                    provider_items,
                                    pair_items,
                                    &provider_count,
                                    stats) != 0) {
        return -1;
    }

    t0 = sb_now_us();
    if (sb_write_pruned_row(base,
                            logical_num_vectors,
                            dim,
                            cfg,
                            runtime,
                            NULL,
                            graph,
                            new_id,
                            provider_items,
                            provider_count,
                            stats) != 0) {
        return -1;
    }
    if (stats) {
        stats->prune_us += sb_now_us() - t0;
    }

    forward_degree = graph->degrees[new_id];
    for (pos = 0; pos < forward_degree; pos++) {
        forward_ids[pos] = graph->neighbors[sb_slot(graph, new_id, pos)];
    }

    t0 = sb_now_us();
    int backlink_rc = backlinks ? sb_backlink_pool_run(backlinks, base, cfg,
        runtime, graph, new_id, forward_ids, forward_degree, stats) : 1;
    if (backlink_rc < 0) return -1;
    if (backlink_rc > 0) for (pos = 0; pos < forward_degree; pos++) {
        int neighbor = forward_ids[pos];
        int old_degree = graph->degrees[neighbor];
        int old_pos;

        for (old_pos = 0; old_pos < old_degree; old_pos++) {
            old_row_ids[old_pos] =
                graph->neighbors[sb_slot(graph, neighbor, old_pos)];
        }
        backlink_item.id = (int32_t)new_id;
        backlink_item.search_dist = 0.0f;
        backlink_item.exact_dist = FLT_MAX;
        if (stats) {
            stats->backlink_rows_considered++;
        }
        if (sb_write_pruned_row(base,
                                logical_num_vectors,
                                dim,
                                cfg,
                                runtime,
                                graph,
                                graph,
                                neighbor,
                                &backlink_item,
                                1,
                                stats) != 0) {
            return -1;
        }
        if (stats && !sb_row_ids_equal(graph,
                                       neighbor,
                                       old_row_ids,
                                       old_degree)) {
            stats->backlink_rows_changed++;
        }
    }
    if (stats) {
        stats->backlink_us += sb_now_us() - t0;
        stats->inserted_vectors++;
    }
    return 0;
}

static int sb_insert_batch(
    const float *base,
    int parent_num_vectors,
    int insert_count,
    int dim,
    const RoverInsertBuildConfig *cfg,
    const RoverInsertBuildRuntime *runtime,
    RetrievalRnnGraph *parent_graph,
    RetrievalRnnGraph *out_child_graph,
    RoverInsertBuildStats *stats,
    int consume_parent) {
    RoverInsertBuildCandidate *provider_items = NULL;
    RoverInsertBuildCandidate *pair_items = NULL;
    RetrievalRnnGraph child_graph;
    SbBacklinkPool *backlinks = NULL;
    int *forward_ids = NULL;
    int *old_row_ids = NULL;
    int child_num_vectors;
    int offset;
    int rc = -1;
    uint64_t insert_t0;
    uint64_t t0;

    if (stats) {
        rover_insert_build_stats_reset(stats);
    }
    if (sb_validate_insert(base,
                           parent_num_vectors,
                           insert_count,
                           dim,
                           cfg,
                           runtime,
                           parent_graph,
                           out_child_graph) != 0) {
        return -1;
    }
    memset(out_child_graph, 0, sizeof(*out_child_graph));
    memset(&child_graph, 0, sizeof(child_graph));
    child_num_vectors = parent_num_vectors + insert_count;
    insert_t0 = sb_now_us();

    t0 = sb_now_us();
    if (sb_clone_parent_graph(parent_graph,
                              child_num_vectors,
                              &child_graph) != 0) {
        goto done;
    }
    if (consume_parent) {
        retrieval_rnn_graph_free(parent_graph);
    }
    if (stats) {
        stats->init_us += sb_now_us() - t0;
        stats->threads_used = 1;
    }

    provider_items = (RoverInsertBuildCandidate *)malloc(
        (size_t)cfg->candidate_count * sizeof(*provider_items));
    if (runtime->search_pair && !runtime->search) {
        pair_items = (RoverInsertBuildCandidate *)malloc(
            (size_t)cfg->candidate_count * sizeof(*pair_items));
    }
    forward_ids = (int *)malloc((size_t)cfg->r * sizeof(*forward_ids));
    old_row_ids = (int *)malloc((size_t)cfg->r * sizeof(*old_row_ids));
    if (!provider_items ||
        (runtime->search_pair && !runtime->search && !pair_items) ||
        !forward_ids || !old_row_ids) {
        goto done;
    }

    if (sb_backlink_pool_open(cfg, runtime, dim, &backlinks)) goto done;
    if (stats && backlinks) stats->threads_used = backlinks->threads;

    for (offset = 0; offset < insert_count; offset++) {
        int logical_num_vectors = parent_num_vectors + offset + 1;
        if (sb_insert_one_inplace(base,
                                  logical_num_vectors,
                                  offset,
                                  dim,
                                  cfg,
                                  runtime,
                                  backlinks,
                                  &child_graph,
                                  provider_items,
                                  pair_items,
                                  forward_ids,
                                  old_row_ids,
                                  stats) != 0) {
            goto done;
        }
    }

    child_graph.num_vectors = child_num_vectors;
    sb_fill_graph_stats(&child_graph, stats);
    *out_child_graph = child_graph;
    memset(&child_graph, 0, sizeof(child_graph));
    rc = 0;

done:
    sb_backlink_pool_close(backlinks);
    if (stats) {
        stats->insert_us += sb_now_us() - insert_t0;
    }
    free(old_row_ids);
    free(forward_ids);
    free(pair_items);
    free(provider_items);
    retrieval_rnn_graph_free(&child_graph);
    return rc;
}

int rover_insert_build_insert_batch(
    const float *base,
    int parent_num_vectors,
    int insert_count,
    int dim,
    const RoverInsertBuildConfig *cfg,
    const RoverInsertBuildRuntime *runtime,
    const RetrievalRnnGraph *parent_graph,
    RetrievalRnnGraph *out_child_graph,
    RoverInsertBuildStats *stats) {
    return sb_insert_batch(base,
                           parent_num_vectors,
                           insert_count,
                           dim,
                           cfg,
                           runtime,
                           (RetrievalRnnGraph *)parent_graph,
                           out_child_graph,
                           stats,
                           0);
}

int rover_insert_build_insert_batch_consume_parent(
    const float *base,
    int parent_num_vectors,
    int insert_count,
    int dim,
    const RoverInsertBuildConfig *cfg,
    const RoverInsertBuildRuntime *runtime,
    RetrievalRnnGraph *parent_graph,
    RetrievalRnnGraph *out_child_graph,
    RoverInsertBuildStats *stats) {
    return sb_insert_batch(base,
                           parent_num_vectors,
                           insert_count,
                           dim,
                           cfg,
                           runtime,
                           parent_graph,
                           out_child_graph,
                           stats,
                           1);
}

#include "retrieval_insert_checkpoint.c"
#include "rover_insert_build_empty_insert.c"
