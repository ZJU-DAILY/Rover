#ifndef RETRIEVAL_SEARCH_BUILD_CORE_H
#define RETRIEVAL_SEARCH_BUILD_CORE_H

#include <stdint.h>

#include "dsp_demo/retrieval_rnn_descent_core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RetrievalSearchBuildConfig {
    int r;
    int iterations;
    int candidate_count;
    int entry_points;
    int expand_count;
    int pool_size;
    uint32_t seed;
    int prune_mode;
    int batch_align;
} RetrievalSearchBuildConfig;

typedef struct RetrievalSearchBuildCandidate {
    int32_t id;
    float search_dist;
    float exact_dist;
} RetrievalSearchBuildCandidate;

typedef int (*RetrievalSearchBuildFn)(
    void *ctx,
    const float *base,
    int num_vectors,
    int dim,
    const RetrievalSearchBuildConfig *cfg,
    const RetrievalRnnGraph *graph_snapshot,
    int query_id,
    int candidate_cap,
    RetrievalSearchBuildCandidate *out_candidates,
    int *out_count);

typedef int (*RetrievalSearchBuildPairFn)(
    void *ctx,
    const float *base,
    int num_vectors,
    int dim,
    const RetrievalSearchBuildConfig *cfg,
    const RetrievalRnnGraph *graph_snapshot,
    int query_id_a,
    int query_id_b,
    int candidate_cap,
    RetrievalSearchBuildCandidate *out_candidates_a,
    int *out_count_a,
    RetrievalSearchBuildCandidate *out_candidates_b,
    int *out_count_b);

typedef int (*RetrievalSearchBuildIterationFn)(
    void *ctx,
    const float *base,
    int num_vectors,
    int dim,
    const RetrievalSearchBuildConfig *cfg,
    const RetrievalRnnGraph *graph_snapshot,
    int iteration);

typedef struct RetrievalSearchBuildRuntime {
    int threads;
    int progress_every_iter;
    RetrievalSearchBuildFn search;
    RetrievalSearchBuildPairFn search_pair;
    void *search_ctx;
    RetrievalSearchBuildIterationFn prepare_iteration;
    void *iteration_ctx;
    RetrievalRnnDistanceBatchFn distance_batch;
    void *distance_ctx;

    uint64_t cooldown_every_queries;
    uint64_t cooldown_us;
    void (*cooldown_event)(void *ctx, uint64_t queries, uint64_t start_us, uint64_t end_us);
    void *cooldown_ctx;

    int (*cooldown_quiesce)(void *ctx, int pause);
    void *cooldown_quiesce_ctx;

    int (*score_candidates)(void *ctx, int query_id, const int32_t *ids,
                            int count, float *scores);
    void *score_ctx;
    int (*read_vector)(void *ctx, int id, float *vector);
    void *read_vector_ctx;
    int row_vector_buffer;

    int reverse_candidate_count;
    int diagnostic_rows_per_worker;

    int (*publish_row)(void *ctx, const RetrievalRnnGraph *search_graph,
                       const RetrievalRnnGraph *completed, int row);
    void *publish_ctx;
} RetrievalSearchBuildRuntime;

typedef struct RetrievalSearchBuildStats {
    uint64_t cooldown_us;
    uint64_t cooldown_events;
    uint64_t init_us;
    uint64_t search_us;
    uint64_t prune_us;
    uint64_t refine_us;
    uint64_t export_us;
    uint64_t searched_queries;
    uint64_t candidates_seen;
    uint64_t candidates_kept;
    uint64_t exact_distance_pairs;
    uint64_t duplicate_rejects;
    uint64_t self_rejects;
    uint64_t rows_filled;
    uint64_t fill_edges;
    uint64_t scores_reused, scores_computed, score_us;
    uint64_t row_vector_loads, row_read_us, pair_distance_us;
    uint64_t reverse_us, reverse_index_us, reverse_rows;
    uint64_t reverse_edges_seen, reverse_candidates_dropped;
    uint64_t reverse_index_peak_bytes;
    uint32_t reverse_max_indegree;
    int iterations_done;
    int threads_used;
    int avg_degree_x100;
    int min_degree;
    int max_degree;
    int isolated_nodes;
} RetrievalSearchBuildStats;

void retrieval_search_build_stats_reset(RetrievalSearchBuildStats *stats);

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
    int *out_count);

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
    int *out_count);

int retrieval_search_build_graph_ex(
    const float *base,
    int num_vectors,
    int dim,
    const RetrievalSearchBuildConfig *cfg,
    const RetrievalSearchBuildRuntime *runtime,
    RetrievalRnnGraph *out_graph,
    RetrievalSearchBuildStats *stats);

#ifdef __cplusplus
}
#endif

#endif
