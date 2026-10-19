#ifndef ROVER_INSERT_BUILD_CORE_H
#define ROVER_INSERT_BUILD_CORE_H

#include <stdint.h>

#include "dsp_demo/retrieval_rnn_descent_core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RoverInsertBuildConfig {
    int r;
    int iterations;
    int candidate_count;
    int entry_points;
    int expand_count;
    int pool_size;
    uint32_t seed;
    int prune_mode;
    int batch_align;
} RoverInsertBuildConfig;

typedef struct RoverInsertBuildCandidate {
    int32_t id;
    float search_dist;
    float exact_dist;
    int vector_slot;
} RoverInsertBuildCandidate;

typedef int (*RoverInsertBuildFn)(
    void *ctx,
    const float *base,
    int num_vectors,
    int dim,
    const RoverInsertBuildConfig *cfg,
    const RetrievalRnnGraph *graph_snapshot,
    int query_id,
    int candidate_cap,
    RoverInsertBuildCandidate *out_candidates,
    int *out_count);

typedef int (*RoverInsertBuildPairFn)(
    void *ctx,
    const float *base,
    int num_vectors,
    int dim,
    const RoverInsertBuildConfig *cfg,
    const RetrievalRnnGraph *graph_snapshot,
    int query_id_a,
    int query_id_b,
    int candidate_cap,
    RoverInsertBuildCandidate *out_candidates_a,
    int *out_count_a,
    RoverInsertBuildCandidate *out_candidates_b,
    int *out_count_b);

typedef int (*RoverInsertBuildIterationFn)(
    void *ctx,
    const float *base,
    int num_vectors,
    int dim,
    const RoverInsertBuildConfig *cfg,
    const RetrievalRnnGraph *graph_snapshot,
    int iteration);

typedef struct RoverInsertBuildRuntime {
    int threads;
    int backlink_threads;
    int progress_every_iter;
    RoverInsertBuildFn search;
    RoverInsertBuildPairFn search_pair;
    void *search_ctx;
    RoverInsertBuildIterationFn prepare_iteration;
    void *iteration_ctx;
    RetrievalRnnDistanceBatchFn distance_batch;
    void *distance_ctx;
    int (*read_vector)(void *ctx, int id, float *out);
    void *vector_ctx;
} RoverInsertBuildRuntime;

typedef struct RoverInsertBuildStats {
    uint64_t init_us;
    uint64_t search_us;
    uint64_t prune_us;
    uint64_t refine_us;
    uint64_t export_us;
    uint64_t insert_us;
    uint64_t backlink_us;
    uint64_t backlink_prepare_us, backlink_parallel_us, backlink_vector_bytes;
    uint64_t backlink_old_requests, backlink_old_decodes, backlink_old_hits;
    uint64_t searched_queries;
    uint64_t candidates_seen;
    uint64_t candidates_kept;
    uint64_t exact_distance_pairs;
    uint64_t duplicate_rejects;
    uint64_t self_rejects;
    uint64_t rows_filled;
    uint64_t fill_edges;
    uint64_t inserted_vectors;
    uint64_t backlink_rows_considered;
    uint64_t backlink_rows_changed;
    uint64_t provider_failures;
    int iterations_done;
    int threads_used;
    int avg_degree_x100;
    int min_degree;
    int max_degree;
    int isolated_nodes;
} RoverInsertBuildStats;

void rover_insert_build_stats_reset(RoverInsertBuildStats *stats);

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
    int *out_count);

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
    int *out_count);

int rover_insert_build_graph_ex(
    const float *base,
    int num_vectors,
    int dim,
    const RoverInsertBuildConfig *cfg,
    const RoverInsertBuildRuntime *runtime,
    RetrievalRnnGraph *out_graph,
    RoverInsertBuildStats *stats);

int rover_insert_build_insert_batch(
    const float *base,
    int parent_num_vectors,
    int insert_count,
    int dim,
    const RoverInsertBuildConfig *cfg,
    const RoverInsertBuildRuntime *runtime,
    const RetrievalRnnGraph *parent_graph,
    RetrievalRnnGraph *out_child_graph,
    RoverInsertBuildStats *stats);

int rover_insert_build_insert_batch_consume_parent(
    const float *base,
    int parent_num_vectors,
    int insert_count,
    int dim,
    const RoverInsertBuildConfig *cfg,
    const RoverInsertBuildRuntime *runtime,
    RetrievalRnnGraph *parent_graph,
    RetrievalRnnGraph *out_child_graph,
    RoverInsertBuildStats *stats);


int rover_insert_build_insert_empty(const float *base, int n, int dim,
    const RoverInsertBuildConfig *cfg, const RoverInsertBuildRuntime *runtime,
    RetrievalRnnGraph *out_graph, RoverInsertBuildStats *stats);

#ifdef __cplusplus
}
#endif

#endif
