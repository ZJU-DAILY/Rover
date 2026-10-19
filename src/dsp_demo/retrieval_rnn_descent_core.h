#ifndef RETRIEVAL_RNN_DESCENT_CORE_H
#define RETRIEVAL_RNN_DESCENT_CORE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RetrievalRnnGraph {
    int num_vectors;
    int dim;
    int max_degree;
    int *degrees;
    int *neighbors;
    float *neighbor_dists;
    uint8_t *is_new;
} RetrievalRnnGraph;

typedef struct RetrievalRnnBuildConfig {
    int s;
    int r;
    int t1;
    int t2;
    uint32_t seed;
} RetrievalRnnBuildConfig;

typedef struct RetrievalRnnBuildStats {
    uint64_t init_us;
    uint64_t update_us;
    uint64_t reverse_us;
    uint64_t export_us;
    uint64_t distance_calls;
    uint64_t distance_batches;
    uint64_t distance_pairs_requested;
    uint64_t distance_pairs_cpu;
    uint64_t distance_pairs_dsp;
    uint64_t distance_rpc_us;
    uint64_t distance_fallbacks;
    uint64_t pending_max;
    uint64_t changed_edges;
    uint64_t edge_insertions;
    uint64_t edge_replacements;
    uint64_t duplicate_rejects;
    uint64_t self_rejects;
    int max_degree;
    int avg_degree_x100;
    int isolated_nodes;
    int threads_used;
    int distance_provider;
} RetrievalRnnBuildStats;

typedef int (*RetrievalRnnDistanceBatchFn)(
    void *ctx,
    const int32_t *ids_a,
    const int32_t *ids_b,
    int pair_count,
    float *out_dists);

typedef struct RetrievalRnnBuildRuntime {
    int threads;
    int batch_size;
    int min_batch;
    int progress_every_t2;
    RetrievalRnnDistanceBatchFn distance_batch;
    void *distance_ctx;
} RetrievalRnnBuildRuntime;

void retrieval_rnn_build_stats_reset(RetrievalRnnBuildStats *stats);

int retrieval_rnn_graph_alloc(RetrievalRnnGraph *graph,
                              int num_vectors,
                              int dim,
                              int max_degree);

void retrieval_rnn_graph_free(RetrievalRnnGraph *graph);


int retrieval_rnn_graph_alloc_without_flags(RetrievalRnnGraph *graph,
    int num_vectors, int dim, int max_degree);

float retrieval_rnn_l2_distance_cpu(const float *base,
                                    int dim,
                                    int a,
                                    int b);

int retrieval_rnn_graph_has_edge(const RetrievalRnnGraph *graph,
                                 int src,
                                 int dst);

int retrieval_rnn_graph_insert_edge(RetrievalRnnGraph *graph,
                                    const float *base,
                                    int src,
                                    int dst,
                                    uint8_t mark_new,
                                    RetrievalRnnBuildStats *stats);

int retrieval_rnn_graph_add_reverse_edges_cpu(RetrievalRnnGraph *graph,
                                              RetrievalRnnBuildStats *stats);

void retrieval_rnn_graph_sort_row(RetrievalRnnGraph *graph, int src);

int retrieval_rnn_export_topk(RetrievalRnnGraph *graph,
                              int top_k,
                              int32_t *out_graph);

int retrieval_rnn_export_topk_threads(RetrievalRnnGraph *graph,
                                      int top_k,
                                      int threads,
                                      int32_t *out_graph);

int retrieval_rnn_build_graph_cpu(const float *base,
                                  int num_vectors,
                                  int dim,
                                  const RetrievalRnnBuildConfig *cfg,
                                  RetrievalRnnGraph *out_graph,
                                  RetrievalRnnBuildStats *stats);

int retrieval_rnn_build_graph_ex(const float *base,
                                 int num_vectors,
                                 int dim,
                                 const RetrievalRnnBuildConfig *cfg,
                                 const RetrievalRnnBuildRuntime *runtime,
                                 RetrievalRnnGraph *out_graph,
                                 RetrievalRnnBuildStats *stats);

int retrieval_rnn_graph_save_cache(const char *path,
                                   const char *layout_name,
                                   const RetrievalRnnGraph *graph,
                                   const RetrievalRnnBuildConfig *cfg,
                                   uint32_t cache_version);

int retrieval_rnn_graph_load_cache(const char *path,
                                   const char *layout_name,
                                   int num_vectors,
                                   int dim,
                                   const RetrievalRnnBuildConfig *cfg,
                                   uint32_t cache_version,
                                   RetrievalRnnGraph *out_graph);

#ifdef __cplusplus
}
#endif

#endif
