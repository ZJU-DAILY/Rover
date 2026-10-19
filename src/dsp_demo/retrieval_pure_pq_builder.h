#ifndef RETRIEVAL_PURE_PQ_BUILDER_H
#define RETRIEVAL_PURE_PQ_BUILDER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PurePqBuildRequest {
    const char *base_fvecs_path;
    const char *order_map_path;
    const char *artifact_prefix;
    int expected_num_vectors;
    int expected_dim;
    int m;
    int ks;
    int train_max;
    int kmeans_iters;
    int encode_chunk_vectors;
    int residual_bits;
    size_t memory_budget_bytes;
    const char *layout_id;
    const uint8_t *layout_fingerprint;
} PurePqBuildRequest;

typedef struct PurePqBuildStats {
    int num_vectors;
    int dim;
    int train_count;
    int used_bounded_training;
    int encode_chunk_vectors;
    size_t estimated_peak_bytes;
} PurePqBuildStats;

int pure_pq_build_mobile_artifacts(const PurePqBuildRequest *request,
                                   PurePqBuildStats *out_stats);

#ifdef __cplusplus
}
#endif

#endif
