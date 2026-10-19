#ifndef RETRIEVAL_PURE_PQ_INDEX_H
#define RETRIEVAL_PURE_PQ_INDEX_H

#include <stddef.h>
#include <stdint.h>

#include "remote.h"

#include "dsp_demo/retrieval_pq_artifact.h"
#include "dsp_demo/retrieval_pq_residual.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum PurePqPrepareMode {
    PURE_PQ_LOAD_ONLY = 0,
    PURE_PQ_TRAIN_IF_MISSING = 1,
    PURE_PQ_REBUILD = 2
} PurePqPrepareMode;

typedef struct PurePqPrepareRequest {
    PurePqPrepareMode mode;
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
    const uint8_t *expected_layout_fingerprint;
} PurePqPrepareRequest;

typedef struct PurePqIndex PurePqIndex;

int pure_pq_index_prepare(const PurePqPrepareRequest *request,
                          PurePqIndex **out_index);

int pure_pq_index_map_dsp(PurePqIndex *index,
                          remote_handle64 handle,
                          int domain_id);

int pure_pq_index_unmap_dsp(PurePqIndex *index);

const RetrievalPqArtifactMetadata *pure_pq_index_metadata(
    const PurePqIndex *index);

int pure_pq_index_build_lut(const PurePqIndex *index,
                            const float *query,
                            float *out_lut,
                            size_t lut_count);

int pure_pq_index_score_ids(const PurePqIndex *index,
                            const int32_t *ids,
                            int count,
                            const float *lut,
                            float *out_distances);

int pure_pq_index_open_residual(
    PurePqIndex *index,
    const char *artifact_prefix,
    int residual_bits,
    int cache_slots,
    RetrievalPqResidualIndex **out_residual);

void pure_pq_index_release(PurePqIndex *index);

#ifdef __cplusplus
}
#endif

#endif
