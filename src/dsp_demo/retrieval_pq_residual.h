#ifndef RETRIEVAL_PQ_RESIDUAL_H
#define RETRIEVAL_PQ_RESIDUAL_H

#include <stddef.h>
#include <stdint.h>

#include "dsp_demo/retrieval_pq_artifact.h"
#include "dsp_demo/retrieval_pq_core.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RETRIEVAL_PQ_RESIDUAL_BLOCK_BYTES 4096u
#define RETRIEVAL_PQ_RESIDUAL_HEADER_BYTES 256u
#define RETRIEVAL_PQ_RESIDUAL_QPARAM_BYTES 8u

typedef struct RetrievalPqResidualQParams {
    float scale;
    float zero_point;
} RetrievalPqResidualQParams;

typedef struct RetrievalPqResidualMetadata {
    uint8_t index_uuid[RETRIEVAL_PQ_ARTIFACT_UUID_BYTES];
    uint64_t num_vectors;
    uint32_t dim;
    uint32_t residual_bits;
    uint32_t residual_code_bytes;
    uint32_t block_bytes;
    uint32_t vectors_per_block;
    uint32_t total_blocks;
    uint8_t dataset_fingerprint[RETRIEVAL_PQ_ARTIFACT_HASH_BYTES];
    uint8_t layout_fingerprint[RETRIEVAL_PQ_ARTIFACT_HASH_BYTES];
} RetrievalPqResidualMetadata;

typedef struct RetrievalPqResidualWriter RetrievalPqResidualWriter;
typedef struct RetrievalPqResidualIndex RetrievalPqResidualIndex;
typedef struct RetrievalPqResidualQuery RetrievalPqResidualQuery;

typedef struct RetrievalPqResidualOpenRequest {
    const char *artifact_prefix;
    int residual_bits;
    int cache_slots;
    const RetrievalPqArtifactMetadata *pq_metadata;
    const float *codebook;
    const uint8_t *const *code_chunks;
    const int *chunk_vector_counts;
    int chunk_count;
    int chunk_vectors;
} RetrievalPqResidualOpenRequest;

typedef struct RetrievalPqResidualStats {
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t cache_errors;
    uint64_t corrected_candidates;
} RetrievalPqResidualStats;

int retrieval_pq_residual_writer_begin(
    const char *artifact_prefix,
    const RetrievalPqArtifactMetadata *pq_metadata,
    int residual_bits,
    RetrievalPqResidualWriter **out_writer);

int retrieval_pq_residual_writer_append(
    RetrievalPqResidualWriter *writer,
    const float *vectors,
    const uint8_t *pq_codes,
    int vector_count,
    const float *codebook,
    const RetrievalPqConfig *config);

int retrieval_pq_residual_writer_commit(RetrievalPqResidualWriter *writer);

void retrieval_pq_residual_writer_abort(RetrievalPqResidualWriter *writer);

int retrieval_pq_residual_index_open(
    const RetrievalPqResidualOpenRequest *request,
    RetrievalPqResidualIndex **out_index);

void retrieval_pq_residual_index_close(RetrievalPqResidualIndex *index);

int retrieval_pq_residual_query_create(
    RetrievalPqResidualIndex *index,
    RetrievalPqResidualQuery **out_query);


int retrieval_pq_residual_query_create_with_slots(
    RetrievalPqResidualIndex *index, int cache_slots,
    RetrievalPqResidualQuery **out_query);

int retrieval_pq_residual_query_reset(RetrievalPqResidualQuery *query,
                                      const float *query_vector);

int retrieval_pq_residual_query_bind(RetrievalPqResidualQuery *query,
                                    const float *query_vector);

int retrieval_pq_residual_query_evaluate(RetrievalPqResidualQuery *query,
                                         int vector_id,
                                         float coarse_distance,
                                         float *out_distance);


int retrieval_pq_residual_query_decode(RetrievalPqResidualQuery *query,
                                       int vector_id, float *out_vector);

void retrieval_pq_residual_query_stats(
    const RetrievalPqResidualQuery *query,
    RetrievalPqResidualStats *out_stats);

void retrieval_pq_residual_query_destroy(RetrievalPqResidualQuery *query);

#ifdef __cplusplus
}
#endif

#endif
