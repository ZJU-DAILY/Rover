#ifndef RETRIEVAL_PQ_ARTIFACT_H
#define RETRIEVAL_PQ_ARTIFACT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RETRIEVAL_PQ_ARTIFACT_HEADER_BYTES 256u
#define RETRIEVAL_PQ_ARTIFACT_UUID_BYTES 16u
#define RETRIEVAL_PQ_ARTIFACT_HASH_BYTES 32u
#define RETRIEVAL_PQ_ARTIFACT_VERSION_MAJOR 1u
#define RETRIEVAL_PQ_ARTIFACT_VERSION_MINOR 1u

typedef enum RetrievalPqArtifactKind {
    RETRIEVAL_PQ_ARTIFACT_CODEBOOK = 1,
    RETRIEVAL_PQ_ARTIFACT_CODES = 2
} RetrievalPqArtifactKind;

typedef enum RetrievalPqTrainerKind {
    RETRIEVAL_PQ_TRAINER_UNKNOWN = 0,
    RETRIEVAL_PQ_TRAINER_MOBILE_C = 1,
    RETRIEVAL_PQ_TRAINER_SERVER_NUMPY = 2,
    RETRIEVAL_PQ_TRAINER_SERVER_FAISS = 3
} RetrievalPqTrainerKind;

typedef enum RetrievalPqSampleStrategy {
    RETRIEVAL_PQ_SAMPLE_UNKNOWN = 0,
    RETRIEVAL_PQ_SAMPLE_UNIFORM_STRIDE = 1
} RetrievalPqSampleStrategy;

typedef struct RetrievalPqArtifactMetadata {
    uint8_t index_uuid[RETRIEVAL_PQ_ARTIFACT_UUID_BYTES];
    uint64_t num_vectors;
    uint32_t dim;
    uint32_t m;
    uint32_t ks;
    uint32_t subdim;
    uint32_t code_bytes;
    uint32_t trainer_kind;
    uint32_t sample_strategy;
    uint64_t train_count;
    uint32_t kmeans_iters;
    uint32_t seed;
    uint8_t dataset_fingerprint[RETRIEVAL_PQ_ARTIFACT_HASH_BYTES];
    uint8_t layout_fingerprint[RETRIEVAL_PQ_ARTIFACT_HASH_BYTES];
} RetrievalPqArtifactMetadata;

typedef struct RetrievalPqArtifacts {
    RetrievalPqArtifactMetadata metadata;
    float *codebook;
    uint8_t *codes;
    size_t codebook_values;
    size_t codes_bytes;
} RetrievalPqArtifacts;

typedef struct RetrievalPqArtifactWriter RetrievalPqArtifactWriter;

int retrieval_pq_artifact_validate_metadata(
    const RetrievalPqArtifactMetadata *metadata);

int retrieval_pq_artifacts_write(const char *artifact_prefix,
                                 const RetrievalPqArtifactMetadata *metadata,
                                 const float *codebook,
                                 const uint8_t *codes);

int retrieval_pq_artifact_writer_begin(
    const char *artifact_prefix,
    const RetrievalPqArtifactMetadata *metadata,
    const float *codebook,
    RetrievalPqArtifactWriter **out_writer);

int retrieval_pq_artifact_writer_append_codes(
    RetrievalPqArtifactWriter *writer,
    const uint8_t *codes,
    size_t code_bytes);

int retrieval_pq_artifact_writer_commit(RetrievalPqArtifactWriter *writer);

void retrieval_pq_artifact_writer_abort(RetrievalPqArtifactWriter *writer);

int retrieval_pq_artifact_hash_bytes(const void *data,
                                     size_t bytes,
                                     uint8_t out_hash[32]);

int retrieval_pq_artifact_hash_file_prefix(const char *path,
                                           uint64_t bytes,
                                           uint8_t out_hash[32]);

int retrieval_pq_artifact_hash_file_range(const char *path,
                                          uint64_t offset,
                                          uint64_t bytes,
                                          uint8_t out_hash[32]);

int retrieval_pq_artifacts_load(const char *artifact_prefix,
                                RetrievalPqArtifacts *out_artifacts);

void retrieval_pq_artifacts_release(RetrievalPqArtifacts *artifacts);

#ifdef __cplusplus
}
#endif

#endif
