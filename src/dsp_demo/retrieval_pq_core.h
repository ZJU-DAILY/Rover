#ifndef RETRIEVAL_PQ_CORE_H
#define RETRIEVAL_PQ_CORE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RetrievalPqConfig {
    int m;
    int ks;
    int subdim;
    int dim;
    int train_max;
    int kmeans_iters;
} RetrievalPqConfig;

int retrieval_pq_validate_config(const RetrievalPqConfig *config);

int retrieval_pq_code_bits(int ks);

int retrieval_pq_code_bytes(int m, int ks);

uint8_t retrieval_pq_code_get(const uint8_t *code, int subspace, int ks);

int retrieval_pq_code_set(uint8_t *code,
                          int subspace,
                          int ks,
                          uint8_t value);

int retrieval_pq_train_subspace(const float *vectors,
                                int vector_count,
                                int vector_stride,
                                int ks,
                                int subdim,
                                int kmeans_iters,
                                float *out_centroids);

int retrieval_pq_train(const float *train_vectors,
                       int train_count,
                       const RetrievalPqConfig *config,
                       float *out_codebook);

int retrieval_pq_encode_batch(const float *vectors,
                              int vector_count,
                              const float *codebook,
                              const RetrievalPqConfig *config,
                              uint8_t *out_codes);

int retrieval_pq_adc_ids_batch(const uint8_t *const *code_chunks,
                               const int *chunk_vector_counts,
                               int chunk_count,
                               int chunk_vectors,
                               const int32_t *ids,
                               int count,
                               const float *lut,
                               int m,
                               int ks,
                               float *out_distances);

int retrieval_pq_adc_ids_batch_scalar(const uint8_t *const *code_chunks,
                                      const int *chunk_vector_counts,
                                      int chunk_count,
                                      int chunk_vectors,
                                      const int32_t *ids,
                                      int count,
                                      const float *lut,
                                      int m,
                                      int ks,
                                      float *out_distances);

int retrieval_pq_adc_ids_batch_hvx(const uint8_t *const *code_chunks,
                                   const int *chunk_vector_counts,
                                   int chunk_count,
                                   int chunk_vectors,
                                   const int32_t *ids,
                                   int count,
                                   const float *lut,
                                   int m,
                                   int ks,
                                   void *gather_buffer,
                                   float *out_distances);

int retrieval_pq_build_lut(const float *query,
                           const float *codebook,
                           int m,
                           int ks,
                           int subdim,
                           float *lut);

float retrieval_pq_adc_scalar(const uint8_t *code,
                              const float *lut,
                              int m,
                              int ks);

float retrieval_pq_adc_dispatch(const uint8_t *code,
                                const float *lut,
                                int m,
                                int ks);

#ifdef __cplusplus
}
#endif

#endif
