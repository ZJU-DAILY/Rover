#ifndef RETRIEVAL_FINAL_REFINE_H
#define RETRIEVAL_FINAL_REFINE_H

#include <stdint.h>


enum { RETRIEVAL_REFINE_OFF = 0, RETRIEVAL_REFINE_CURRENT = 1,
       RETRIEVAL_REFINE_ERROR4 = 2 };
enum { RETRIEVAL_REFINE_INVALID = -1, RETRIEVAL_REFINE_IO = -2,
       RETRIEVAL_REFINE_ARTIFACT = -3, RETRIEVAL_REFINE_NUMERIC = -4 };
typedef int (*RetrievalRefineDecode)(void *provider, int graph_id, float *vector);
typedef struct RetrievalFinalRefineContext RetrievalFinalRefineContext;
typedef struct {
    int mode, num_vectors, dim, capacity, cache_slots;
    RetrievalRefineDecode decode;
    void *provider;
    const int32_t *graph_to_original;
    const int32_t *graph_to_storage;
    const char *artifact_path;
    const uint8_t *artifact_sha256;
    const char *base_paths[3];
} RetrievalFinalRefineConfig;
typedef struct {
    uint64_t candidates, cache_hits, code_blocks, code_bytes;
    uint64_t qparams_bytes, resident_bytes, artifact_bytes;
} RetrievalFinalRefineStats;

int retrieval_final_refine_open(const RetrievalFinalRefineConfig *config,
                               RetrievalFinalRefineContext **out);

int retrieval_final_refine_open_pq(const RetrievalFinalRefineConfig *config,
                                 const char *codebook_path,
                                 RetrievalFinalRefineContext **out);
void retrieval_final_refine_close(RetrievalFinalRefineContext *ctx);

int retrieval_final_refine_clone(const RetrievalFinalRefineContext *source,
    RetrievalRefineDecode decode, void *provider, int cache_slots,
    RetrievalFinalRefineContext **out);

void retrieval_final_refine_reset(RetrievalFinalRefineContext *ctx);

int retrieval_final_refine_read_vector(RetrievalFinalRefineContext *ctx,
                                      int graph_id, float *out);

int retrieval_final_refine(RetrievalFinalRefineContext *ctx, const float *query,
                          int32_t *ids, float *distances, int count, int output_k,
                          RetrievalFinalRefineStats *stats);

int retrieval_final_refine_score_ids(RetrievalFinalRefineContext *ctx, const float *query,
                                    const int32_t *ids, float *distances, int count);

int retrieval_final_refine_decode_sq(const uint8_t *l1, int l1_bits,
    float l1_scale, float l1_zero, const uint8_t *residual, int residual_bits,
    float residual_scale, float residual_zero, int dim, float *out);
#endif
