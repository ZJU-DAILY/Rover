#ifndef RETRIEVAL_L1_INT8_CORE_H
#define RETRIEVAL_L1_INT8_CORE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int retrieval_quantize_query_u8(const float *query,
                                int dim,
                                float scale,
                                float zero_point,
                                uint8_t *out_query_u8);

int retrieval_quantize_query_u4(const float *query,
                                int dim,
                                float scale,
                                float zero_point,
                                uint8_t *out_query_u4);

typedef struct RetrievalQueryU8Stats {
    uint64_t sum_q;
    uint64_t clipped_low;
    uint64_t clipped_high;
    double reconstruction_error_sq;
    double query_norm_sq;
} RetrievalQueryU8Stats;

int retrieval_quantize_query_l1_aligned_u8(
    const float *query,
    int dim,
    float l1_scale,
    float l1_zero_point,
    uint8_t *out_query_u8,
    RetrievalQueryU8Stats *out_stats);


float retrieval_compute_int8_native_distance_u8(const uint8_t *query_u8,
                                                const uint8_t *db_u8,
                                                int dim,
                                                float scale);


float retrieval_compute_int8_native_distance_u8_scalar(const uint8_t *query_u8,
                                                       const uint8_t *db_u8,
                                                       int dim,
                                                       float scale);


float retrieval_compute_int4_packed_distance_u8(const uint8_t *query_u4,
                                                const uint8_t *db_packed_u4,
                                                int dim,
                                                float scale);

float retrieval_compute_int4_packed_distance_u8_scalar(const uint8_t *query_u4,
                                                       const uint8_t *db_packed_u4,
                                                       int dim,
                                                       float scale);

float retrieval_compute_u8_u4_l1_distance(
    const uint8_t *query_u8,
    const uint8_t *db_packed_u4,
    int dim,
    float l1_scale);

float retrieval_compute_u8_u4_l1_distance_scalar(
    const uint8_t *query_u8,
    const uint8_t *db_packed_u4,
    int dim,
    float l1_scale);

double retrieval_compute_u8_u4_residual_dot_scalar(
    const uint8_t *query_u8,
    uint64_t sum_query_u8,
    const uint8_t *residual_packed_u4,
    int dim,
    float l1_scale,
    float l1_zero_point,
    float residual_scale,
    float residual_zero_point);


double retrieval_compute_u8_u8_residual_dot_scalar(
    const uint8_t *query_u8,
    uint64_t sum_query_u8,
    const uint8_t *residual_u8,
    int dim,
    float l1_scale,
    float l1_zero_point,
    float residual_scale,
    float residual_zero_point);


double retrieval_compute_u8_u4_residual_dot(
    const uint8_t *query, uint64_t sum_query, const uint8_t *residual, int dim,
    float l1_scale, float l1_zero, float residual_scale, float residual_zero);
double retrieval_compute_u8_u8_residual_dot(
    const uint8_t *query, uint64_t sum_query, const uint8_t *residual, int dim,
    float l1_scale, float l1_zero, float residual_scale, float residual_zero);

int retrieval_compute_int8_native_distance_batch(const uint8_t *query_u8,
                                                 const uint8_t *flat_db_u8,
                                                 int dim,
                                                 int num_vectors,
                                                 float scale,
                                                 float *distances);

#ifdef __cplusplus
}
#endif

#endif
