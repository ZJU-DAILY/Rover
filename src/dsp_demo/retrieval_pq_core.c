#include "dsp_demo/retrieval_pq_core.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__hexagon__)
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#endif

static float pq_sub_l2(const float *a, const float *b, int subdim) {
    float acc = 0.0f;
    int d;

    for (d = 0; d < subdim; d++) {
        float diff = a[d] - b[d];
        acc += diff * diff;
    }
    return acc;
}

static int pq_size_mul(size_t a, size_t b, size_t *out) {
    if (!out || (a != 0u && b > SIZE_MAX / a)) {
        return -1;
    }
    *out = a * b;
    return 0;
}

int retrieval_pq_code_bits(int ks) {
    if (ks > 0 && ks <= 16) {
        return 4;
    }
    if (ks > 16 && ks <= 256) {
        return 8;
    }
    return -1;
}

int retrieval_pq_code_bytes(int m, int ks) {
    int bits = retrieval_pq_code_bits(ks);

    if (m <= 0 || bits < 0) {
        return -1;
    }
    return bits == 4 ? m / 2 + m % 2 : m;
}

uint8_t retrieval_pq_code_get(const uint8_t *code, int subspace, int ks) {
    if (!code || subspace < 0 || retrieval_pq_code_bits(ks) < 0) {
        return UINT8_MAX;
    }
    if (ks <= 16) {
        uint8_t packed = code[subspace >> 1];
        return (subspace & 1) ? (uint8_t)(packed >> 4)
                              : (uint8_t)(packed & 0x0fu);
    }
    return code[subspace];
}

int retrieval_pq_code_set(uint8_t *code,
                          int subspace,
                          int ks,
                          uint8_t value) {
    if (!code || subspace < 0 || (int)value >= ks ||
        retrieval_pq_code_bits(ks) < 0) {
        return -1;
    }
    if (ks <= 16) {
        uint8_t *packed = code + (subspace >> 1);
        if (subspace & 1) {
            *packed = (uint8_t)((*packed & 0x0fu) | (uint8_t)(value << 4));
        } else {
            *packed = (uint8_t)((*packed & 0xf0u) | value);
        }
    } else {
        code[subspace] = value;
    }
    return 0;
}

int retrieval_pq_validate_config(const RetrievalPqConfig *config) {
    if (!config || config->m <= 0 || config->ks <= 0 ||
        config->ks > 256 || config->subdim <= 0 || config->dim <= 0 ||
        config->m > INT_MAX / config->subdim ||
        config->m * config->subdim != config->dim) {
        return -1;
    }
    return 0;
}

int retrieval_pq_train_subspace(const float *vectors,
                                int vector_count,
                                int vector_stride,
                                int ks,
                                int subdim,
                                int kmeans_iters,
                                float *out_centroids) {
    float *sums = NULL;
    int *counts = NULL;
    size_t centroid_values;
    const int iters = (kmeans_iters > 0) ? kmeans_iters : 1;
    int k;
    int iter;

    if (!vectors || !out_centroids || vector_count <= 0 ||
        vector_stride < subdim || ks <= 0 || ks > 256 || subdim <= 0 ||
        pq_size_mul((size_t)ks, (size_t)subdim, &centroid_values) != 0 ||
        centroid_values > SIZE_MAX / sizeof(float)) {
        return -1;
    }

    {
        int vector_index;
        for (vector_index = 0; vector_index < vector_count; vector_index++) {
            const float *vector = vectors +
                (size_t)vector_index * (size_t)vector_stride;
            int dimension;
            for (dimension = 0; dimension < subdim; dimension++) {
                if (!isfinite(vector[dimension])) {
                    return -1;
                }
            }
        }
    }

    sums = (float *)malloc(centroid_values * sizeof(float));
    counts = (int *)malloc((size_t)ks * sizeof(int));
    if (!sums || !counts) {
        free(sums);
        free(counts);
        return -1;
    }

    for (k = 0; k < ks; k++) {
        int src = (vector_count == 1)
                      ? 0
                      : (int)(((long long)k * (long long)(vector_count - 1)) /
                              (long long)((ks > 1) ? (ks - 1) : 1));
        const float *source = vectors + (size_t)src * (size_t)vector_stride;
        memcpy(out_centroids + (size_t)k * (size_t)subdim,
               source,
               (size_t)subdim * sizeof(float));
    }

    for (iter = 0; iter < iters; iter++) {
        int i;

        memset(sums, 0, (size_t)ks * (size_t)subdim * sizeof(float));
        memset(counts, 0, (size_t)ks * sizeof(int));

        for (i = 0; i < vector_count; i++) {
            const float *vec = vectors + (size_t)i * (size_t)vector_stride;
            float best = FLT_MAX;
            int best_k = 0;

            for (k = 0; k < ks; k++) {
                float dist = pq_sub_l2(vec,
                                       out_centroids +
                                           (size_t)k * (size_t)subdim,
                                       subdim);
                if (dist < best) {
                    best = dist;
                    best_k = k;
                }
            }

            counts[best_k]++;
            for (k = 0; k < subdim; k++) {
                sums[(size_t)best_k * (size_t)subdim + (size_t)k] += vec[k];
            }
        }

        for (k = 0; k < ks; k++) {
            if (counts[k] > 0) {
                int d;
                float inv = 1.0f / (float)counts[k];
                for (d = 0; d < subdim; d++) {
                    out_centroids[(size_t)k * (size_t)subdim + (size_t)d] =
                        sums[(size_t)k * (size_t)subdim + (size_t)d] * inv;
                }
            }
        }
    }

    free(sums);
    free(counts);
    return 0;
}

int retrieval_pq_train(const float *train_vectors,
                       int train_count,
                       const RetrievalPqConfig *config,
                       float *out_codebook) {
    int subspace;

    if (!train_vectors || !out_codebook || train_count <= 0 ||
        retrieval_pq_validate_config(config) != 0) {
        return -1;
    }

    for (subspace = 0; subspace < config->m; subspace++) {
        float *centroids = out_codebook +
            (size_t)subspace * (size_t)config->ks * (size_t)config->subdim;
        const float *subspace_vectors = train_vectors +
            (size_t)subspace * (size_t)config->subdim;

        if (retrieval_pq_train_subspace(subspace_vectors,
                                        train_count,
                                        config->dim,
                                        config->ks,
                                        config->subdim,
                                        config->kmeans_iters,
                                        centroids) != 0) {
            return -1;
        }
    }
    return 0;
}

int retrieval_pq_encode_batch(const float *vectors,
                              int vector_count,
                              const float *codebook,
                              const RetrievalPqConfig *config,
                              uint8_t *out_codes) {
    int code_bytes;
    int vector_index;

    if (!vectors || !codebook || !out_codes || vector_count < 0 ||
        retrieval_pq_validate_config(config) != 0 ||
        (code_bytes = retrieval_pq_code_bytes(config->m, config->ks)) <= 0) {
        return -1;
    }

    for (vector_index = 0; vector_index < vector_count; vector_index++) {
        uint8_t *vector_code = out_codes +
            (size_t)vector_index * (size_t)code_bytes;
        int subspace;

        memset(vector_code, 0, (size_t)code_bytes);

        for (subspace = 0; subspace < config->m; subspace++) {
            const float *vector = vectors +
                (size_t)vector_index * (size_t)config->dim +
                (size_t)subspace * (size_t)config->subdim;
            const float *centroids = codebook +
                (size_t)subspace * (size_t)config->ks *
                    (size_t)config->subdim;
            float best = FLT_MAX;
            int best_k = 0;
            int k;

            for (k = 0; k < config->ks; k++) {
                float distance = pq_sub_l2(
                    vector,
                    centroids + (size_t)k * (size_t)config->subdim,
                    config->subdim);
                if (!isfinite(distance)) {
                    return -1;
                }
                if (distance < best) {
                    best = distance;
                    best_k = k;
                }
            }
            if (retrieval_pq_code_set(vector_code, subspace, config->ks,
                                      (uint8_t)best_k) != 0) {
                return -1;
            }
        }
    }
    return 0;
}

static int retrieval_pq_adc_ids_batch_impl(
        const uint8_t *const *code_chunks,
        const int *chunk_vector_counts,
        int chunk_count,
        int chunk_vectors,
        const int32_t *ids,
        int count,
        const float *lut,
        int m,
        int ks,
        float *out_distances,
        int force_scalar) {
    int code_bytes;
    int index;

    if (!code_chunks || !chunk_vector_counts || !ids || !lut ||
        !out_distances || chunk_count <= 0 || chunk_vectors <= 0 ||
        count < 0 || m <= 0 || ks <= 0 || ks > 256 ||
        (code_bytes = retrieval_pq_code_bytes(m, ks)) <= 0) {
        return -1;
    }

    for (index = 0; index < count; index++) {
        int32_t id = ids[index];
        int chunk_index;
        int chunk_offset;
        const uint8_t *code;
        float distance;

        if (id < 0) {
            return -1;
        }
        chunk_index = id / chunk_vectors;
        chunk_offset = id % chunk_vectors;
        if (chunk_index < 0 || chunk_index >= chunk_count ||
            !code_chunks[chunk_index] || chunk_vector_counts[chunk_index] < 0 ||
            chunk_offset >= chunk_vector_counts[chunk_index]) {
            return -1;
        }
        code = code_chunks[chunk_index] +
            (size_t)chunk_offset * (size_t)code_bytes;
        distance = force_scalar
            ? retrieval_pq_adc_scalar(code, lut, m, ks)
            : retrieval_pq_adc_dispatch(code, lut, m, ks);
        if (!isfinite(distance) || distance == FLT_MAX) {
            return -1;
        }
        out_distances[index] = distance;
    }
    return 0;
}

int retrieval_pq_adc_ids_batch(const uint8_t *const *code_chunks,
                               const int *chunk_vector_counts,
                               int chunk_count,
                               int chunk_vectors,
                               const int32_t *ids,
                               int count,
                               const float *lut,
                               int m,
                               int ks,
                               float *out_distances) {
    return retrieval_pq_adc_ids_batch_impl(
        code_chunks, chunk_vector_counts, chunk_count, chunk_vectors,
        ids, count, lut, m, ks, out_distances, 0);
}

int retrieval_pq_adc_ids_batch_scalar(const uint8_t *const *code_chunks,
                                      const int *chunk_vector_counts,
                                      int chunk_count,
                                      int chunk_vectors,
                                      const int32_t *ids,
                                      int count,
                                      const float *lut,
                                      int m,
                                      int ks,
                                      float *out_distances) {
    return retrieval_pq_adc_ids_batch_impl(
        code_chunks, chunk_vector_counts, chunk_count, chunk_vectors,
        ids, count, lut, m, ks, out_distances, 1);
}

int retrieval_pq_build_lut(const float *query,
                           const float *codebook,
                           int m,
                           int ks,
                           int subdim,
                           float *lut) {
    int mi;
    int k;

    if (!query || !codebook || !lut || m <= 0 || ks <= 0 || ks > 256 ||
        subdim <= 0) {
        return -1;
    }

    for (mi = 0; mi < m; mi++) {
        const float *q = query + (size_t)mi * (size_t)subdim;
        for (k = 0; k < ks; k++) {
            const float *centroid = codebook +
                ((size_t)mi * (size_t)ks + (size_t)k) * (size_t)subdim;
            float distance = pq_sub_l2(q, centroid, subdim);
            if (!isfinite(distance)) {
                return -1;
            }
            lut[(size_t)mi * (size_t)ks + (size_t)k] = distance;
        }
    }

    return 0;
}

float retrieval_pq_adc_scalar(const uint8_t *code,
                              const float *lut,
                              int m,
                              int ks) {
    float acc = 0.0f;
    int mi;

    if (!code || !lut || m <= 0 || ks <= 0 || ks > 256) {
        return FLT_MAX;
    }

    for (mi = 0; mi < m; mi++) {
        uint8_t centroid = retrieval_pq_code_get(code, mi, ks);

        if ((int)centroid >= ks ||
            !isfinite(lut[(size_t)mi * (size_t)ks + (size_t)centroid])) {
            return FLT_MAX;
        }
        acc += lut[(size_t)mi * (size_t)ks + (size_t)centroid];
        if (!isfinite(acc)) {
            return FLT_MAX;
        }
    }
    return acc;
}

#if defined(__hexagon__) && !defined(RETRIEVAL_FORCE_SCALAR) && \
    !defined(RETRIEVAL_PQ_FORCE_SCALAR)

#define RETRIEVAL_PQ_HVX_WORDS 32

typedef union RetrievalPqHvxVector {
    HVX_Vector vector;
    uint32_t words[RETRIEVAL_PQ_HVX_WORDS];
} RetrievalPqHvxVector;

static float retrieval_pq_adc_hvx(const uint8_t *code,
                                  const float *lut,
                                  int m,
                                  int ks,
                                  HVX_Vector *gather_buffer) {
    HVX_Vector accumulator = Q6_V_vzero();
    const HVX_Vector zero = Q6_V_vzero();
    const size_t lut_values = (size_t)m * (size_t)ks;
    const uint32_t lut_last_valid =
        (uint32_t)(lut_values * sizeof(float) - 1u);
    const int32_t lut_address = (int32_t)(uintptr_t)lut;
    int subspace = 0;

    for (; subspace + RETRIEVAL_PQ_HVX_WORDS <= m;
         subspace += RETRIEVAL_PQ_HVX_WORDS) {
        RetrievalPqHvxVector offsets;
        RetrievalPqHvxVector gathered;
        int lane;

        for (lane = 0; lane < RETRIEVAL_PQ_HVX_WORDS; lane++) {
            const int current_subspace = subspace + lane;
            const uint8_t centroid =
                retrieval_pq_code_get(code, current_subspace, ks);
            offsets.words[lane] =
                ((uint32_t)current_subspace * (uint32_t)ks + centroid) *
                (uint32_t)sizeof(float);
        }

        Q6_vgather_ARMVw(gather_buffer, lut_address, lut_last_valid,
                         offsets.vector);
        gathered.vector = *(volatile HVX_Vector *)gather_buffer;
        accumulator = Q6_Vqf32_vadd_Vqf32Vqf32(
            accumulator,
            Q6_Vqf32_vadd_VsfVsf(gathered.vector, zero));
    }

    accumulator = Q6_Vqf32_vadd_Vqf32Vqf32(
        accumulator, Q6_V_vror_VR(accumulator, 64));
    accumulator = Q6_Vqf32_vadd_Vqf32Vqf32(
        accumulator, Q6_V_vror_VR(accumulator, 32));
    accumulator = Q6_Vqf32_vadd_Vqf32Vqf32(
        accumulator, Q6_V_vror_VR(accumulator, 16));
    accumulator = Q6_Vqf32_vadd_Vqf32Vqf32(
        accumulator, Q6_V_vror_VR(accumulator, 8));
    accumulator = Q6_Vqf32_vadd_Vqf32Vqf32(
        accumulator, Q6_V_vror_VR(accumulator, 4));

    {
        RetrievalPqHvxVector reduced;
        float total;

        reduced.vector = Q6_Vsf_equals_Vqf32(accumulator);
        memcpy(&total, &reduced.words[0], sizeof(total));
        for (; subspace < m; subspace++) {
            const uint8_t centroid = retrieval_pq_code_get(code, subspace, ks);
            total += lut[(size_t)subspace * (size_t)ks + centroid];
        }
        return isfinite(total) ? total : FLT_MAX;
    }
}

#endif

float retrieval_pq_adc_dispatch(const uint8_t *code,
                                const float *lut,
                                int m,
                                int ks) {
    return retrieval_pq_adc_scalar(code, lut, m, ks);
}

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
                                   float *out_distances) {
#if defined(__hexagon__) && !defined(RETRIEVAL_FORCE_SCALAR) && \
    !defined(RETRIEVAL_PQ_FORCE_SCALAR)
    int code_bytes;
    int index;

    if (!code_chunks || !chunk_vector_counts || !ids || !lut ||
        !gather_buffer || !out_distances || chunk_count <= 0 ||
        chunk_vectors <= 0 || count < 0 || m < RETRIEVAL_PQ_HVX_WORDS ||
        (ks != 16 && ks != 256) ||
        (code_bytes = retrieval_pq_code_bytes(m, ks)) <= 0 ||
        (size_t)m * (size_t)ks > UINT32_MAX / sizeof(float) ||
        ((uintptr_t)gather_buffer & (sizeof(HVX_Vector) - 1u)) != 0u) {
        return -1;
    }

    for (index = 0; index < count; index++) {
        int32_t id = ids[index];
        int chunk_index;
        int chunk_offset;
        const uint8_t *code;
        float distance;

        if (id < 0) {
            return -1;
        }
        chunk_index = id / chunk_vectors;
        chunk_offset = id % chunk_vectors;
        if (chunk_index < 0 || chunk_index >= chunk_count ||
            !code_chunks[chunk_index] || chunk_vector_counts[chunk_index] < 0 ||
            chunk_offset >= chunk_vector_counts[chunk_index]) {
            return -1;
        }
        code = code_chunks[chunk_index] +
            (size_t)chunk_offset * (size_t)code_bytes;
        distance = retrieval_pq_adc_hvx(
            code, lut, m, ks, (HVX_Vector *)gather_buffer);
        if (!isfinite(distance) || distance == FLT_MAX) {
            return -1;
        }
        out_distances[index] = distance;
    }
    return 0;
#else
    (void)gather_buffer;
    return retrieval_pq_adc_ids_batch_scalar(
        code_chunks, chunk_vector_counts, chunk_count, chunk_vectors,
        ids, count, lut, m, ks, out_distances);
#endif
}
