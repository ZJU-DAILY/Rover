#include "dsp_demo/retrieval_l1_int8_core.h"

#include <float.h>
#include <math.h>
#include <string.h>

#if defined(__hexagon__)
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#define HVX_ALWAYS_INLINE static inline __attribute__((unused,always_inline))
#endif

int retrieval_quantize_query_u8(const float *query,
                                int dim,
                                float scale,
                                float zero_point,
                                uint8_t *out_query_u8) {
    int i = 0;
    float inv_scale = 0.0f;

    if (!query || !out_query_u8 || dim <= 0 || scale == 0.0f) {
        return -1;
    }

    inv_scale = 1.0f / scale;
    for (i = 0; i < dim; i++) {
        int v = (int)roundf((query[i] - zero_point) * inv_scale);
        if (v < 0) {
            v = 0;
        } else if (v > 255) {
            v = 255;
        }
        out_query_u8[i] = (uint8_t)v;
    }

    return 0;
}

int retrieval_quantize_query_u4(const float *query,
                                int dim,
                                float scale,
                                float zero_point,
                                uint8_t *out_query_u4) {
    int i = 0;
    float inv_scale = 0.0f;

    if (!query || !out_query_u4 || dim <= 0 || scale == 0.0f) {
        return -1;
    }

    inv_scale = 1.0f / scale;
    for (i = 0; i < dim; i++) {
        int v = (int)roundf((query[i] - zero_point) * inv_scale);
        if (v < 0) {
            v = 0;
        } else if (v > 15) {
            v = 15;
        }
        out_query_u4[i] = (uint8_t)v;
    }

    return 0;
}

int retrieval_quantize_query_l1_aligned_u8(
        const float *query,
        int dim,
        float l1_scale,
        float l1_zero_point,
        uint8_t *out_query_u8,
        RetrievalQueryU8Stats *out_stats) {
    RetrievalQueryU8Stats stats;
    const double query_scale = (double)l1_scale / 17.0;
    int i;

    if (!query || !out_query_u8 || dim <= 0 || !isfinite(l1_scale) ||
        !isfinite(l1_zero_point) || l1_scale <= 0.0f ||
        !isfinite(query_scale) || query_scale <= 0.0)
        return -1;

    memset(&stats, 0, sizeof(stats));
    for (i = 0; i < dim; i++) {
        const double raw = ((double)query[i] - (double)l1_zero_point) /
                           query_scale;
        long code;
        double reconstructed;
        double error;

        if (!isfinite(query[i]) || !isfinite(raw))
            return -1;
        code = lround(raw);
        if (code < 0) {
            code = 0;
            stats.clipped_low++;
        } else if (code > 255) {
            code = 255;
            stats.clipped_high++;
        }
        out_query_u8[i] = (uint8_t)code;
        stats.sum_q += (uint64_t)code;
        reconstructed = query_scale * (double)code +
                        (double)l1_zero_point;
        error = (double)query[i] - reconstructed;
        stats.reconstruction_error_sq += error * error;
        stats.query_norm_sq += (double)query[i] * (double)query[i];
    }

    if (out_stats)
        *out_stats = stats;
    return 0;
}

static uint8_t retrieval_unpack_packed_u4_at(const uint8_t *db_packed_u4, int index) {
    uint8_t packed = db_packed_u4[index >> 1];
    return (uint8_t)(((index & 1) == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F));
}


float retrieval_compute_int8_native_distance_u8_scalar(const uint8_t *query_u8,
                                                       const uint8_t *db_u8,
                                                       int dim,
                                                       float scale) {
    int i = 0;
    int64_t acc = 0;
    float scale_sq = scale * scale;

    if (!query_u8 || !db_u8 || dim <= 0) {
        return FLT_MAX;
    }

    for (i = 0; i + 3 < dim; i += 4) {
        int16_t d0 = (int16_t)query_u8[i] - (int16_t)db_u8[i];
        int16_t d1 = (int16_t)query_u8[i + 1] - (int16_t)db_u8[i + 1];
        int16_t d2 = (int16_t)query_u8[i + 2] - (int16_t)db_u8[i + 2];
        int16_t d3 = (int16_t)query_u8[i + 3] - (int16_t)db_u8[i + 3];
        acc += (int32_t)d0 * d0;
        acc += (int32_t)d1 * d1;
        acc += (int32_t)d2 * d2;
        acc += (int32_t)d3 * d3;
    }

    for (; i < dim; i++) {
        int16_t d = (int16_t)query_u8[i] - (int16_t)db_u8[i];
        acc += (int32_t)d * d;
    }

    return (float)acc * scale_sq;
}

float retrieval_compute_int4_packed_distance_u8_scalar(const uint8_t *query_u4,
                                                       const uint8_t *db_packed_u4,
                                                       int dim,
                                                       float scale) {
    int i = 0;
    int64_t acc = 0;
    float scale_sq = scale * scale;

    if (!query_u4 || !db_packed_u4 || dim <= 0) {
        return FLT_MAX;
    }

    for (i = 0; i < dim; i++) {
        int16_t d = (int16_t)(query_u4[i] & 0x0F) -
                    (int16_t)retrieval_unpack_packed_u4_at(db_packed_u4, i);
        acc += (int32_t)d * d;
    }

    return (float)acc * scale_sq;
}

float retrieval_compute_u8_u4_l1_distance_scalar(
        const uint8_t *query_u8,
        const uint8_t *db_packed_u4,
        int dim,
        float l1_scale) {
    int64_t acc = 0;
    const double query_scale = (double)l1_scale / 17.0;
    int i;

    if (!query_u8 || !db_packed_u4 || dim <= 0 || dim > 33025 ||
        !isfinite(l1_scale) || l1_scale <= 0.0f)
        return FLT_MAX;

    for (i = 0; i < dim; i++) {
        const int32_t mapped_l1 =
            17 * (int32_t)retrieval_unpack_packed_u4_at(db_packed_u4, i);
        const int32_t diff = (int32_t)query_u8[i] - mapped_l1;
        acc += (int64_t)diff * (int64_t)diff;
    }
    return (float)((double)acc * query_scale * query_scale);
}

double retrieval_compute_u8_u4_residual_dot_scalar(
        const uint8_t *query_u8,
        uint64_t sum_query_u8,
        const uint8_t *residual_packed_u4,
        int dim,
        float l1_scale,
        float l1_zero_point,
        float residual_scale,
        float residual_zero_point) {
    uint64_t sum_qv = 0;
    uint64_t sum_v = 0;
    const double query_scale = (double)l1_scale / 17.0;
    int i;

    if (!query_u8 || !residual_packed_u4 || dim <= 0 ||
        !isfinite(l1_scale) || !isfinite(l1_zero_point) ||
        !isfinite(residual_scale) || !isfinite(residual_zero_point) ||
        l1_scale <= 0.0f || residual_scale < 0.0f)
        return NAN;

    for (i = 0; i < dim; i++) {
        const uint8_t v = retrieval_unpack_packed_u4_at(
            residual_packed_u4, i);
        sum_qv += (uint64_t)query_u8[i] * (uint64_t)v;
        sum_v += (uint64_t)v;
    }

    return query_scale * (double)residual_scale * (double)sum_qv +
           query_scale * (double)residual_zero_point *
               (double)sum_query_u8 +
           (double)l1_zero_point * (double)residual_scale *
               (double)sum_v +
           (double)dim * (double)l1_zero_point *
               (double)residual_zero_point;
}

double retrieval_compute_u8_u8_residual_dot_scalar(
        const uint8_t *query_u8,
        uint64_t sum_query_u8,
        const uint8_t *residual_u8,
        int dim,
        float l1_scale,
        float l1_zero_point,
        float residual_scale,
        float residual_zero_point) {
    uint64_t sum_qv = 0;
    uint64_t sum_v = 0;
    const double query_scale = (double)l1_scale / 17.0;
    int i;

    if (!query_u8 || !residual_u8 || dim <= 0 ||
        !isfinite(l1_scale) || !isfinite(l1_zero_point) ||
        !isfinite(residual_scale) || !isfinite(residual_zero_point) ||
        l1_scale <= 0.0f || residual_scale < 0.0f)
        return NAN;

    for (i = 0; i < dim; i++) {
        sum_qv += (uint64_t)query_u8[i] * (uint64_t)residual_u8[i];
        sum_v += (uint64_t)residual_u8[i];
    }

    return query_scale * (double)residual_scale * (double)sum_qv +
           query_scale * (double)residual_zero_point *
               (double)sum_query_u8 +
           (double)l1_zero_point * (double)residual_scale *
               (double)sum_v +
           (double)dim * (double)l1_zero_point *
               (double)residual_zero_point;
}

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
static double retrieval_residual_dot_neon(const uint8_t *query_u8,
    uint64_t sum_query_u8, const uint8_t *residual, int dim, int bits,
    float l1_scale, float l1_zero_point, float residual_scale, float residual_zero_point) {
    uint64_t sum_qv=0, sum_v=0;
    const double query_scale=(double)l1_scale/17.0;
    int i=0;
    if (!query_u8 || !residual || dim<=0 || !isfinite(l1_scale) ||
        !isfinite(l1_zero_point) || !isfinite(residual_scale) ||
        !isfinite(residual_zero_point) || l1_scale<=0 || residual_scale<0) return NAN;

    for (; dim-i>=16; i+=16) {
        uint8x16_t q=vld1q_u8(query_u8+i), v;
        if (bits==4) {
            uint8x8_t packed=vld1_u8(residual+i/2);
            uint8x8x2_t pair=vzip_u8(vand_u8(packed,vdup_n_u8(15)),vshr_n_u8(packed,4));
            v=vcombine_u8(pair.val[0],pair.val[1]);
        } else v=vld1q_u8(residual+i);
        uint16x8_t lo=vmull_u8(vget_low_u8(q),vget_low_u8(v));
        uint16x8_t hi=vmull_u8(vget_high_u8(q),vget_high_u8(v));
        sum_qv+=(uint64_t)vaddvq_u32(vpaddlq_u16(lo))+
                (uint64_t)vaddvq_u32(vpaddlq_u16(hi));
        sum_v+=(uint64_t)vaddlvq_u8(v);
    }
    for (;i<dim;i++) {
        uint8_t v=bits==4?retrieval_unpack_packed_u4_at(residual,i):residual[i];
        sum_qv+=(uint64_t)query_u8[i]*v;sum_v+=v;
    }
    return query_scale * (double)residual_scale * (double)sum_qv +
           query_scale * (double)residual_zero_point * (double)sum_query_u8 +
           (double)l1_zero_point * (double)residual_scale * (double)sum_v +
           (double)dim * (double)l1_zero_point * (double)residual_zero_point;
}
#endif

double retrieval_compute_u8_u4_residual_dot(const uint8_t *q,uint64_t sum,
    const uint8_t *r,int dim,float ls,float lz,float rs,float rz) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    return retrieval_residual_dot_neon(q,sum,r,dim,4,ls,lz,rs,rz);
#else
    return retrieval_compute_u8_u4_residual_dot_scalar(q,sum,r,dim,ls,lz,rs,rz);
#endif
}
double retrieval_compute_u8_u8_residual_dot(const uint8_t *q,uint64_t sum,
    const uint8_t *r,int dim,float ls,float lz,float rs,float rz) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    return retrieval_residual_dot_neon(q,sum,r,dim,8,ls,lz,rs,rz);
#else
    return retrieval_compute_u8_u8_residual_dot_scalar(q,sum,r,dim,ls,lz,rs,rz);
#endif
}


#if defined(__hexagon__) && !defined(RETRIEVAL_FORCE_SCALAR)


HVX_ALWAYS_INLINE HVX_Vector
hvx_sq_diff_128(const uint8_t *q, const uint8_t *d, HVX_Vector acc) {
    HVX_Vector q_vec  = *((HVX_UVector *)q);
    HVX_Vector d_vec  = *((HVX_UVector *)d);

    HVX_VectorPair diff = Q6_Wh_vsub_VubVub(q_vec, d_vec);
    HVX_Vector diff_lo  = Q6_V_lo_W(diff);
    HVX_Vector diff_hi  = Q6_V_hi_W(diff);

    HVX_Vector sq_lo = Q6_Vw_vdmpy_VhVh_sat(diff_lo, diff_lo);
    HVX_Vector sq_hi = Q6_Vw_vdmpy_VhVh_sat(diff_hi, diff_hi);

    acc = Q6_Vw_vadd_VwVw(acc, sq_lo);
    acc = Q6_Vw_vadd_VwVw(acc, sq_hi);
    return acc;
}


HVX_ALWAYS_INLINE int32_t hvx_reduce_32w(HVX_Vector v) {
    union { HVX_Vector v; int32_t w[32]; } u;

    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 64));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 32));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 16));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 8));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 4));

    u.v = v;
    return u.w[0];
}


static float retrieval_compute_int8_native_distance_u8_hvx(
        const uint8_t *query_u8,
        const uint8_t *db_u8,
        int dim,
        float scale) {

    HVX_Vector acc;
    int offset;
    int32_t total;

    if (!query_u8 || !db_u8 || dim <= 0) {
        return FLT_MAX;
    }

    acc = Q6_V_vzero();


    for (offset = 0; offset + 128 <= dim; offset += 128) {
        acc = hvx_sq_diff_128(query_u8 + offset, db_u8 + offset, acc);
    }


    total = hvx_reduce_32w(acc);


    for (; offset < dim; offset++) {
        int16_t d = (int16_t)query_u8[offset] - (int16_t)db_u8[offset];
        total += (int32_t)d * d;
    }

    return (float)total * scale * scale;
}

HVX_ALWAYS_INLINE void hvx_unpack_u4_128(const uint8_t *src_packed_u4,
                                         uint8_t *dst_u4) {
    int i;

    for (i = 0; i < 64; i++) {
        uint8_t packed = src_packed_u4[i];
        dst_u4[2 * i]     = (uint8_t)(packed & 0x0F);
        dst_u4[2 * i + 1] = (uint8_t)((packed >> 4) & 0x0F);
    }
}

static float retrieval_compute_int4_packed_distance_u8_hvx(
        const uint8_t *query_u4,
        const uint8_t *db_packed_u4,
        int dim,
        float scale) {
    HVX_Vector acc;
    int offset;
    int32_t total;
    uint8_t unpacked_db_u4[128] __attribute__((aligned(128)));

    if (!query_u4 || !db_packed_u4 || dim <= 0) {
        return FLT_MAX;
    }

    acc = Q6_V_vzero();

    for (offset = 0; offset + 128 <= dim; offset += 128) {
        hvx_unpack_u4_128(db_packed_u4 + (offset >> 1), unpacked_db_u4);
        acc = hvx_sq_diff_128(query_u4 + offset, unpacked_db_u4, acc);
    }

    total = hvx_reduce_32w(acc);
    for (; offset < dim; offset++) {
        int16_t d = (int16_t)(query_u4[offset] & 0x0F) -
                    (int16_t)retrieval_unpack_packed_u4_at(db_packed_u4, offset);
        total += (int32_t)d * d;
    }

    return (float)total * scale * scale;
}

static float retrieval_compute_u8_u4_l1_distance_hvx(
        const uint8_t *query_u8,
        const uint8_t *db_packed_u4,
        int dim,
        float l1_scale) {
    HVX_Vector acc;
    int offset;
    int32_t total;
    uint8_t mapped_db_u8[128] __attribute__((aligned(128)));

    if (!query_u8 || !db_packed_u4 || dim <= 0 || dim > 33025 ||
        !isfinite(l1_scale) || l1_scale <= 0.0f)
        return FLT_MAX;

    acc = Q6_V_vzero();
    for (offset = 0; offset + 128 <= dim; offset += 128) {
        int i;
        hvx_unpack_u4_128(db_packed_u4 + (offset >> 1), mapped_db_u8);
        for (i = 0; i < 128; i++)
            mapped_db_u8[i] = (uint8_t)(17u * mapped_db_u8[i]);
        acc = hvx_sq_diff_128(query_u8 + offset, mapped_db_u8, acc);
    }

    total = hvx_reduce_32w(acc);
    for (; offset < dim; offset++) {
        const int32_t mapped_l1 = 17 * (int32_t)
            retrieval_unpack_packed_u4_at(db_packed_u4, offset);
        const int32_t diff = (int32_t)query_u8[offset] - mapped_l1;
        total += diff * diff;
    }
    return (float)total * (l1_scale / 17.0f) * (l1_scale / 17.0f);
}

#endif


float retrieval_compute_int8_native_distance_u8(const uint8_t *query_u8,
                                                const uint8_t *db_u8,
                                                int dim,
                                                float scale) {
#if defined(__hexagon__) && !defined(RETRIEVAL_FORCE_SCALAR)
    if (dim > 0) {
        return retrieval_compute_int8_native_distance_u8_hvx(query_u8, db_u8,
                                                             dim, scale);
    }
#endif
    return retrieval_compute_int8_native_distance_u8_scalar(query_u8, db_u8,
                                                            dim, scale);
}

float retrieval_compute_int4_packed_distance_u8(const uint8_t *query_u4,
                                                const uint8_t *db_packed_u4,
                                                int dim,
                                                float scale) {
#if defined(__hexagon__) && !defined(RETRIEVAL_FORCE_SCALAR)
    if (dim > 0) {
        return retrieval_compute_int4_packed_distance_u8_hvx(
            query_u4, db_packed_u4, dim, scale);
    }
#endif
    return retrieval_compute_int4_packed_distance_u8_scalar(
        query_u4, db_packed_u4, dim, scale);
}

float retrieval_compute_u8_u4_l1_distance(
        const uint8_t *query_u8,
        const uint8_t *db_packed_u4,
        int dim,
        float l1_scale) {
#if defined(__hexagon__) && !defined(RETRIEVAL_FORCE_SCALAR)
    if (dim > 0)
        return retrieval_compute_u8_u4_l1_distance_hvx(
            query_u8, db_packed_u4, dim, l1_scale);
#endif
    return retrieval_compute_u8_u4_l1_distance_scalar(
        query_u8, db_packed_u4, dim, l1_scale);
}

int retrieval_compute_int8_native_distance_batch(const uint8_t *query_u8,
                                                 const uint8_t *flat_db_u8,
                                                 int dim,
                                                 int num_vectors,
                                                 float scale,
                                                 float *distances) {
    int i = 0;

    if (!query_u8 || !flat_db_u8 || !distances || dim <= 0 || num_vectors <= 0) {
        return -1;
    }

    for (i = 0; i < num_vectors; i++) {
        distances[i] = retrieval_compute_int8_native_distance_u8(
            query_u8,
            flat_db_u8 + (i * dim),
            dim,
            scale
        );
    }

    return 0;
}
