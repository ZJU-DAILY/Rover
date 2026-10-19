#ifndef QUANTIZE_COMPRESS_H
#define QUANTIZE_COMPRESS_H

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif


typedef struct {
    float scale;
    float zero_point;
} QCQuantParams;

typedef struct {
    uint32_t compressed_size;
    uint8_t *compressed_data;
} QCCompressedVector;

typedef struct {
    float scale;
    float zero_point;
} QCResidualQuantParams;

typedef struct {
    int num_vectors;
    int dim;
    QCQuantParams global_qparams;


    QCCompressedVector *compressed_vectors;


    int use_residual;


    QCResidualQuantParams *residual_qparams;
    uint8_t *residual_codes;


    float *original_data;
} QCDataset;


void qc_compute_global_params(const float *data, int total_elems, QCQuantParams *out);
void qc_quantize_vector(const float *in, int dim, const QCQuantParams *qp, uint8_t *out);
void qc_dequantize_vector(const uint8_t *in, int dim, const QCQuantParams *qp, float *out);
void qc_compute_residual(const float *original, const float *dequantized, int dim, float *residual);
void qc_quantize_residual(const float *residual, int dim, QCResidualQuantParams *rqp, uint8_t *codes);
void qc_dequantize_residual(const uint8_t *codes, int dim, const QCResidualQuantParams *rqp, float *out);
int  qc_encode_dataset(const float *data, int num_vectors, int dim, int use_residual, QCDataset *ds);
int  qc_decode_vector(const QCDataset *ds, int vec_idx, float *out);
void qc_free_dataset(QCDataset *ds);

#ifdef __cplusplus
}
#endif

#endif
