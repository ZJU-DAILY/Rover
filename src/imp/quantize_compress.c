#ifndef _QUANTIZE_COMPRESS_C
#define _QUANTIZE_COMPRESS_C

#include "quantize_compress.h"


void qc_compute_global_params(const float *data, int total_elems,
                                     QCQuantParams *out) {
  float gmin = FLT_MAX;
  float gmax = -FLT_MAX;
  for (int i = 0; i < total_elems; i++) {
    if (data[i] < gmin)
      gmin = data[i];
    if (data[i] > gmax)
      gmax = data[i];
  }
  float range = gmax - gmin;
  if (range < 1e-12f)
    range = 1e-12f;

  out->scale = range / 255.0f;
  out->zero_point = gmin;
}

void qc_quantize_vector(const float *in, int dim,
                               const QCQuantParams *qp, uint8_t *out) {
  float inv_scale = 1.0f / qp->scale;
  for (int i = 0; i < dim; i++) {
    float v = (in[i] - qp->zero_point) * inv_scale;
    int q = (int)roundf(v);
    if (q < 0)
      q = 0;
    if (q > 255)
      q = 255;
    out[i] = (uint8_t)q;
  }
}

void qc_dequantize_vector(const uint8_t *in, int dim,
                                 const QCQuantParams *qp, float *out) {
  for (int i = 0; i < dim; i++) {
    out[i] = (float)in[i] * qp->scale + qp->zero_point;
  }
}


void qc_compute_residual(const float *original, const float *dequantized,
                                int dim, float *residual) {
  for (int i = 0; i < dim; i++) {
    residual[i] = original[i] - dequantized[i];
  }
}

void qc_quantize_residual(const float *residual, int dim,
                                 QCResidualQuantParams *rqp, uint8_t *codes) {
  float rmin = FLT_MAX;
  float rmax = -FLT_MAX;
  for (int i = 0; i < dim; i++) {
    if (residual[i] < rmin)
      rmin = residual[i];
    if (residual[i] > rmax)
      rmax = residual[i];
  }
  float range = rmax - rmin;
  if (range < 1e-12f)
    range = 1e-12f;

  rqp->scale = range / 255.0f;
  rqp->zero_point = rmin;

  float inv_scale = 1.0f / rqp->scale;
  for (int i = 0; i < dim; i++) {
    float v = (residual[i] - rqp->zero_point) * inv_scale;
    int q = (int)roundf(v);
    if (q < 0)
      q = 0;
    if (q > 255)
      q = 255;
    codes[i] = (uint8_t)q;
  }
}

void qc_dequantize_residual(const uint8_t *codes, int dim,
                                   const QCResidualQuantParams *rqp,
                                   float *out) {
  for (int i = 0; i < dim; i++) {
    out[i] = (float)codes[i] * rqp->scale + rqp->zero_point;
  }
}


int qc_encode_dataset(const float *data, int num_vectors, int dim,
                             int use_residual, QCDataset *ds) {
  ds->num_vectors = num_vectors;
  ds->dim = dim;
  ds->use_residual = use_residual;

  qc_compute_global_params(data, num_vectors * dim, &ds->global_qparams);

  ds->compressed_vectors =
      (QCCompressedVector *)calloc(num_vectors, sizeof(QCCompressedVector));
  if (!ds->compressed_vectors)
    return -1;

  float *deq_buf = (float *)malloc(dim * sizeof(float));
  float *res_buf = (float *)malloc(dim * sizeof(float));
  if (!deq_buf || !res_buf) {
    free(deq_buf);
    free(res_buf);
    return -1;
  }

  if (use_residual) {
    ds->residual_qparams = (QCResidualQuantParams *)calloc(
        num_vectors, sizeof(QCResidualQuantParams));
    ds->residual_codes = (uint8_t *)calloc(num_vectors * dim, 1);
    ds->original_data = NULL;
    if (!ds->residual_qparams || !ds->residual_codes) {
      free(deq_buf);
      free(res_buf);
      return -1;
    }
  } else {
    ds->residual_qparams = NULL;
    ds->residual_codes = NULL;
    ds->original_data = (float *)malloc(num_vectors * dim * sizeof(float));
    if (!ds->original_data) {
      free(deq_buf);
      free(res_buf);
      return -1;
    }
    memcpy(ds->original_data, data, num_vectors * dim * sizeof(float));
  }

  for (int v = 0; v < num_vectors; v++) {
    const float *vec = data + v * dim;


    ds->compressed_vectors[v].compressed_size = (uint32_t)dim;
    ds->compressed_vectors[v].compressed_data = (uint8_t *)malloc(dim);
    if (!ds->compressed_vectors[v].compressed_data) {
      free(deq_buf);
      free(res_buf);
      return -1;
    }

    qc_quantize_vector(vec, dim, &ds->global_qparams,
                       ds->compressed_vectors[v].compressed_data);


    if (use_residual) {
      qc_dequantize_vector(ds->compressed_vectors[v].compressed_data, dim,
                           &ds->global_qparams, deq_buf);
      qc_compute_residual(vec, deq_buf, dim, res_buf);
      qc_quantize_residual(res_buf, dim, &ds->residual_qparams[v],
                           ds->residual_codes + v * dim);
    }
  }

  free(deq_buf);
  free(res_buf);
  return 0;
}

int qc_decode_vector(const QCDataset *ds, int vec_idx, float *out) {
  if (vec_idx < 0 || vec_idx >= ds->num_vectors)
    return -1;

  int dim = ds->dim;
  const QCCompressedVector *cv = &ds->compressed_vectors[vec_idx];

  qc_dequantize_vector(cv->compressed_data, dim, &ds->global_qparams, out);

  if (ds->use_residual && ds->residual_codes && ds->residual_qparams) {
    float *res = (float *)malloc(dim * sizeof(float));
    if (!res)
      return -1;
    qc_dequantize_residual(ds->residual_codes + vec_idx * dim, dim,
                           &ds->residual_qparams[vec_idx], res);
    for (int i = 0; i < dim; i++) {
      out[i] += res[i];
    }
    free(res);
  } else if (!ds->use_residual && ds->original_data) {
    memcpy(out, ds->original_data + vec_idx * dim, dim * sizeof(float));
  }

  return 0;
}

void qc_free_dataset(QCDataset *ds) {
  if (!ds)
    return;
  if (ds->compressed_vectors) {
    for (int i = 0; i < ds->num_vectors; i++) {
      free(ds->compressed_vectors[i].compressed_data);
    }
    free(ds->compressed_vectors);
  }
  free(ds->residual_qparams);
  free(ds->residual_codes);
  free(ds->original_data);
  memset(ds, 0, sizeof(QCDataset));
}

#endif
