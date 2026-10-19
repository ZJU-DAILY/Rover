#include "dsp_demo/retrieval_pure_pq_builder.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "dsp_demo/retrieval_pq_artifact.h"
#include "dsp_demo/retrieval_pq_core.h"
#include "dsp_demo/retrieval_pq_residual.h"

#define PURE_PQ_LAYOUT_FINGERPRINT_DOMAIN "pure-pq-gorder-layout-v1"

static int pure_pq_size_mul(size_t left, size_t right, size_t *result) {
    if (!result || (left != 0u && right > SIZE_MAX / left)) {
        return -1;
    }
    *result = left * right;
    return 0;
}

static uint32_t pure_pq_read_le32(const uint8_t bytes[4]) {
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
           ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

static int pure_pq_read_dim(FILE *file, int expected_dim) {
    uint8_t bytes[4];
    uint32_t value;
    if (fread(bytes, 1, sizeof(bytes), file) != sizeof(bytes)) {
        return -1;
    }
    value = pure_pq_read_le32(bytes);
    if (value == 0u || value > (uint32_t)INT_MAX ||
        (expected_dim > 0 && value != (uint32_t)expected_dim)) {
        return -1;
    }
    return (int)value;
}

static int pure_pq_fvecs_metadata(const char *path,
                                  int expected_num_vectors,
                                  int expected_dim,
                                  int *out_num_vectors,
                                  int *out_dim,
                                  uint64_t *out_file_bytes) {
    struct stat status;
    FILE *file = NULL;
    int dim;
    uint64_t record_bytes;
    uint64_t num_vectors;

    if (!path || !out_num_vectors || !out_dim || !out_file_bytes ||
        stat(path, &status) != 0 || status.st_size < 4) {
        return -1;
    }
    file = fopen(path, "rb");
    if (!file) {
        return -1;
    }
    dim = pure_pq_read_dim(file, expected_dim);
    fclose(file);
    if (dim <= 0) {
        return -1;
    }
    record_bytes = 4u + (uint64_t)(uint32_t)dim * sizeof(float);
    if ((uint64_t)status.st_size % record_bytes != 0u) {
        return -1;
    }
    num_vectors = (uint64_t)status.st_size / record_bytes;
    if (num_vectors == 0u || num_vectors > (uint64_t)INT_MAX ||
        (expected_num_vectors > 0 &&
         num_vectors != (uint64_t)expected_num_vectors)) {
        return -1;
    }
    *out_num_vectors = (int)num_vectors;
    *out_dim = dim;
    *out_file_bytes = (uint64_t)status.st_size;
    return 0;
}

static int pure_pq_sample_id(int sample_index, int train_count,
                             int num_vectors) {
    if (train_count == 1) {
        return 0;
    }
    return (int)(((int64_t)sample_index * (int64_t)(num_vectors - 1)) /
                 (int64_t)(train_count - 1));
}

static int pure_pq_load_order_map(const char *path,
                                  int num_vectors,
                                  uint32_t **out_new_to_old) {
    FILE *file = NULL;
    uint32_t *new_to_old = NULL;
    uint8_t *seen = NULL;
    size_t seen_bytes;
    int new_id;
    int result = -1;

    if (!out_new_to_old || num_vectors <= 0) {
        return -1;
    }
    *out_new_to_old = NULL;
    if (!path || path[0] == '\0') {
        return 0;
    }
    seen_bytes = ((size_t)num_vectors + 7u) / 8u;
    new_to_old = (uint32_t *)malloc(
        (size_t)num_vectors * sizeof(*new_to_old));
    seen = (uint8_t *)calloc(seen_bytes, 1u);
    file = fopen(path, "r");
    if (!new_to_old || !seen || !file) {
        goto done;
    }
    for (new_id = 0; new_id < num_vectors; new_id++) {
        unsigned long long parsed;
        uint32_t old_id;
        uint8_t bit;
        if (fscanf(file, "%llu", &parsed) != 1 ||
            parsed >= (unsigned long long)num_vectors) {
            goto done;
        }
        old_id = (uint32_t)parsed;
        bit = (uint8_t)(1u << (old_id & 7u));
        if ((seen[old_id >> 3] & bit) != 0u) {
            goto done;
        }
        seen[old_id >> 3] |= bit;
        new_to_old[new_id] = old_id;
    }
    {
        int trailing;
        do {
            trailing = fgetc(file);
        } while (trailing != EOF && isspace((unsigned char)trailing));
        if (trailing != EOF || ferror(file)) {
            goto done;
        }
    }
    *out_new_to_old = new_to_old;
    new_to_old = NULL;
    result = 0;

done:
    if (file) {
        fclose(file);
    }
    free(seen);
    free(new_to_old);
    return result;
}

static int pure_pq_compute_layout_fingerprint(
        const char *layout_id,
        const uint32_t *new_to_old,
        int num_vectors,
        uint8_t fingerprint[32]) {
    const char domain[] = PURE_PQ_LAYOUT_FINGERPRINT_DOMAIN;
    uint8_t order_map_fingerprint[32];
    uint8_t *input = NULL;
    size_t layout_bytes;
    size_t input_bytes;
    int result;

    if (!layout_id || !fingerprint || num_vectors <= 0) {
        return -1;
    }
    if (!new_to_old) {
        return retrieval_pq_artifact_hash_bytes(
            layout_id, strlen(layout_id), fingerprint);
    }
    if (retrieval_pq_artifact_hash_bytes(
            new_to_old, (size_t)num_vectors * sizeof(*new_to_old),
            order_map_fingerprint) != 0) {
        return -1;
    }
    layout_bytes = strlen(layout_id);
    input_bytes = sizeof(domain) + layout_bytes + 1u +
                  sizeof(order_map_fingerprint);
    input = (uint8_t *)malloc(input_bytes);
    if (!input) {
        return -1;
    }
    memcpy(input, domain, sizeof(domain));
    memcpy(input + sizeof(domain), layout_id, layout_bytes);
    input[sizeof(domain) + layout_bytes] = 0u;
    memcpy(input + sizeof(domain) + layout_bytes + 1u,
           order_map_fingerprint, sizeof(order_map_fingerprint));
    result = retrieval_pq_artifact_hash_bytes(
        input, input_bytes, fingerprint);
    free(input);
    return result;
}

static int pure_pq_load_training_sample(const char *path,
                                        int num_vectors,
                                        int dim,
                                        int train_count,
                                        int subspace,
                                        int subdim,
                                        float *sample) {
    FILE *file = fopen(path, "rb");
    int sample_index = 0;
    int vector_index;
    int offset = subspace < 0 ? 0 : subspace * subdim;
    int values = subspace < 0 ? dim : subdim;

    if (!file || !sample) {
        if (file) {
            fclose(file);
        }
        return -1;
    }
    for (vector_index = 0; vector_index < num_vectors; vector_index++) {
        int target = sample_index < train_count
                         ? pure_pq_sample_id(sample_index, train_count,
                                             num_vectors)
                         : -1;
        if (pure_pq_read_dim(file, dim) != dim) {
            fclose(file);
            return -1;
        }
        if (vector_index == target) {
            if ((offset > 0 &&
                 fseek(file, (long)((size_t)offset * sizeof(float)),
                       SEEK_CUR) != 0) ||
                fread(sample + (size_t)sample_index * (size_t)values,
                      sizeof(float), (size_t)values, file) != (size_t)values ||
                (offset + values < dim &&
                 fseek(file,
                       (long)((size_t)(dim - offset - values) * sizeof(float)),
                       SEEK_CUR) != 0)) {
                fclose(file);
                return -1;
            }
            sample_index++;
        } else if (fseek(file, (long)((size_t)dim * sizeof(float)),
                         SEEK_CUR) != 0) {
            fclose(file);
            return -1;
        }
    }
    if (fclose(file) != 0 || sample_index != train_count) {
        return -1;
    }
    return 0;
}

static int pure_pq_fill_uuid(uint8_t uuid[16],
                             const uint8_t dataset_fingerprint[32]) {
    FILE *random = fopen("/dev/urandom", "rb");
    if (random) {
        size_t bytes = fread(uuid, 1, 16, random);
        fclose(random);
        if (bytes == 16u) {
            uuid[6] = (uint8_t)((uuid[6] & 0x0fu) | 0x40u);
            uuid[8] = (uint8_t)((uuid[8] & 0x3fu) | 0x80u);
            return 0;
        }
    }
    {
        uint8_t input[48];
        uint8_t digest[32];
        uint64_t timestamp = (uint64_t)time(NULL);
        uint64_t process = (uint64_t)getpid();
        memcpy(input, dataset_fingerprint, 32);
        memcpy(input + 32, &timestamp, sizeof(timestamp));
        memcpy(input + 40, &process, sizeof(process));
        if (retrieval_pq_artifact_hash_bytes(input, sizeof(input), digest) != 0) {
            return -1;
        }
        memcpy(uuid, digest, 16);
        uuid[6] = (uint8_t)((uuid[6] & 0x0fu) | 0x40u);
        uuid[8] = (uint8_t)((uuid[8] & 0x3fu) | 0x80u);
    }
    return 0;
}

static int pure_pq_stream_encode(const PurePqBuildRequest *request,
                                 int num_vectors,
                                 int dim,
                                 int chunk_vectors,
                                 const uint32_t *new_to_old,
                                 const RetrievalPqConfig *config,
                                 const float *codebook,
                                 RetrievalPqArtifactWriter *writer,
                                 RetrievalPqResidualWriter *residual_writer) {
    FILE *file = NULL;
    float *vectors = NULL;
    uint8_t *codes = NULL;
    int code_bytes = retrieval_pq_code_bytes(config->m, config->ks);
    int remaining = num_vectors;
    uint64_t record_bytes = 4u + (uint64_t)(uint32_t)dim * sizeof(float);
    int result = -1;

    if (code_bytes <= 0) {
        return -1;
    }
    vectors = (float *)malloc((size_t)chunk_vectors * (size_t)dim *
                              sizeof(float));
    codes = (uint8_t *)malloc((size_t)chunk_vectors * (size_t)code_bytes);
    file = fopen(request->base_fvecs_path, "rb");
    if (!vectors || !codes || !file) {
        goto done;
    }
    while (remaining > 0) {
        int count = remaining < chunk_vectors ? remaining : chunk_vectors;
        int row;
        for (row = 0; row < count; row++) {
            int new_id = num_vectors - remaining + row;
            uint32_t old_id = new_to_old
                                  ? new_to_old[new_id]
                                  : (uint32_t)new_id;
            uint64_t byte_offset = (uint64_t)old_id * record_bytes;
            if (new_to_old &&
                (byte_offset > (uint64_t)LONG_MAX ||
                 fseek(file, (long)byte_offset, SEEK_SET) != 0)) {
                goto done;
            }
            if (pure_pq_read_dim(file, dim) != dim ||
                fread(vectors + (size_t)row * (size_t)dim,
                      sizeof(float), (size_t)dim, file) != (size_t)dim) {
                goto done;
            }
        }
        if (retrieval_pq_encode_batch(vectors, count, codebook,
                                      config, codes) != 0 ||
            retrieval_pq_artifact_writer_append_codes(
                writer, codes, (size_t)count * (size_t)code_bytes) != 0 ||
            (residual_writer && retrieval_pq_residual_writer_append(
                residual_writer, vectors, codes, count,
                codebook, config) != 0)) {
            goto done;
        }
        remaining -= count;
    }
    if ((!new_to_old && fgetc(file) != EOF) || ferror(file)) {
        goto done;
    }
    result = 0;

done:
    if (file) {
        fclose(file);
    }
    free(vectors);
    free(codes);
    return result;
}

int pure_pq_build_mobile_artifacts(const PurePqBuildRequest *request,
                                   PurePqBuildStats *out_stats) {
    RetrievalPqConfig config;
    RetrievalPqArtifactMetadata metadata;
    RetrievalPqArtifactWriter *writer = NULL;
    RetrievalPqResidualWriter *residual_writer = NULL;
    uint32_t *new_to_old = NULL;
    float *codebook = NULL;
    float *sample = NULL;
    uint8_t layout_fingerprint[32];
    size_t codebook_values;
    size_t codebook_bytes;
    size_t fast_sample_bytes;
    size_t bounded_sample_bytes;
    size_t training_scratch_bytes;
    size_t fast_peak;
    size_t bounded_peak;
    size_t encode_bytes_per_vector;
    size_t encode_budget;
    size_t publication_peak;
    size_t order_map_bytes;
    size_t order_validation_peak;
    uint64_t file_bytes;
    int num_vectors;
    int dim;
    int train_count;
    int bounded;
    int code_bytes;
    int encode_chunk;
    int subspace;
    int has_order_map;
    int result = -1;

    if (out_stats) {
        memset(out_stats, 0, sizeof(*out_stats));
    }
    if (!request || !request->base_fvecs_path || !request->artifact_prefix ||
        request->m <= 0 ||
        (request->ks != 16 && request->ks != 256) ||
        request->train_max <= 0 ||
        request->kmeans_iters <= 0 || request->encode_chunk_vectors <= 0 ||
        (request->residual_bits != 0 && request->residual_bits != 4 &&
         request->residual_bits != 8) ||
        request->memory_budget_bytes == 0u ||
        pure_pq_fvecs_metadata(request->base_fvecs_path,
                               request->expected_num_vectors,
                               request->expected_dim,
                               &num_vectors, &dim, &file_bytes) != 0 ||
        dim % request->m != 0) {
        return -1;
    }
    has_order_map = request->order_map_path &&
                    request->order_map_path[0] != '\0';
    if (has_order_map && (!request->layout_id ||
                          request->layout_id[0] == '\0')) {
        return -1;
    }
    train_count = num_vectors < request->train_max
                      ? num_vectors : request->train_max;
    memset(&config, 0, sizeof(config));
    config.m = request->m;
    config.ks = request->ks;
    config.subdim = dim / request->m;
    config.dim = dim;
    config.train_max = train_count;
    config.kmeans_iters = request->kmeans_iters;
    code_bytes = retrieval_pq_code_bytes(config.m, config.ks);
    if (retrieval_pq_validate_config(&config) != 0 || code_bytes <= 0 ||
        pure_pq_size_mul((size_t)config.m, (size_t)config.ks,
                         &codebook_values) != 0 ||
        pure_pq_size_mul(codebook_values, (size_t)config.subdim,
                         &codebook_values) != 0 ||
        pure_pq_size_mul(codebook_values, sizeof(float),
                         &codebook_bytes) != 0 ||
        pure_pq_size_mul((size_t)train_count, (size_t)dim,
                         &fast_sample_bytes) != 0 ||
        pure_pq_size_mul(fast_sample_bytes, sizeof(float),
                         &fast_sample_bytes) != 0 ||
        pure_pq_size_mul((size_t)train_count, (size_t)config.subdim,
                         &bounded_sample_bytes) != 0 ||
        pure_pq_size_mul(bounded_sample_bytes, sizeof(float),
                         &bounded_sample_bytes) != 0) {
        return -1;
    }
    training_scratch_bytes =
        (size_t)config.ks * (size_t)config.subdim * sizeof(float) +
        (size_t)config.ks * sizeof(int);
    order_map_bytes = has_order_map
                          ? (size_t)num_vectors * sizeof(*new_to_old)
                          : 0u;
    order_validation_peak = order_map_bytes +
        (has_order_map ? ((size_t)num_vectors + 7u) / 8u : 0u);
    if (codebook_bytes > SIZE_MAX - training_scratch_bytes ||
        codebook_bytes + training_scratch_bytes >
            SIZE_MAX - fast_sample_bytes ||
        codebook_bytes + training_scratch_bytes >
            SIZE_MAX - bounded_sample_bytes ||
        codebook_bytes + training_scratch_bytes + fast_sample_bytes >
            SIZE_MAX - order_map_bytes ||
        codebook_bytes + training_scratch_bytes + bounded_sample_bytes >
            SIZE_MAX - order_map_bytes) {
        return -1;
    }
    fast_peak = order_map_bytes + codebook_bytes +
                training_scratch_bytes + fast_sample_bytes;
    bounded_peak = codebook_bytes + training_scratch_bytes +
                   bounded_sample_bytes + order_map_bytes;
    if (codebook_bytes > (SIZE_MAX - order_map_bytes) / 2u) {
        return -1;
    }
    publication_peak = order_map_bytes + codebook_bytes * 2u;
    bounded = fast_peak > request->memory_budget_bytes;
    if ((bounded && bounded_peak > request->memory_budget_bytes) ||
        publication_peak > request->memory_budget_bytes ||
        order_validation_peak > request->memory_budget_bytes) {
        return -1;
    }
    encode_bytes_per_vector = (size_t)dim * sizeof(float) +
                              (size_t)code_bytes;
    if (request->memory_budget_bytes <= codebook_bytes + order_map_bytes ||
        encode_bytes_per_vector == 0u) {
        return -1;
    }
    encode_budget = request->memory_budget_bytes - codebook_bytes -
                    order_map_bytes;
    encode_chunk = (int)(encode_budget / encode_bytes_per_vector);
    if (encode_chunk <= 0) {
        return -1;
    }
    if (encode_chunk > request->encode_chunk_vectors) {
        encode_chunk = request->encode_chunk_vectors;
    }

    if (pure_pq_load_order_map(request->order_map_path, num_vectors,
                               &new_to_old) != 0 ||
        pure_pq_compute_layout_fingerprint(
            request->layout_id ? request->layout_id : "",
            new_to_old, num_vectors, layout_fingerprint) != 0 ||
        (request->layout_fingerprint &&
         memcmp(request->layout_fingerprint, layout_fingerprint,
                sizeof(layout_fingerprint)) != 0)) {
        goto done;
    }

    codebook = (float *)malloc(codebook_bytes);
    if (!codebook) {
        goto done;
    }
    if (!bounded) {
        sample = (float *)malloc(fast_sample_bytes);
        if (!sample ||
            pure_pq_load_training_sample(request->base_fvecs_path,
                                         num_vectors, dim, train_count,
                                         -1, config.subdim, sample) != 0 ||
            retrieval_pq_train(sample, train_count, &config,
                               codebook) != 0) {
            goto done;
        }
    } else {
        sample = (float *)malloc(bounded_sample_bytes);
        if (!sample) {
            goto done;
        }
        for (subspace = 0; subspace < config.m; subspace++) {
            float *centroids = codebook +
                (size_t)subspace * (size_t)config.ks * (size_t)config.subdim;
            if (pure_pq_load_training_sample(request->base_fvecs_path,
                                             num_vectors, dim, train_count,
                                             subspace, config.subdim,
                                             sample) != 0 ||
                retrieval_pq_train_subspace(sample, train_count,
                                             config.subdim, config.ks,
                                             config.subdim,
                                             config.kmeans_iters,
                                             centroids) != 0) {
                goto done;
            }
        }
    }
    free(sample);
    sample = NULL;

    memset(&metadata, 0, sizeof(metadata));
    metadata.num_vectors = (uint64_t)num_vectors;
    metadata.dim = (uint32_t)dim;
    metadata.m = (uint32_t)config.m;
    metadata.ks = (uint32_t)config.ks;
    metadata.subdim = (uint32_t)config.subdim;
    metadata.code_bytes = (uint32_t)code_bytes;
    metadata.trainer_kind = RETRIEVAL_PQ_TRAINER_MOBILE_C;
    metadata.sample_strategy = RETRIEVAL_PQ_SAMPLE_UNIFORM_STRIDE;
    metadata.train_count = (uint64_t)train_count;
    metadata.kmeans_iters = (uint32_t)config.kmeans_iters;
    if (retrieval_pq_artifact_hash_file_prefix(
            request->base_fvecs_path, file_bytes,
            metadata.dataset_fingerprint) != 0 ||
        pure_pq_fill_uuid(metadata.index_uuid,
                          metadata.dataset_fingerprint) != 0) {
        goto done;
    }
    memcpy(metadata.layout_fingerprint, layout_fingerprint,
           sizeof(metadata.layout_fingerprint));
    if (retrieval_pq_artifact_writer_begin(request->artifact_prefix,
                                           &metadata, codebook,
                                           &writer) != 0 ||
        (request->residual_bits != 0 &&
         retrieval_pq_residual_writer_begin(
             request->artifact_prefix, &metadata,
             request->residual_bits, &residual_writer) != 0) ||
        pure_pq_stream_encode(request, num_vectors, dim, encode_chunk,
                              new_to_old, &config, codebook, writer,
                              residual_writer) != 0 ||
        retrieval_pq_artifact_writer_commit(writer) != 0) {
        goto done;
    }
    writer = NULL;
    if (residual_writer &&
        retrieval_pq_residual_writer_commit(residual_writer) != 0) {
        residual_writer = NULL;
        goto done;
    }
    residual_writer = NULL;
    if (out_stats) {
        out_stats->num_vectors = num_vectors;
        out_stats->dim = dim;
        out_stats->train_count = train_count;
        out_stats->used_bounded_training = bounded;
        out_stats->encode_chunk_vectors = encode_chunk;
        out_stats->estimated_peak_bytes = bounded ? bounded_peak : fast_peak;
        if (publication_peak > out_stats->estimated_peak_bytes) {
            out_stats->estimated_peak_bytes = publication_peak;
        }
        {
            size_t encode_peak = order_map_bytes + codebook_bytes +
                (size_t)encode_chunk * encode_bytes_per_vector;
            if (encode_peak > out_stats->estimated_peak_bytes) {
                out_stats->estimated_peak_bytes = encode_peak;
            }
        }
    }
    result = 0;

done:
    if (writer) {
        retrieval_pq_artifact_writer_abort(writer);
    }
    if (residual_writer) {
        retrieval_pq_residual_writer_abort(residual_writer);
    }
    free(sample);
    free(codebook);
    free(new_to_old);
    return result;
}
