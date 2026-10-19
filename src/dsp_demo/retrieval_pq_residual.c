#include "rover_config.h"
#include "dsp_demo/retrieval_pq_residual.h"

#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define PQ_RESIDUAL_ENDIAN_MARKER 0x01020304u
#define PQ_RESIDUAL_VERSION_MAJOR 1u
#define PQ_RESIDUAL_VERSION_MINOR 0u
#define PQ_RESIDUAL_KIND_BLOCKS 1u
#define PQ_RESIDUAL_KIND_QPARAMS 2u

static const uint8_t pq_residual_magic[8] = {
    'P', 'Q', 'R', 'E', 'S', '0', '1', '\0'
};

typedef struct RetrievalPqResidualCacheSlot {
    int64_t block_index;
    uint8_t data[RETRIEVAL_PQ_RESIDUAL_BLOCK_BYTES];
} RetrievalPqResidualCacheSlot;

struct RetrievalPqResidualWriter {
    RetrievalPqResidualMetadata metadata;
    char *prefix;
    char *blocks_path;
    char *blocks_temporary;
    char *qparams_path;
    char *qparams_temporary;
    char *manifest_path;
    char *manifest_temporary;
    int blocks_fd;
    int qparams_fd;
    uint8_t block[RETRIEVAL_PQ_RESIDUAL_BLOCK_BYTES];
    uint8_t *quantized;
    uint64_t vectors_written;
    uint32_t block_vectors;
    uint32_t blocks_written;
};

struct RetrievalPqResidualIndex {
    RetrievalPqResidualMetadata metadata;
    int fd;
    int cache_slots;
    uint8_t *resident_blocks;
    RetrievalPqResidualQParams *qparams;
    const RetrievalPqArtifactMetadata *pq_metadata;
    const float *codebook;
    const uint8_t *const *code_chunks;
    const int *chunk_vector_counts;
    int chunk_count;
    int chunk_vectors;
};

struct RetrievalPqResidualQuery {
    RetrievalPqResidualIndex *index;
    int cache_slots;
    const float *query_vector;
    RetrievalPqResidualCacheSlot *slots;
    RetrievalPqResidualStats stats;
};

static void pq_residual_put_u16(uint8_t *output, uint16_t value) {
    output[0] = (uint8_t)value;
    output[1] = (uint8_t)(value >> 8);
}

static void pq_residual_put_u32(uint8_t *output, uint32_t value) {
    output[0] = (uint8_t)value;
    output[1] = (uint8_t)(value >> 8);
    output[2] = (uint8_t)(value >> 16);
    output[3] = (uint8_t)(value >> 24);
}

static void pq_residual_put_u64(uint8_t *output, uint64_t value) {
    unsigned int index;
    for (index = 0; index < 8; index++) {
        output[index] = (uint8_t)(value >> (index * 8));
    }
}

static uint16_t pq_residual_get_u16(const uint8_t *input) {
    return (uint16_t)((uint16_t)input[0] | ((uint16_t)input[1] << 8));
}

static uint32_t pq_residual_get_u32(const uint8_t *input) {
    return (uint32_t)input[0] | ((uint32_t)input[1] << 8) |
           ((uint32_t)input[2] << 16) | ((uint32_t)input[3] << 24);
}

static uint64_t pq_residual_get_u64(const uint8_t *input) {
    uint64_t value = 0u;
    int index;
    for (index = 0; index < 8; index++) {
        value |= (uint64_t)input[index] << (index * 8);
    }
    return value;
}

static int pq_residual_write_full(int fd, const void *data, size_t bytes,
                                  uint64_t offset) {
    size_t written = 0u;
    while (written < bytes) {
        ssize_t result = pwrite(fd, (const uint8_t *)data + written,
                                bytes - written,
                                (off_t)(offset + (uint64_t)written));
        if (result < 0 && errno == EINTR) {
            continue;
        }
        if (result <= 0) {
            return -1;
        }
        written += (size_t)result;
    }
    return 0;
}

static int pq_residual_read_full(int fd, void *data, size_t bytes,
                                 uint64_t offset) {
    size_t read_bytes = 0u;
    while (read_bytes < bytes) {
        ssize_t result = pread(fd, (uint8_t *)data + read_bytes,
                               bytes - read_bytes,
                               (off_t)(offset + (uint64_t)read_bytes));
        if (result < 0 && errno == EINTR) {
            continue;
        }
        if (result <= 0) {
            return -1;
        }
        read_bytes += (size_t)result;
    }
    return 0;
}

static char *pq_residual_path(const char *prefix, int residual_bits,
                              const char *suffix) {
    size_t bytes;
    char *path;
    int result;
    if (!prefix || !suffix) {
        return NULL;
    }
    bytes = strlen(prefix) + strlen(suffix) + 16u;
    path = (char *)malloc(bytes);
    if (!path) {
        return NULL;
    }
    result = snprintf(path, bytes, "%s.pqres%d%s",
                      prefix, residual_bits, suffix);
    if (result < 0 || (size_t)result >= bytes) {
        free(path);
        return NULL;
    }
    return path;
}

static char *pq_residual_temporary_path(const char *path) {
    size_t bytes;
    char *temporary;
    if (!path) {
        return NULL;
    }
    bytes = strlen(path) + 5u;
    temporary = (char *)malloc(bytes);
    if (!temporary) {
        return NULL;
    }
    snprintf(temporary, bytes, "%s.tmp", path);
    return temporary;
}

static int pq_residual_valid_bits(int residual_bits) {
    return residual_bits == 4 || residual_bits == 8;
}

static uint32_t pq_residual_code_bytes(uint32_t dim, int residual_bits) {
    return residual_bits == 4 ? dim / 2u + dim % 2u : dim;
}

static int pq_residual_validate_metadata(
        const RetrievalPqResidualMetadata *metadata) {
    uint32_t expected_code_bytes;
    uint64_t expected_blocks;
    if (!metadata || !pq_residual_valid_bits((int)metadata->residual_bits) ||
        metadata->num_vectors == 0u || metadata->dim == 0u ||
        metadata->block_bytes != RETRIEVAL_PQ_RESIDUAL_BLOCK_BYTES ||
        metadata->vectors_per_block == 0u) {
        return -1;
    }
    expected_code_bytes = pq_residual_code_bytes(
        metadata->dim, (int)metadata->residual_bits);
    if (metadata->residual_code_bytes != expected_code_bytes ||
        expected_code_bytes > metadata->block_bytes ||
        metadata->vectors_per_block !=
            metadata->block_bytes / expected_code_bytes) {
        return -1;
    }
    expected_blocks = (metadata->num_vectors +
                       metadata->vectors_per_block - 1u) /
                      metadata->vectors_per_block;
    if (expected_blocks == 0u || expected_blocks > UINT32_MAX ||
        metadata->total_blocks != (uint32_t)expected_blocks) {
        return -1;
    }
    return 0;
}

static void pq_residual_encode_header(
        uint8_t header[RETRIEVAL_PQ_RESIDUAL_HEADER_BYTES],
        uint32_t kind,
        uint32_t storage_header_bytes,
        const RetrievalPqResidualMetadata *metadata,
        uint64_t payload_bytes,
        const uint8_t payload_hash[RETRIEVAL_PQ_ARTIFACT_HASH_BYTES]) {
    memset(header, 0, RETRIEVAL_PQ_RESIDUAL_HEADER_BYTES);
    memcpy(header, pq_residual_magic, sizeof(pq_residual_magic));
    pq_residual_put_u16(header + 8, PQ_RESIDUAL_VERSION_MAJOR);
    pq_residual_put_u16(header + 10, PQ_RESIDUAL_VERSION_MINOR);
    pq_residual_put_u32(header + 12, storage_header_bytes);
    pq_residual_put_u32(header + 16, kind);
    pq_residual_put_u32(header + 20, PQ_RESIDUAL_ENDIAN_MARKER);
    pq_residual_put_u32(header + 24, metadata->residual_bits);
    pq_residual_put_u32(header + 28, metadata->residual_code_bytes);
    memcpy(header + 32, metadata->index_uuid,
           RETRIEVAL_PQ_ARTIFACT_UUID_BYTES);
    pq_residual_put_u64(header + 48, metadata->num_vectors);
    pq_residual_put_u32(header + 56, metadata->dim);
    pq_residual_put_u32(header + 60, metadata->block_bytes);
    pq_residual_put_u32(header + 64, metadata->vectors_per_block);
    pq_residual_put_u32(header + 68, metadata->total_blocks);
    pq_residual_put_u64(header + 72, payload_bytes);
    memcpy(header + 80, payload_hash, RETRIEVAL_PQ_ARTIFACT_HASH_BYTES);
    memcpy(header + 112, metadata->dataset_fingerprint,
           RETRIEVAL_PQ_ARTIFACT_HASH_BYTES);
    memcpy(header + 144, metadata->layout_fingerprint,
           RETRIEVAL_PQ_ARTIFACT_HASH_BYTES);
    pq_residual_put_u32(header + 176, RETRIEVAL_PQ_RESIDUAL_QPARAM_BYTES);
}

static int pq_residual_decode_header(
        const uint8_t header[RETRIEVAL_PQ_RESIDUAL_HEADER_BYTES],
        uint32_t expected_kind,
        uint32_t expected_storage_header_bytes,
        RetrievalPqResidualMetadata *metadata,
        uint64_t *payload_bytes,
        uint8_t payload_hash[RETRIEVAL_PQ_ARTIFACT_HASH_BYTES]) {
    if (!header || !metadata || !payload_bytes || !payload_hash ||
        memcmp(header, pq_residual_magic, sizeof(pq_residual_magic)) != 0 ||
        pq_residual_get_u16(header + 8) != PQ_RESIDUAL_VERSION_MAJOR ||
        pq_residual_get_u16(header + 10) > PQ_RESIDUAL_VERSION_MINOR ||
        pq_residual_get_u32(header + 12) != expected_storage_header_bytes ||
        pq_residual_get_u32(header + 16) != expected_kind ||
        pq_residual_get_u32(header + 20) != PQ_RESIDUAL_ENDIAN_MARKER ||
        pq_residual_get_u32(header + 176) !=
            RETRIEVAL_PQ_RESIDUAL_QPARAM_BYTES) {
        return -1;
    }
    memset(metadata, 0, sizeof(*metadata));
    metadata->residual_bits = pq_residual_get_u32(header + 24);
    metadata->residual_code_bytes = pq_residual_get_u32(header + 28);
    memcpy(metadata->index_uuid, header + 32,
           RETRIEVAL_PQ_ARTIFACT_UUID_BYTES);
    metadata->num_vectors = pq_residual_get_u64(header + 48);
    metadata->dim = pq_residual_get_u32(header + 56);
    metadata->block_bytes = pq_residual_get_u32(header + 60);
    metadata->vectors_per_block = pq_residual_get_u32(header + 64);
    metadata->total_blocks = pq_residual_get_u32(header + 68);
    *payload_bytes = pq_residual_get_u64(header + 72);
    memcpy(payload_hash, header + 80, RETRIEVAL_PQ_ARTIFACT_HASH_BYTES);
    memcpy(metadata->dataset_fingerprint, header + 112,
           RETRIEVAL_PQ_ARTIFACT_HASH_BYTES);
    memcpy(metadata->layout_fingerprint, header + 144,
           RETRIEVAL_PQ_ARTIFACT_HASH_BYTES);
    return pq_residual_validate_metadata(metadata);
}

static int pq_residual_metadata_matches_pq(
        const RetrievalPqResidualMetadata *residual,
        const RetrievalPqArtifactMetadata *pq) {
    return residual && pq && residual->num_vectors == pq->num_vectors &&
           residual->dim == pq->dim &&
           memcmp(residual->index_uuid, pq->index_uuid,
                  RETRIEVAL_PQ_ARTIFACT_UUID_BYTES) == 0 &&
           memcmp(residual->dataset_fingerprint, pq->dataset_fingerprint,
                  RETRIEVAL_PQ_ARTIFACT_HASH_BYTES) == 0 &&
           memcmp(residual->layout_fingerprint, pq->layout_fingerprint,
                  RETRIEVAL_PQ_ARTIFACT_HASH_BYTES) == 0;
}

static int pq_residual_headers_match(
        const RetrievalPqResidualMetadata *left,
        const RetrievalPqResidualMetadata *right) {
    return left && right &&
           memcmp(left, right, sizeof(*left)) == 0;
}

static int pq_residual_centroid_id(const uint8_t *codes, int subspace,
                                   int ks) {
    if (ks == 16) {
        uint8_t packed = codes[subspace >> 1];
        return (subspace & 1) ? (int)(packed >> 4) : (int)(packed & 0x0fu);
    }
    return (int)codes[subspace];
}

static int pq_residual_quantize_vector(
        RetrievalPqResidualWriter *writer,
        const float *vector,
        const uint8_t *pq_codes,
        const float *codebook,
        const RetrievalPqConfig *config,
        RetrievalPqResidualQParams *qparams) {
    float minimum = FLT_MAX;
    float maximum = -FLT_MAX;
    float inverse_scale;
    int levels = writer->metadata.residual_bits == 4u ? 15 : 255;
    int subspace;
    int dimension = 0;

    for (subspace = 0; subspace < config->m; subspace++) {
        int centroid = pq_residual_centroid_id(pq_codes, subspace, config->ks);
        const float *centroid_values = codebook +
            ((size_t)subspace * (size_t)config->ks + (size_t)centroid) *
                (size_t)config->subdim;
        int inner;
        for (inner = 0; inner < config->subdim; inner++, dimension++) {
            float residual = vector[dimension] - centroid_values[inner];
            if (!isfinite(residual)) {
                return -1;
            }
            if (residual < minimum) {
                minimum = residual;
            }
            if (residual > maximum) {
                maximum = residual;
            }
        }
    }
    qparams->zero_point = minimum;
    qparams->scale = (maximum - minimum) / (float)levels;
    if (qparams->scale < 1.0e-12f) {
        qparams->scale = 1.0e-12f;
    }
    inverse_scale = 1.0f / qparams->scale;
    memset(writer->quantized, 0, writer->metadata.residual_code_bytes);
    dimension = 0;
    for (subspace = 0; subspace < config->m; subspace++) {
        int centroid = pq_residual_centroid_id(pq_codes, subspace, config->ks);
        const float *centroid_values = codebook +
            ((size_t)subspace * (size_t)config->ks + (size_t)centroid) *
                (size_t)config->subdim;
        int inner;
        for (inner = 0; inner < config->subdim; inner++, dimension++) {
            float residual = vector[dimension] - centroid_values[inner];
            int quantized = (int)roundf(
                (residual - qparams->zero_point) * inverse_scale);
            if (quantized < 0) {
                quantized = 0;
            }
            if (quantized > levels) {
                quantized = levels;
            }
            if (writer->metadata.residual_bits == 4u) {
                if (dimension & 1) {
                    writer->quantized[dimension >> 1] |=
                        (uint8_t)(quantized << 4);
                } else {
                    writer->quantized[dimension >> 1] =
                        (uint8_t)quantized;
                }
            } else {
                writer->quantized[dimension] = (uint8_t)quantized;
            }
        }
    }
    return 0;
}

static int pq_residual_writer_flush_block(RetrievalPqResidualWriter *writer) {
    uint64_t offset;
    if (!writer || writer->block_vectors == 0u) {
        return 0;
    }
    offset = RETRIEVAL_PQ_RESIDUAL_BLOCK_BYTES +
             (uint64_t)writer->blocks_written *
                 RETRIEVAL_PQ_RESIDUAL_BLOCK_BYTES;
    if (pq_residual_write_full(writer->blocks_fd, writer->block,
                               sizeof(writer->block), offset) != 0) {
        return -1;
    }
    writer->blocks_written++;
    writer->block_vectors = 0u;
    memset(writer->block, 0, sizeof(writer->block));
    return 0;
}

static void pq_residual_writer_release(RetrievalPqResidualWriter *writer,
                                       int unlink_temporary) {
    if (!writer) {
        return;
    }
    if (writer->blocks_fd >= 0) {
        close(writer->blocks_fd);
    }
    if (writer->qparams_fd >= 0) {
        close(writer->qparams_fd);
    }
    if (unlink_temporary) {
        if (writer->blocks_temporary) {
            unlink(writer->blocks_temporary);
        }
        if (writer->qparams_temporary) {
            unlink(writer->qparams_temporary);
        }
        if (writer->manifest_temporary) {
            unlink(writer->manifest_temporary);
        }
    }
    free(writer->prefix);
    free(writer->blocks_path);
    free(writer->blocks_temporary);
    free(writer->qparams_path);
    free(writer->qparams_temporary);
    free(writer->manifest_path);
    free(writer->manifest_temporary);
    free(writer->quantized);
    free(writer);
}

int retrieval_pq_residual_writer_begin(
        const char *artifact_prefix,
        const RetrievalPqArtifactMetadata *pq_metadata,
        int residual_bits,
        RetrievalPqResidualWriter **out_writer) {
    RetrievalPqResidualWriter *writer;
    uint8_t zero_header[RETRIEVAL_PQ_RESIDUAL_BLOCK_BYTES];
    if (!artifact_prefix || !pq_metadata || !out_writer ||
        !pq_residual_valid_bits(residual_bits) ||
        retrieval_pq_artifact_validate_metadata(pq_metadata) != 0) {
        return -1;
    }
    *out_writer = NULL;
    writer = (RetrievalPqResidualWriter *)calloc(1, sizeof(*writer));
    if (!writer) {
        return -1;
    }
    writer->blocks_fd = -1;
    writer->qparams_fd = -1;
    writer->prefix = strdup(artifact_prefix);
    writer->blocks_path = pq_residual_path(artifact_prefix, residual_bits, "");
    writer->qparams_path = pq_residual_path(
        artifact_prefix, residual_bits, ".qparams");
    writer->manifest_path = pq_residual_path(
        artifact_prefix, residual_bits, ".json");
    writer->blocks_temporary = pq_residual_temporary_path(writer->blocks_path);
    writer->qparams_temporary = pq_residual_temporary_path(writer->qparams_path);
    writer->manifest_temporary = pq_residual_temporary_path(writer->manifest_path);
    writer->metadata.num_vectors = pq_metadata->num_vectors;
    writer->metadata.dim = pq_metadata->dim;
    writer->metadata.residual_bits = (uint32_t)residual_bits;
    writer->metadata.residual_code_bytes = pq_residual_code_bytes(
        pq_metadata->dim, residual_bits);
    writer->metadata.block_bytes = RETRIEVAL_PQ_RESIDUAL_BLOCK_BYTES;
    writer->metadata.vectors_per_block =
        RETRIEVAL_PQ_RESIDUAL_BLOCK_BYTES /
        writer->metadata.residual_code_bytes;
    writer->metadata.total_blocks = (uint32_t)(
        (pq_metadata->num_vectors + writer->metadata.vectors_per_block - 1u) /
        writer->metadata.vectors_per_block);
    memcpy(writer->metadata.index_uuid, pq_metadata->index_uuid,
           RETRIEVAL_PQ_ARTIFACT_UUID_BYTES);
    memcpy(writer->metadata.dataset_fingerprint,
           pq_metadata->dataset_fingerprint,
           RETRIEVAL_PQ_ARTIFACT_HASH_BYTES);
    memcpy(writer->metadata.layout_fingerprint,
           pq_metadata->layout_fingerprint,
           RETRIEVAL_PQ_ARTIFACT_HASH_BYTES);
    writer->quantized = (uint8_t *)malloc(
        writer->metadata.residual_code_bytes);
    if (!writer->prefix || !writer->blocks_path || !writer->qparams_path ||
        !writer->manifest_path || !writer->blocks_temporary ||
        !writer->qparams_temporary || !writer->manifest_temporary ||
        !writer->quantized ||
        pq_residual_validate_metadata(&writer->metadata) != 0) {
        pq_residual_writer_release(writer, 1);
        return -1;
    }
    unlink(writer->blocks_temporary);
    unlink(writer->qparams_temporary);
    writer->blocks_fd = open(writer->blocks_temporary,
                             O_CREAT | O_TRUNC | O_RDWR, 0644);
    writer->qparams_fd = open(writer->qparams_temporary,
                              O_CREAT | O_TRUNC | O_RDWR, 0644);
    memset(zero_header, 0, sizeof(zero_header));
    if (writer->blocks_fd < 0 || writer->qparams_fd < 0 ||
        pq_residual_write_full(writer->blocks_fd, zero_header,
                               sizeof(zero_header), 0u) != 0 ||
        pq_residual_write_full(writer->qparams_fd, zero_header,
                               RETRIEVAL_PQ_RESIDUAL_HEADER_BYTES, 0u) != 0) {
        pq_residual_writer_release(writer, 1);
        return -1;
    }
    *out_writer = writer;
    return 0;
}

int retrieval_pq_residual_writer_append(
        RetrievalPqResidualWriter *writer,
        const float *vectors,
        const uint8_t *pq_codes,
        int vector_count,
        const float *codebook,
        const RetrievalPqConfig *config) {
    int code_bytes;
    int vector_index;
    if (!writer || !vectors || !pq_codes || vector_count <= 0 || !codebook ||
        !config || retrieval_pq_validate_config(config) != 0 ||
        config->dim != (int)writer->metadata.dim ||
        writer->vectors_written + (uint64_t)vector_count >
            writer->metadata.num_vectors) {
        return -1;
    }
    code_bytes = retrieval_pq_code_bytes(config->m, config->ks);
    if (code_bytes <= 0) {
        return -1;
    }
    for (vector_index = 0; vector_index < vector_count; vector_index++) {
        RetrievalPqResidualQParams qparams;
        size_t block_offset;
        uint64_t qparams_offset;
        if (pq_residual_quantize_vector(
                writer,
                vectors + (size_t)vector_index * writer->metadata.dim,
                pq_codes + (size_t)vector_index * (size_t)code_bytes,
                codebook, config, &qparams) != 0) {
            return -1;
        }
        qparams_offset = RETRIEVAL_PQ_RESIDUAL_HEADER_BYTES +
            writer->vectors_written * RETRIEVAL_PQ_RESIDUAL_QPARAM_BYTES;
        if (pq_residual_write_full(writer->qparams_fd, &qparams,
                                   sizeof(qparams), qparams_offset) != 0) {
            return -1;
        }
        block_offset = (size_t)writer->block_vectors *
                       writer->metadata.residual_code_bytes;
        memcpy(writer->block + block_offset, writer->quantized,
               writer->metadata.residual_code_bytes);
        writer->block_vectors++;
        writer->vectors_written++;
        if (writer->block_vectors == writer->metadata.vectors_per_block &&
            pq_residual_writer_flush_block(writer) != 0) {
            return -1;
        }
    }
    return 0;
}

static int pq_residual_write_manifest(RetrievalPqResidualWriter *writer,
                                      const uint8_t blocks_hash[32],
                                      const uint8_t qparams_hash[32]) {
    FILE *file;
    unsigned int index;
    file = fopen(writer->manifest_temporary, "w");
    if (!file) {
        return -1;
    }
    if (fprintf(file,
                "{\n  \"format\": \"pure-pq-residual-v1\",\n"
                "  \"index_uuid\": \"") < 0) {
        fclose(file);
        return -1;
    }
    for (index = 0; index < RETRIEVAL_PQ_ARTIFACT_UUID_BYTES; index++) {
        if (fprintf(file, "%02x", writer->metadata.index_uuid[index]) < 0) {
            fclose(file);
            return -1;
        }
    }
    if (fprintf(file,
                "\",\n  \"num_vectors\": %llu,\n  \"dim\": %u,\n"
                "  \"residual_bits\": %u,\n"
                "  \"residual_code_bytes\": %u,\n"
                "  \"block_bytes\": %u,\n"
                "  \"vectors_per_block\": %u,\n"
                "  \"total_blocks\": %u,\n"
                "  \"blocks_payload_sha256\": \"",
                (unsigned long long)writer->metadata.num_vectors,
                writer->metadata.dim, writer->metadata.residual_bits,
                writer->metadata.residual_code_bytes,
                writer->metadata.block_bytes,
                writer->metadata.vectors_per_block,
                writer->metadata.total_blocks) < 0) {
        fclose(file);
        return -1;
    }
    for (index = 0; index < 32; index++) {
        if (fprintf(file, "%02x", blocks_hash[index]) < 0) {
            fclose(file);
            return -1;
        }
    }
    if (fprintf(file, "\",\n  \"qparams_payload_sha256\": \"") < 0) {
        fclose(file);
        return -1;
    }
    for (index = 0; index < 32; index++) {
        if (fprintf(file, "%02x", qparams_hash[index]) < 0) {
            fclose(file);
            return -1;
        }
    }
    if (fprintf(file, "\"\n}\n") < 0 || fflush(file) != 0 ||
        fsync(fileno(file)) != 0 || fclose(file) != 0) {
        return -1;
    }
    return 0;
}

int retrieval_pq_residual_writer_commit(RetrievalPqResidualWriter *writer) {
    uint8_t blocks_header[RETRIEVAL_PQ_RESIDUAL_HEADER_BYTES];
    uint8_t qparams_header[RETRIEVAL_PQ_RESIDUAL_HEADER_BYTES];
    uint8_t blocks_hash[RETRIEVAL_PQ_ARTIFACT_HASH_BYTES];
    uint8_t qparams_hash[RETRIEVAL_PQ_ARTIFACT_HASH_BYTES];
    uint64_t blocks_payload_bytes;
    uint64_t qparams_payload_bytes;
    int result = -1;
    if (!writer || writer->vectors_written != writer->metadata.num_vectors ||
        pq_residual_writer_flush_block(writer) != 0 ||
        writer->blocks_written != writer->metadata.total_blocks) {
        return -1;
    }
    blocks_payload_bytes =
        (uint64_t)writer->metadata.total_blocks *
        RETRIEVAL_PQ_RESIDUAL_BLOCK_BYTES;
    qparams_payload_bytes = writer->metadata.num_vectors *
                            RETRIEVAL_PQ_RESIDUAL_QPARAM_BYTES;
    if (fsync(writer->blocks_fd) != 0 || fsync(writer->qparams_fd) != 0 ||
        retrieval_pq_artifact_hash_file_range(
            writer->blocks_temporary,
            RETRIEVAL_PQ_RESIDUAL_BLOCK_BYTES,
            blocks_payload_bytes, blocks_hash) != 0 ||
        retrieval_pq_artifact_hash_file_range(
            writer->qparams_temporary,
            RETRIEVAL_PQ_RESIDUAL_HEADER_BYTES,
            qparams_payload_bytes, qparams_hash) != 0) {
        return -1;
    }
    pq_residual_encode_header(blocks_header, PQ_RESIDUAL_KIND_BLOCKS,
                              RETRIEVAL_PQ_RESIDUAL_BLOCK_BYTES,
                              &writer->metadata, blocks_payload_bytes,
                              blocks_hash);
    pq_residual_encode_header(qparams_header, PQ_RESIDUAL_KIND_QPARAMS,
                              RETRIEVAL_PQ_RESIDUAL_HEADER_BYTES,
                              &writer->metadata, qparams_payload_bytes,
                              qparams_hash);
    if (pq_residual_write_full(writer->blocks_fd, blocks_header,
                               sizeof(blocks_header), 0u) != 0 ||
        pq_residual_write_full(writer->qparams_fd, qparams_header,
                               sizeof(qparams_header), 0u) != 0 ||
        fsync(writer->blocks_fd) != 0 || fsync(writer->qparams_fd) != 0 ||
        close(writer->blocks_fd) != 0) {
        writer->blocks_fd = -1;
        return -1;
    }
    writer->blocks_fd = -1;
    if (close(writer->qparams_fd) != 0) {
        writer->qparams_fd = -1;
        return -1;
    }
    writer->qparams_fd = -1;
    if (pq_residual_write_manifest(writer, blocks_hash, qparams_hash) != 0 ||
        rename(writer->blocks_temporary, writer->blocks_path) != 0 ||
        rename(writer->qparams_temporary, writer->qparams_path) != 0 ||
        rename(writer->manifest_temporary, writer->manifest_path) != 0) {
        goto done;
    }
    result = 0;

done:
    pq_residual_writer_release(writer, result != 0);
    return result;
}

void retrieval_pq_residual_writer_abort(RetrievalPqResidualWriter *writer) {
    pq_residual_writer_release(writer, 1);
}

static int pq_residual_load_header(int fd,
                                   uint32_t kind,
                                   uint32_t storage_header_bytes,
                                   RetrievalPqResidualMetadata *metadata,
                                   uint64_t *payload_bytes,
                                   uint8_t payload_hash[32]) {
    uint8_t header[RETRIEVAL_PQ_RESIDUAL_HEADER_BYTES];
    if (pq_residual_read_full(fd, header, sizeof(header), 0u) != 0) {
        return -1;
    }
    return pq_residual_decode_header(header, kind, storage_header_bytes,
                                     metadata, payload_bytes, payload_hash);
}

static int pq_residual_file_size_matches(int fd, uint64_t expected) {
    struct stat status;
    return fstat(fd, &status) == 0 && status.st_size >= 0 &&
           (uint64_t)status.st_size == expected;
}

int retrieval_pq_residual_index_open(
        const RetrievalPqResidualOpenRequest *request,
        RetrievalPqResidualIndex **out_index) {
    RetrievalPqResidualIndex *index = NULL;
    RetrievalPqResidualMetadata blocks_metadata;
    RetrievalPqResidualMetadata qparams_metadata;
    uint8_t blocks_hash[32];
    uint8_t qparams_hash[32];
    uint8_t actual_qparams_hash[32];
    uint64_t blocks_payload_bytes;
    uint64_t qparams_payload_bytes;
    uint64_t chunk_vectors = 0u;
    char *blocks_path = NULL;
    char *qparams_path = NULL;
    int qparams_fd = -1;
    int chunk;
    if (!request || !out_index || !request->artifact_prefix ||
        !pq_residual_valid_bits(request->residual_bits) ||
        request->cache_slots <= 0 ||
        (request->cache_slots & (request->cache_slots - 1)) != 0 ||
        !request->pq_metadata || !request->codebook ||
        !request->code_chunks || !request->chunk_vector_counts ||
        request->chunk_count <= 0 || request->chunk_vectors <= 0 ||
        retrieval_pq_artifact_validate_metadata(request->pq_metadata) != 0) {
        return -1;
    }
    *out_index = NULL;
    for (chunk = 0; chunk < request->chunk_count; chunk++) {
        if (!request->code_chunks[chunk] ||
            request->chunk_vector_counts[chunk] <= 0 ||
            request->chunk_vector_counts[chunk] > request->chunk_vectors) {
            return -1;
        }
        chunk_vectors += (uint64_t)request->chunk_vector_counts[chunk];
    }
    if (chunk_vectors != request->pq_metadata->num_vectors) {
        return -1;
    }
    blocks_path = pq_residual_path(
        request->artifact_prefix, request->residual_bits, "");
    qparams_path = pq_residual_path(
        request->artifact_prefix, request->residual_bits, ".qparams");
    index = (RetrievalPqResidualIndex *)calloc(1, sizeof(*index));
    if (!blocks_path || !qparams_path || !index) {
        goto fail;
    }
    index->fd = -1;
    index->fd = open(blocks_path, O_RDONLY);
    qparams_fd = open(qparams_path, O_RDONLY);
    if (index->fd < 0 || qparams_fd < 0 ||
        pq_residual_load_header(index->fd, PQ_RESIDUAL_KIND_BLOCKS,
                                RETRIEVAL_PQ_RESIDUAL_BLOCK_BYTES,
                                &blocks_metadata, &blocks_payload_bytes,
                                blocks_hash) != 0 ||
        pq_residual_load_header(qparams_fd, PQ_RESIDUAL_KIND_QPARAMS,
                                RETRIEVAL_PQ_RESIDUAL_HEADER_BYTES,
                                &qparams_metadata, &qparams_payload_bytes,
                                qparams_hash) != 0 ||
        !pq_residual_headers_match(&blocks_metadata, &qparams_metadata) ||
        !pq_residual_metadata_matches_pq(&blocks_metadata,
                                         request->pq_metadata) ||
        blocks_metadata.residual_bits != (uint32_t)request->residual_bits ||
        blocks_payload_bytes !=
            (uint64_t)blocks_metadata.total_blocks *
                RETRIEVAL_PQ_RESIDUAL_BLOCK_BYTES ||
        qparams_payload_bytes !=
            blocks_metadata.num_vectors * RETRIEVAL_PQ_RESIDUAL_QPARAM_BYTES ||
        !pq_residual_file_size_matches(
            index->fd,
            RETRIEVAL_PQ_RESIDUAL_BLOCK_BYTES + blocks_payload_bytes) ||
        !pq_residual_file_size_matches(
            qparams_fd,
            RETRIEVAL_PQ_RESIDUAL_HEADER_BYTES + qparams_payload_bytes) ||
        qparams_payload_bytes > SIZE_MAX) {
        goto fail;
    }
    index->qparams = (RetrievalPqResidualQParams *)malloc(
        (size_t)qparams_payload_bytes);
    if (!index->qparams ||
        pq_residual_read_full(qparams_fd, index->qparams,
                              (size_t)qparams_payload_bytes,
                              RETRIEVAL_PQ_RESIDUAL_HEADER_BYTES) != 0 ||
        retrieval_pq_artifact_hash_bytes(
            index->qparams, (size_t)qparams_payload_bytes,
            actual_qparams_hash) != 0 ||
        memcmp(qparams_hash, actual_qparams_hash, sizeof(qparams_hash)) != 0) {
        goto fail;
    }
    close(qparams_fd);
    qparams_fd = -1;
    index->metadata = blocks_metadata;
    if (!rover_config.disk) {
        size_t bytes=(size_t)blocks_metadata.total_blocks*RETRIEVAL_PQ_RESIDUAL_BLOCK_BYTES;
        index->resident_blocks=malloc(bytes);
        if(!index->resident_blocks || pq_residual_read_full(index->fd,index->resident_blocks,bytes,
                RETRIEVAL_PQ_RESIDUAL_BLOCK_BYTES))goto fail;
    }
    index->cache_slots = request->cache_slots;
    index->pq_metadata = request->pq_metadata;
    index->codebook = request->codebook;
    index->code_chunks = request->code_chunks;
    index->chunk_vector_counts = request->chunk_vector_counts;
    index->chunk_count = request->chunk_count;
    index->chunk_vectors = request->chunk_vectors;
    free(blocks_path);
    free(qparams_path);
    *out_index = index;
    return 0;

fail:
    if (qparams_fd >= 0) {
        close(qparams_fd);
    }
    if (index) {
        retrieval_pq_residual_index_close(index);
    }
    free(blocks_path);
    free(qparams_path);
    return -1;
}

void retrieval_pq_residual_index_close(RetrievalPqResidualIndex *index) {
    if (!index) {
        return;
    }
    if (index->fd >= 0) {
        close(index->fd);
    }
    free(index->resident_blocks);
    free(index->qparams);
    free(index);
}

int retrieval_pq_residual_query_create(
        RetrievalPqResidualIndex *index,
        RetrievalPqResidualQuery **out_query) {
    return retrieval_pq_residual_query_create_with_slots(
        index, index ? index->cache_slots : 0, out_query);
}

int retrieval_pq_residual_query_create_with_slots(
        RetrievalPqResidualIndex *index, int cache_slots,
        RetrievalPqResidualQuery **out_query) {
    RetrievalPqResidualQuery *query;
    if (!index || !out_query || cache_slots < 1 || cache_slots > 65536 ||
        (cache_slots & (cache_slots - 1))) {
        return -1;
    }
    *out_query = NULL;
    query = (RetrievalPqResidualQuery *)calloc(1, sizeof(*query));
    if (!query) {
        return -1;
    }
    query->slots = (RetrievalPqResidualCacheSlot *)malloc(
        (size_t)cache_slots * sizeof(*query->slots));
    if (!query->slots) {
        free(query);
        return -1;
    }
    query->index = index;
    query->cache_slots = cache_slots;
    *out_query = query;
    return retrieval_pq_residual_query_reset(query, NULL);
}

int retrieval_pq_residual_query_reset(RetrievalPqResidualQuery *query,
                                      const float *query_vector) {
    int slot;
    if (!query || !query->index || !query->slots) {
        return -1;
    }
    for (slot = 0; slot < query->cache_slots; slot++) {
        query->slots[slot].block_index = -1;
    }
    return retrieval_pq_residual_query_bind(query, query_vector);
}

int retrieval_pq_residual_query_bind(RetrievalPqResidualQuery *query,
                                    const float *query_vector) {
    if (!query || !query->index || !query->slots) return -1;
    query->query_vector = query_vector;
    memset(&query->stats, 0, sizeof(query->stats));
    return 0;
}

static const uint8_t *pq_residual_query_codes(
        const RetrievalPqResidualQuery *query, int vector_id) {
    const RetrievalPqResidualIndex *index = query->index;
    int chunk = vector_id / index->chunk_vectors;
    int offset = vector_id % index->chunk_vectors;
    if (chunk < 0 || chunk >= index->chunk_count ||
        offset < 0 || offset >= index->chunk_vector_counts[chunk]) {
        return NULL;
    }
    return index->code_chunks[chunk] +
        (size_t)offset * index->pq_metadata->code_bytes;
}

static const uint8_t *pq_residual_query_block(
        RetrievalPqResidualQuery *query, int vector_id) {
    RetrievalPqResidualIndex *index = query->index;
    uint64_t block_index = (uint64_t)vector_id /
                           index->metadata.vectors_per_block;
    uint64_t within = ((uint64_t)vector_id %
                       index->metadata.vectors_per_block) *
                      index->metadata.residual_code_bytes;
    if(index->resident_blocks) {
        query->stats.cache_hits++;
        return index->resident_blocks+block_index*RETRIEVAL_PQ_RESIDUAL_BLOCK_BYTES+within;
    }
    int slot_index = (int)(block_index &
                           (uint64_t)(query->cache_slots - 1));
    RetrievalPqResidualCacheSlot *slot = &query->slots[slot_index];
    if (slot->block_index == (int64_t)block_index) {
        query->stats.cache_hits++;
        return slot->data + (size_t)within;
    }
    query->stats.cache_misses++;
    slot->block_index = -1;
    if (pq_residual_read_full(
            index->fd, slot->data, sizeof(slot->data),
            RETRIEVAL_PQ_RESIDUAL_BLOCK_BYTES +
                block_index * RETRIEVAL_PQ_RESIDUAL_BLOCK_BYTES) != 0) {
        query->stats.cache_errors++;
        return NULL;
    }
    slot->block_index = (int64_t)block_index;
    return slot->data + (size_t)within;
}

static float pq_residual_decode_value(float centroid, int code,
                                      const RetrievalPqResidualQParams *qp) {
    volatile float product = (float)code * qp->scale;
    volatile float sum = centroid + product;
    volatile float value = sum + qp->zero_point;
    return value;
}

int retrieval_pq_residual_query_decode(RetrievalPqResidualQuery *query,
                                       int vector_id, float *out_vector) {
    RetrievalPqResidualIndex *index;
    const RetrievalPqResidualQParams *qp;
    const uint8_t *codes, *residual;
    int subspace, dimension = 0;
    if (!query || !query->index || !out_vector || vector_id < 0) return -1;
    index = query->index;
    if ((uint64_t)vector_id >= index->metadata.num_vectors) return -1;
    qp = &index->qparams[vector_id];
    if (!isfinite(qp->scale) || qp->scale <= 0 || !isfinite(qp->zero_point)) return -1;
    codes = pq_residual_query_codes(query, vector_id);
    residual = pq_residual_query_block(query, vector_id);
    if (!codes || !residual) return -1;
    for (subspace = 0; subspace < (int)index->pq_metadata->m; ++subspace) {
        int centroid = pq_residual_centroid_id(codes, subspace, (int)index->pq_metadata->ks);
        const float *values = index->codebook +
            ((size_t)subspace * index->pq_metadata->ks + centroid) * index->pq_metadata->subdim;
        int inner;
        for (inner = 0; inner < (int)index->pq_metadata->subdim; ++inner, ++dimension) {
            int code = index->metadata.residual_bits == 4u ?
                (residual[dimension >> 1] >> (4 * (dimension & 1))) & 15 : residual[dimension];
            out_vector[dimension] = pq_residual_decode_value(values[inner], code, qp);
            if (!isfinite(out_vector[dimension])) return -1;
        }
    }
    return 0;
}

int retrieval_pq_residual_query_evaluate(
        RetrievalPqResidualQuery *query,
        int vector_id,
        float coarse_distance,
        float *out_distance) {
    RetrievalPqResidualIndex *index;
    const RetrievalPqResidualQParams *qparams;
    const uint8_t *pq_codes;
    const uint8_t *residual_codes;
    float distance = 0.0f;
    int dimension = 0;
    int subspace;
    (void)coarse_distance;
    if (!query || !query->index || !query->query_vector || !out_distance) {
        return -1;
    }
    index = query->index;
    if (vector_id < 0 ||
        (uint64_t)vector_id >= index->metadata.num_vectors) {
        return -1;
    }
    pq_codes = pq_residual_query_codes(query, vector_id);
    residual_codes = pq_residual_query_block(query, vector_id);
    qparams = &index->qparams[vector_id];
    if (!pq_codes || !residual_codes || !isfinite(qparams->scale) ||
        !isfinite(qparams->zero_point) || qparams->scale <= 0.0f) {
        return -1;
    }
    for (subspace = 0; subspace < (int)index->pq_metadata->m; subspace++) {
        int centroid = pq_residual_centroid_id(
            pq_codes, subspace, (int)index->pq_metadata->ks);
        const float *centroid_values = index->codebook +
            ((size_t)subspace * index->pq_metadata->ks + (size_t)centroid) *
                index->pq_metadata->subdim;
        int inner;
        for (inner = 0; inner < (int)index->pq_metadata->subdim;
             inner++, dimension++) {
            int quantized;
            float reconstructed;
            float difference;
            if (index->metadata.residual_bits == 4u) {
                uint8_t packed = residual_codes[dimension >> 1];
                quantized = (dimension & 1) ? (int)(packed >> 4)
                                              : (int)(packed & 0x0fu);
            } else {
                quantized = (int)residual_codes[dimension];
            }
            reconstructed = pq_residual_decode_value(centroid_values[inner],
                                                     quantized, qparams);
            difference = query->query_vector[dimension] - reconstructed;
            distance += difference * difference;
        }
    }
    if (!isfinite(distance)) {
        return -1;
    }
    query->stats.corrected_candidates++;
    *out_distance = distance;
    return 0;
}

void retrieval_pq_residual_query_stats(
        const RetrievalPqResidualQuery *query,
        RetrievalPqResidualStats *out_stats) {
    if (!out_stats) {
        return;
    }
    if (!query) {
        memset(out_stats, 0, sizeof(*out_stats));
        return;
    }
    *out_stats = query->stats;
}

void retrieval_pq_residual_query_destroy(RetrievalPqResidualQuery *query) {
    if (!query) {
        return;
    }
    free(query->slots);
    free(query);
}
