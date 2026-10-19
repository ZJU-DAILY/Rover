#include "dsp_demo/retrieval_pq_artifact.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define PQ_ENDIAN_MARKER 0x01020304u
#define PQ_METRIC_SQUARED_L2 1u
#define PQ_CODE_DTYPE_UINT8 1u
#define PQ_CODEBOOK_DTYPE_FLOAT32 1u

static const uint8_t pq_artifact_magic[8] = {
    'P', 'U', 'R', 'E', 'P', 'Q', '0', '1'
};

typedef struct PqSha256 {
    uint32_t state[8];
    uint64_t total_bytes;
    uint8_t block[64];
    size_t block_bytes;
} PqSha256;

struct RetrievalPqArtifactWriter {
    RetrievalPqArtifactMetadata metadata;
    char *prefix;
    char *book_path;
    char *book_temporary;
    char *codes_path;
    char *codes_temporary;
    FILE *codes_file;
    PqSha256 codes_hash;
    size_t expected_codes_bytes;
    size_t written_codes_bytes;
};

static uint32_t pq_rotr32(uint32_t value, unsigned shift) {
    return (value >> shift) | (value << (32u - shift));
}

static void pq_sha256_transform(PqSha256 *ctx, const uint8_t block[64]) {
    static const uint32_t constants[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
        0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
        0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
        0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
    };
    uint32_t words[64];
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
    uint32_t e;
    uint32_t f;
    uint32_t g;
    uint32_t h;
    int index;

    for (index = 0; index < 16; index++) {
        words[index] = ((uint32_t)block[index * 4] << 24) |
                       ((uint32_t)block[index * 4 + 1] << 16) |
                       ((uint32_t)block[index * 4 + 2] << 8) |
                       (uint32_t)block[index * 4 + 3];
    }
    for (index = 16; index < 64; index++) {
        uint32_t s0 = pq_rotr32(words[index - 15], 7) ^
                      pq_rotr32(words[index - 15], 18) ^
                      (words[index - 15] >> 3);
        uint32_t s1 = pq_rotr32(words[index - 2], 17) ^
                      pq_rotr32(words[index - 2], 19) ^
                      (words[index - 2] >> 10);
        words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }

    a = ctx->state[0];
    b = ctx->state[1];
    c = ctx->state[2];
    d = ctx->state[3];
    e = ctx->state[4];
    f = ctx->state[5];
    g = ctx->state[6];
    h = ctx->state[7];

    for (index = 0; index < 64; index++) {
        uint32_t sum1 = pq_rotr32(e, 6) ^ pq_rotr32(e, 11) ^
                        pq_rotr32(e, 25);
        uint32_t choose = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + sum1 + choose + constants[index] + words[index];
        uint32_t sum0 = pq_rotr32(a, 2) ^ pq_rotr32(a, 13) ^
                        pq_rotr32(a, 22);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = sum0 + majority;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
    ctx->state[5] += f;
    ctx->state[6] += g;
    ctx->state[7] += h;
}

static void pq_sha256_init(PqSha256 *ctx) {
    static const uint32_t initial[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };
    memcpy(ctx->state, initial, sizeof(initial));
    ctx->total_bytes = 0u;
    ctx->block_bytes = 0u;
}

static void pq_sha256_update(PqSha256 *ctx, const uint8_t *data, size_t bytes) {
    while (bytes > 0u) {
        size_t available = sizeof(ctx->block) - ctx->block_bytes;
        size_t take = bytes < available ? bytes : available;

        memcpy(ctx->block + ctx->block_bytes, data, take);
        ctx->block_bytes += take;
        ctx->total_bytes += take;
        data += take;
        bytes -= take;
        if (ctx->block_bytes == sizeof(ctx->block)) {
            pq_sha256_transform(ctx, ctx->block);
            ctx->block_bytes = 0u;
        }
    }
}

static void pq_sha256_final(PqSha256 *ctx, uint8_t digest[32]) {
    uint64_t total_bits = ctx->total_bytes * 8u;
    int index;

    ctx->block[ctx->block_bytes++] = 0x80u;
    if (ctx->block_bytes > 56u) {
        memset(ctx->block + ctx->block_bytes, 0,
               sizeof(ctx->block) - ctx->block_bytes);
        pq_sha256_transform(ctx, ctx->block);
        ctx->block_bytes = 0u;
    }
    memset(ctx->block + ctx->block_bytes, 0, 56u - ctx->block_bytes);
    for (index = 0; index < 8; index++) {
        ctx->block[63 - index] = (uint8_t)(total_bits >> (index * 8));
    }
    pq_sha256_transform(ctx, ctx->block);

    for (index = 0; index < 8; index++) {
        digest[index * 4] = (uint8_t)(ctx->state[index] >> 24);
        digest[index * 4 + 1] = (uint8_t)(ctx->state[index] >> 16);
        digest[index * 4 + 2] = (uint8_t)(ctx->state[index] >> 8);
        digest[index * 4 + 3] = (uint8_t)ctx->state[index];
    }
}

static void pq_sha256_bytes(const uint8_t *data, size_t bytes,
                            uint8_t digest[32]) {
    PqSha256 ctx;
    pq_sha256_init(&ctx);
    pq_sha256_update(&ctx, data, bytes);
    pq_sha256_final(&ctx, digest);
}

int retrieval_pq_artifact_hash_bytes(const void *data,
                                     size_t bytes,
                                     uint8_t out_hash[32]) {
    if ((!data && bytes != 0u) || !out_hash) {
        return -1;
    }
    pq_sha256_bytes((const uint8_t *)data, bytes, out_hash);
    return 0;
}

int retrieval_pq_artifact_hash_file_prefix(const char *path,
                                           uint64_t bytes,
                                           uint8_t out_hash[32]) {
    return retrieval_pq_artifact_hash_file_range(path, 0u, bytes, out_hash);
}

int retrieval_pq_artifact_hash_file_range(const char *path,
                                          uint64_t offset,
                                          uint64_t bytes,
                                          uint8_t out_hash[32]) {
    uint8_t buffer[4 * 1024];
    PqSha256 context;
    int fd;

    if (!path || !out_hash || offset > (uint64_t)INT64_MAX) {
        return -1;
    }
    fd = open(path, O_RDONLY);
    if (fd < 0) {
        return -1;
    }
    pq_sha256_init(&context);
    while (bytes > 0u) {
        size_t wanted = bytes > sizeof(buffer) ? sizeof(buffer) : (size_t)bytes;
        ssize_t actual = pread(fd, buffer, wanted, (off_t)offset);
        if (actual < 0 && errno == EINTR) {
            continue;
        }
        if (actual != (ssize_t)wanted) {
            close(fd);
            return -1;
        }
        pq_sha256_update(&context, buffer, wanted);
        offset += (uint64_t)wanted;
        bytes -= (uint64_t)wanted;
    }
    if (close(fd) != 0) {
        return -1;
    }
    pq_sha256_final(&context, out_hash);
    return 0;
}

static void pq_put_u16(uint8_t *output, uint16_t value) {
    output[0] = (uint8_t)value;
    output[1] = (uint8_t)(value >> 8);
}

static void pq_put_u32(uint8_t *output, uint32_t value) {
    output[0] = (uint8_t)value;
    output[1] = (uint8_t)(value >> 8);
    output[2] = (uint8_t)(value >> 16);
    output[3] = (uint8_t)(value >> 24);
}

static void pq_put_u64(uint8_t *output, uint64_t value) {
    int index;
    for (index = 0; index < 8; index++) {
        output[index] = (uint8_t)(value >> (index * 8));
    }
}

static uint16_t pq_get_u16(const uint8_t *input) {
    return (uint16_t)((uint16_t)input[0] | ((uint16_t)input[1] << 8));
}

static uint32_t pq_get_u32(const uint8_t *input) {
    return (uint32_t)input[0] | ((uint32_t)input[1] << 8) |
           ((uint32_t)input[2] << 16) | ((uint32_t)input[3] << 24);
}

static uint64_t pq_get_u64(const uint8_t *input) {
    uint64_t value = 0u;
    int index;
    for (index = 0; index < 8; index++) {
        value |= (uint64_t)input[index] << (index * 8);
    }
    return value;
}

static int pq_size_mul(size_t left, size_t right, size_t *result) {
    if (!result || (left != 0u && right > SIZE_MAX / left)) {
        return -1;
    }
    *result = left * right;
    return 0;
}

static int pq_uuid_is_zero(const uint8_t uuid[16]) {
    uint8_t combined = 0u;
    int index;
    for (index = 0; index < 16; index++) {
        combined |= uuid[index];
    }
    return combined == 0u;
}

int retrieval_pq_artifact_validate_metadata(
    const RetrievalPqArtifactMetadata *metadata) {
    uint32_t expected_code_bytes;

    if (!metadata || (metadata->ks != 16u && metadata->ks != 256u)) {
        return -1;
    }
    expected_code_bytes = metadata->ks == 16u
                              ? metadata->m / 2u + metadata->m % 2u
                              : metadata->m;
    if (pq_uuid_is_zero(metadata->index_uuid) ||
        metadata->num_vectors == 0u || metadata->num_vectors > SIZE_MAX ||
        metadata->dim == 0u || metadata->m == 0u ||
        metadata->subdim == 0u || metadata->m > UINT32_MAX / metadata->subdim ||
        metadata->m * metadata->subdim != metadata->dim ||
        metadata->code_bytes != expected_code_bytes ||
        metadata->train_count == 0u ||
        metadata->train_count > metadata->num_vectors ||
        metadata->kmeans_iters == 0u) {
        return -1;
    }
    return 0;
}

static int pq_payload_sizes(const RetrievalPqArtifactMetadata *metadata,
                            size_t *codebook_values,
                            size_t *codebook_bytes,
                            size_t *codes_bytes) {
    size_t values;
    size_t bytes;

    if (retrieval_pq_artifact_validate_metadata(metadata) != 0 ||
        pq_size_mul((size_t)metadata->m, (size_t)metadata->ks, &values) != 0 ||
        pq_size_mul(values, (size_t)metadata->subdim, &values) != 0 ||
        pq_size_mul(values, sizeof(float), &bytes) != 0 ||
        pq_size_mul((size_t)metadata->num_vectors,
                    (size_t)metadata->code_bytes, codes_bytes) != 0) {
        return -1;
    }
    *codebook_values = values;
    *codebook_bytes = bytes;
    return 0;
}

static void pq_encode_header(uint8_t header[256], uint32_t kind,
                             const RetrievalPqArtifactMetadata *metadata,
                             uint64_t payload_bytes,
                             const uint8_t payload_hash[32]) {
    memset(header, 0, RETRIEVAL_PQ_ARTIFACT_HEADER_BYTES);
    memcpy(header, pq_artifact_magic, sizeof(pq_artifact_magic));
    pq_put_u16(header + 8, RETRIEVAL_PQ_ARTIFACT_VERSION_MAJOR);
    pq_put_u16(header + 10, RETRIEVAL_PQ_ARTIFACT_VERSION_MINOR);
    pq_put_u32(header + 12, RETRIEVAL_PQ_ARTIFACT_HEADER_BYTES);
    pq_put_u32(header + 16, kind);
    pq_put_u32(header + 20, PQ_ENDIAN_MARKER);
    memcpy(header + 32, metadata->index_uuid, 16);
    pq_put_u64(header + 48, metadata->num_vectors);
    pq_put_u32(header + 56, metadata->dim);
    pq_put_u32(header + 60, metadata->m);
    pq_put_u32(header + 64, metadata->ks);
    pq_put_u32(header + 68, metadata->subdim);
    pq_put_u32(header + 72, metadata->code_bytes);
    pq_put_u32(header + 76, PQ_METRIC_SQUARED_L2);
    pq_put_u32(header + 80, PQ_CODE_DTYPE_UINT8);
    pq_put_u32(header + 84, PQ_CODEBOOK_DTYPE_FLOAT32);
    pq_put_u32(header + 88, metadata->trainer_kind);
    pq_put_u32(header + 92, metadata->sample_strategy);
    pq_put_u64(header + 96, metadata->train_count);
    pq_put_u32(header + 104, metadata->kmeans_iters);
    pq_put_u32(header + 108, metadata->seed);
    pq_put_u64(header + 112, payload_bytes);
    memcpy(header + 120, payload_hash, 32);
    memcpy(header + 152, metadata->dataset_fingerprint, 32);
    memcpy(header + 184, metadata->layout_fingerprint, 32);
}

static int pq_decode_header(const uint8_t header[256], uint32_t expected_kind,
                            RetrievalPqArtifactMetadata *metadata,
                            uint64_t *payload_bytes,
                            uint8_t payload_hash[32]) {
    if (memcmp(header, pq_artifact_magic, sizeof(pq_artifact_magic)) != 0 ||
        pq_get_u16(header + 8) != RETRIEVAL_PQ_ARTIFACT_VERSION_MAJOR ||
        pq_get_u16(header + 10) > RETRIEVAL_PQ_ARTIFACT_VERSION_MINOR ||
        pq_get_u32(header + 12) != RETRIEVAL_PQ_ARTIFACT_HEADER_BYTES ||
        pq_get_u32(header + 16) != expected_kind ||
        pq_get_u32(header + 20) != PQ_ENDIAN_MARKER ||
        pq_get_u32(header + 76) != PQ_METRIC_SQUARED_L2 ||
        pq_get_u32(header + 80) != PQ_CODE_DTYPE_UINT8 ||
        pq_get_u32(header + 84) != PQ_CODEBOOK_DTYPE_FLOAT32) {
        return -1;
    }

    memset(metadata, 0, sizeof(*metadata));
    memcpy(metadata->index_uuid, header + 32, 16);
    metadata->num_vectors = pq_get_u64(header + 48);
    metadata->dim = pq_get_u32(header + 56);
    metadata->m = pq_get_u32(header + 60);
    metadata->ks = pq_get_u32(header + 64);
    metadata->subdim = pq_get_u32(header + 68);
    metadata->code_bytes = pq_get_u32(header + 72);
    metadata->trainer_kind = pq_get_u32(header + 88);
    metadata->sample_strategy = pq_get_u32(header + 92);
    metadata->train_count = pq_get_u64(header + 96);
    metadata->kmeans_iters = pq_get_u32(header + 104);
    metadata->seed = pq_get_u32(header + 108);
    *payload_bytes = pq_get_u64(header + 112);
    memcpy(payload_hash, header + 120, 32);
    memcpy(metadata->dataset_fingerprint, header + 152, 32);
    memcpy(metadata->layout_fingerprint, header + 184, 32);
    return retrieval_pq_artifact_validate_metadata(metadata);
}

static char *pq_path_with_suffix(const char *prefix, const char *suffix) {
    size_t prefix_bytes;
    size_t suffix_bytes;
    char *path;

    if (!prefix || !suffix) {
        return NULL;
    }
    prefix_bytes = strlen(prefix);
    suffix_bytes = strlen(suffix);
    if (prefix_bytes > SIZE_MAX - suffix_bytes - 1u) {
        return NULL;
    }
    path = (char *)malloc(prefix_bytes + suffix_bytes + 1u);
    if (!path) {
        return NULL;
    }
    memcpy(path, prefix, prefix_bytes);
    memcpy(path + prefix_bytes, suffix, suffix_bytes + 1u);
    return path;
}

static int pq_fsync_parent(const char *path) {
    char *parent;
    char *slash;
    int fd;
    int result;

    parent = (char *)malloc(strlen(path) + 1u);
    if (!parent) {
        return -1;
    }
    strcpy(parent, path);
    slash = strrchr(parent, '/');
    if (slash) {
        if (slash == parent) {
            slash[1] = '\0';
        } else {
            *slash = '\0';
        }
    } else {
        strcpy(parent, ".");
    }
#ifdef O_DIRECTORY
    fd = open(parent, O_RDONLY | O_DIRECTORY);
#else
    fd = open(parent, O_RDONLY);
#endif
    free(parent);
    if (fd < 0) {
        return -1;
    }
    result = fsync(fd);
    close(fd);
    return result;
}

static int pq_write_atomic_file(const char *path,
                                const uint8_t header[256],
                                const uint8_t *payload,
                                size_t payload_bytes) {
    char *temporary = pq_path_with_suffix(path, ".tmp");
    FILE *file = NULL;
    int result = -1;

    if (!temporary) {
        return -1;
    }
    file = fopen(temporary, "wb");
    if (!file || fwrite(header, 1, 256, file) != 256u ||
        (payload_bytes > 0u &&
         fwrite(payload, 1, payload_bytes, file) != payload_bytes) ||
        fflush(file) != 0 || fsync(fileno(file)) != 0) {
        goto done;
    }
    if (fclose(file) != 0) {
        file = NULL;
        goto done;
    }
    file = NULL;
    if (rename(temporary, path) != 0 || pq_fsync_parent(path) != 0) {
        goto done;
    }
    result = 0;

done:
    if (file) {
        fclose(file);
    }
    if (result != 0) {
        unlink(temporary);
    }
    free(temporary);
    return result;
}

static int pq_encode_codebook(const float *codebook, size_t values,
                              uint8_t **out_payload) {
    uint8_t *payload;
    size_t index;

    if (values > SIZE_MAX / sizeof(float)) {
        return -1;
    }
    payload = (uint8_t *)malloc(values * sizeof(float));
    if (!payload) {
        return -1;
    }
    for (index = 0; index < values; index++) {
        uint32_t bits;
        memcpy(&bits, codebook + index, sizeof(bits));
        pq_put_u32(payload + index * sizeof(float), bits);
    }
    *out_payload = payload;
    return 0;
}

static int pq_write_temporary_file(const char *temporary,
                                   const uint8_t header[256],
                                   const uint8_t *payload,
                                   size_t payload_bytes) {
    FILE *file = fopen(temporary, "wb");
    int result = -1;

    if (!file || fwrite(header, 1, 256, file) != 256u ||
        (payload_bytes > 0u &&
         fwrite(payload, 1, payload_bytes, file) != payload_bytes) ||
        fflush(file) != 0 || fsync(fileno(file)) != 0) {
        if (file) {
            fclose(file);
        }
        unlink(temporary);
        return -1;
    }
    if (fclose(file) != 0) {
        unlink(temporary);
        return -1;
    }
    result = 0;
    return result;
}

static void pq_hex(const uint8_t *bytes, size_t count, char *output) {
    static const char digits[] = "0123456789abcdef";
    size_t index;
    for (index = 0; index < count; index++) {
        output[index * 2] = digits[bytes[index] >> 4];
        output[index * 2 + 1] = digits[bytes[index] & 15u];
    }
    output[count * 2] = '\0';
}

static int pq_write_manifest(const char *prefix,
                             const RetrievalPqArtifactMetadata *metadata) {
    char *path = pq_path_with_suffix(prefix, ".pq.json");
    char *temporary = NULL;
    char uuid[33];
    char dataset[65];
    char layout[65];
    FILE *file = NULL;
    int result = -1;

    if (!path) {
        return -1;
    }
    temporary = pq_path_with_suffix(path, ".tmp");
    if (!temporary) {
        free(path);
        return -1;
    }
    pq_hex(metadata->index_uuid, 16, uuid);
    pq_hex(metadata->dataset_fingerprint, 32, dataset);
    pq_hex(metadata->layout_fingerprint, 32, layout);
    file = fopen(temporary, "wb");
    if (!file) {
        goto done;
    }
    if (fprintf(file,
                "{\n"
                "  \"format\": \"pure-pq-v1\",\n"
                "  \"index_uuid\": \"%s\",\n"
                "  \"num_vectors\": %llu,\n"
                "  \"dim\": %u,\n"
                "  \"m\": %u,\n"
                "  \"ks\": %u,\n"
                "  \"subdim\": %u,\n"
                "  \"code_bits\": %u,\n"
                "  \"code_bytes\": %u,\n"
                "  \"trainer_kind\": %u,\n"
                "  \"sample_strategy\": %u,\n"
                "  \"train_count\": %llu,\n"
                "  \"kmeans_iters\": %u,\n"
                "  \"seed\": %u,\n"
                "  \"dataset_fingerprint\": \"%s\",\n"
                "  \"layout_fingerprint\": \"%s\"\n"
                "}\n",
                uuid, (unsigned long long)metadata->num_vectors,
                metadata->dim, metadata->m, metadata->ks, metadata->subdim,
                metadata->ks == 16u ? 4u : 8u,
                metadata->code_bytes, metadata->trainer_kind,
                metadata->sample_strategy,
                (unsigned long long)metadata->train_count,
                metadata->kmeans_iters, metadata->seed, dataset, layout) < 0 ||
        fflush(file) != 0 || fsync(fileno(file)) != 0) {
        goto done;
    }
    if (fclose(file) != 0) {
        file = NULL;
        goto done;
    }
    file = NULL;
    if (rename(temporary, path) != 0 || pq_fsync_parent(path) != 0) {
        goto done;
    }
    result = 0;

done:
    if (file) {
        fclose(file);
    }
    if (result != 0) {
        unlink(temporary);
    }
    free(temporary);
    free(path);
    return result;
}

int retrieval_pq_artifacts_write(const char *artifact_prefix,
                                 const RetrievalPqArtifactMetadata *metadata,
                                 const float *codebook,
                                 const uint8_t *codes) {
    char *book_path = NULL;
    char *codes_path = NULL;
    uint8_t *book_payload = NULL;
    uint8_t book_header[256];
    uint8_t codes_header[256];
    uint8_t book_hash[32];
    uint8_t codes_hash[32];
    size_t codebook_values;
    size_t codebook_bytes;
    size_t codes_bytes;
    int result = -1;

    if (!artifact_prefix || !codebook || !codes ||
        pq_payload_sizes(metadata, &codebook_values, &codebook_bytes,
                         &codes_bytes) != 0 ||
        pq_encode_codebook(codebook, codebook_values, &book_payload) != 0) {
        return -1;
    }
    book_path = pq_path_with_suffix(artifact_prefix, ".pqbook");
    codes_path = pq_path_with_suffix(artifact_prefix, ".pqcodes");
    if (!book_path || !codes_path) {
        goto done;
    }

    pq_sha256_bytes(book_payload, codebook_bytes, book_hash);
    pq_sha256_bytes(codes, codes_bytes, codes_hash);
    pq_encode_header(book_header, RETRIEVAL_PQ_ARTIFACT_CODEBOOK,
                     metadata, codebook_bytes, book_hash);
    pq_encode_header(codes_header, RETRIEVAL_PQ_ARTIFACT_CODES,
                     metadata, codes_bytes, codes_hash);
    if (pq_write_atomic_file(book_path, book_header,
                             book_payload, codebook_bytes) != 0 ||
        pq_write_atomic_file(codes_path, codes_header,
                             codes, codes_bytes) != 0 ||
        pq_write_manifest(artifact_prefix, metadata) != 0) {
        goto done;
    }
    result = 0;

done:
    free(book_payload);
    free(book_path);
    free(codes_path);
    return result;
}

int retrieval_pq_artifact_writer_begin(
        const char *artifact_prefix,
        const RetrievalPqArtifactMetadata *metadata,
        const float *codebook,
        RetrievalPqArtifactWriter **out_writer) {
    RetrievalPqArtifactWriter *writer = NULL;
    uint8_t *book_payload = NULL;
    uint8_t book_hash[32];
    uint8_t book_header[256];
    size_t codebook_values;
    size_t codebook_bytes;
    size_t codes_bytes;

    if (!artifact_prefix || !metadata || !codebook || !out_writer ||
        pq_payload_sizes(metadata, &codebook_values, &codebook_bytes,
                         &codes_bytes) != 0 ||
        pq_encode_codebook(codebook, codebook_values, &book_payload) != 0) {
        return -1;
    }
    *out_writer = NULL;
    writer = (RetrievalPqArtifactWriter *)calloc(1, sizeof(*writer));
    if (!writer) {
        free(book_payload);
        return -1;
    }
    writer->metadata = *metadata;
    writer->prefix = pq_path_with_suffix(artifact_prefix, "");
    writer->book_path = pq_path_with_suffix(artifact_prefix, ".pqbook");
    writer->codes_path = pq_path_with_suffix(artifact_prefix, ".pqcodes");
    if (writer->book_path) {
        writer->book_temporary = pq_path_with_suffix(writer->book_path, ".tmp");
    }
    if (writer->codes_path) {
        writer->codes_temporary = pq_path_with_suffix(writer->codes_path,
                                                       ".tmp");
    }
    if (!writer->prefix || !writer->book_path || !writer->codes_path ||
        !writer->book_temporary || !writer->codes_temporary) {
        free(book_payload);
        retrieval_pq_artifact_writer_abort(writer);
        return -1;
    }

    pq_sha256_bytes(book_payload, codebook_bytes, book_hash);
    pq_encode_header(book_header, RETRIEVAL_PQ_ARTIFACT_CODEBOOK,
                     metadata, codebook_bytes, book_hash);
    if (pq_write_temporary_file(writer->book_temporary, book_header,
                                book_payload, codebook_bytes) != 0) {
        free(book_payload);
        retrieval_pq_artifact_writer_abort(writer);
        return -1;
    }
    free(book_payload);
    writer->codes_file = fopen(writer->codes_temporary, "w+b");
    if (!writer->codes_file ||
        fwrite((uint8_t[256]){0}, 1, 256, writer->codes_file) != 256u) {
        retrieval_pq_artifact_writer_abort(writer);
        return -1;
    }
    writer->expected_codes_bytes = codes_bytes;
    pq_sha256_init(&writer->codes_hash);
    *out_writer = writer;
    return 0;
}

int retrieval_pq_artifact_writer_append_codes(
        RetrievalPqArtifactWriter *writer,
        const uint8_t *codes,
        size_t code_bytes) {
    if (!writer || !writer->codes_file || (!codes && code_bytes != 0u) ||
        code_bytes > writer->expected_codes_bytes - writer->written_codes_bytes ||
        (code_bytes > 0u &&
         fwrite(codes, 1, code_bytes, writer->codes_file) != code_bytes)) {
        return -1;
    }
    pq_sha256_update(&writer->codes_hash, codes, code_bytes);
    writer->written_codes_bytes += code_bytes;
    return 0;
}

int retrieval_pq_artifact_writer_commit(RetrievalPqArtifactWriter *writer) {
    uint8_t codes_hash[32];
    uint8_t codes_header[256];

    if (!writer || !writer->codes_file ||
        writer->written_codes_bytes != writer->expected_codes_bytes) {
        return -1;
    }
    pq_sha256_final(&writer->codes_hash, codes_hash);
    pq_encode_header(codes_header, RETRIEVAL_PQ_ARTIFACT_CODES,
                     &writer->metadata, writer->written_codes_bytes,
                     codes_hash);
    if (fseek(writer->codes_file, 0, SEEK_SET) != 0 ||
        fwrite(codes_header, 1, sizeof(codes_header), writer->codes_file) !=
            sizeof(codes_header) ||
        fflush(writer->codes_file) != 0 ||
        fsync(fileno(writer->codes_file)) != 0 ||
        fclose(writer->codes_file) != 0) {
        writer->codes_file = NULL;
        return -1;
    }
    writer->codes_file = NULL;
    if (rename(writer->book_temporary, writer->book_path) != 0 ||
        rename(writer->codes_temporary, writer->codes_path) != 0 ||
        pq_write_manifest(writer->prefix, &writer->metadata) != 0 ||
        pq_fsync_parent(writer->book_path) != 0) {
        return -1;
    }
    free(writer->prefix);
    free(writer->book_path);
    free(writer->book_temporary);
    free(writer->codes_path);
    free(writer->codes_temporary);
    free(writer);
    return 0;
}

void retrieval_pq_artifact_writer_abort(RetrievalPqArtifactWriter *writer) {
    if (!writer) {
        return;
    }
    if (writer->codes_file) {
        fclose(writer->codes_file);
    }
    if (writer->book_temporary) {
        unlink(writer->book_temporary);
    }
    if (writer->codes_temporary) {
        unlink(writer->codes_temporary);
    }
    free(writer->prefix);
    free(writer->book_path);
    free(writer->book_temporary);
    free(writer->codes_path);
    free(writer->codes_temporary);
    free(writer);
}

static int pq_read_artifact(const char *path, uint32_t kind,
                            RetrievalPqArtifactMetadata *metadata,
                            uint8_t **payload, size_t *payload_bytes) {
    uint8_t header[256];
    uint8_t expected_hash[32];
    uint8_t actual_hash[32];
    uint64_t encoded_bytes;
    uint8_t *data = NULL;
    FILE *file = NULL;
    int trailing;
    int result = -1;

    file = fopen(path, "rb");
    if (!file || fread(header, 1, sizeof(header), file) != sizeof(header) ||
        pq_decode_header(header, kind, metadata, &encoded_bytes,
                         expected_hash) != 0 || encoded_bytes > SIZE_MAX) {
        goto done;
    }
    data = (uint8_t *)malloc(encoded_bytes == 0u ? 1u : (size_t)encoded_bytes);
    if (!data || fread(data, 1, (size_t)encoded_bytes, file) != encoded_bytes) {
        goto done;
    }
    trailing = fgetc(file);
    if (trailing != EOF || ferror(file)) {
        goto done;
    }
    pq_sha256_bytes(data, (size_t)encoded_bytes, actual_hash);
    if (memcmp(actual_hash, expected_hash, sizeof(actual_hash)) != 0) {
        goto done;
    }
    *payload = data;
    *payload_bytes = (size_t)encoded_bytes;
    data = NULL;
    result = 0;

done:
    if (file) {
        fclose(file);
    }
    free(data);
    return result;
}

static int pq_metadata_equal(const RetrievalPqArtifactMetadata *left,
                             const RetrievalPqArtifactMetadata *right) {
    return left->num_vectors == right->num_vectors &&
           left->dim == right->dim && left->m == right->m &&
           left->ks == right->ks && left->subdim == right->subdim &&
           left->code_bytes == right->code_bytes &&
           left->trainer_kind == right->trainer_kind &&
           left->sample_strategy == right->sample_strategy &&
           left->train_count == right->train_count &&
           left->kmeans_iters == right->kmeans_iters &&
           left->seed == right->seed &&
           memcmp(left->index_uuid, right->index_uuid, 16) == 0 &&
           memcmp(left->dataset_fingerprint,
                  right->dataset_fingerprint, 32) == 0 &&
           memcmp(left->layout_fingerprint,
                  right->layout_fingerprint, 32) == 0;
}

int retrieval_pq_artifacts_load(const char *artifact_prefix,
                                RetrievalPqArtifacts *out_artifacts) {
    RetrievalPqArtifactMetadata book_metadata;
    RetrievalPqArtifactMetadata codes_metadata;
    char *book_path = NULL;
    char *codes_path = NULL;
    uint8_t *book_payload = NULL;
    uint8_t *codes_payload = NULL;
    size_t book_payload_bytes = 0u;
    size_t codes_payload_bytes = 0u;
    size_t expected_values;
    size_t expected_book_bytes;
    size_t expected_codes_bytes;
    float *codebook = NULL;
    size_t index;
    int result = -1;

    if (!artifact_prefix || !out_artifacts) {
        return -1;
    }
    memset(out_artifacts, 0, sizeof(*out_artifacts));
    book_path = pq_path_with_suffix(artifact_prefix, ".pqbook");
    codes_path = pq_path_with_suffix(artifact_prefix, ".pqcodes");
    if (!book_path || !codes_path ||
        pq_read_artifact(book_path, RETRIEVAL_PQ_ARTIFACT_CODEBOOK,
                         &book_metadata, &book_payload,
                         &book_payload_bytes) != 0 ||
        pq_read_artifact(codes_path, RETRIEVAL_PQ_ARTIFACT_CODES,
                         &codes_metadata, &codes_payload,
                         &codes_payload_bytes) != 0 ||
        !pq_metadata_equal(&book_metadata, &codes_metadata) ||
        pq_payload_sizes(&book_metadata, &expected_values,
                         &expected_book_bytes, &expected_codes_bytes) != 0 ||
        book_payload_bytes != expected_book_bytes ||
        codes_payload_bytes != expected_codes_bytes) {
        goto done;
    }

    codebook = (float *)malloc(expected_book_bytes);
    if (!codebook) {
        goto done;
    }
    for (index = 0; index < expected_values; index++) {
        uint32_t bits = pq_get_u32(book_payload + index * sizeof(float));
        float value;
        memcpy(&value, &bits, sizeof(value));
        if (!(value == value) || value > 3.402823466e+38F ||
            value < -3.402823466e+38F) {
            goto done;
        }
        codebook[index] = value;
    }

    out_artifacts->metadata = book_metadata;
    out_artifacts->codebook = codebook;
    out_artifacts->codes = codes_payload;
    out_artifacts->codebook_values = expected_values;
    out_artifacts->codes_bytes = expected_codes_bytes;
    codebook = NULL;
    codes_payload = NULL;
    result = 0;

done:
    free(book_path);
    free(codes_path);
    free(book_payload);
    free(codes_payload);
    free(codebook);
    if (result != 0) {
        retrieval_pq_artifacts_release(out_artifacts);
    }
    return result;
}

void retrieval_pq_artifacts_release(RetrievalPqArtifacts *artifacts) {
    if (!artifacts) {
        return;
    }
    free(artifacts->codebook);
    free(artifacts->codes);
    memset(artifacts, 0, sizeof(*artifacts));
}
