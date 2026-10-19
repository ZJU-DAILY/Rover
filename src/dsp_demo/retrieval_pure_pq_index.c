#include "dsp_demo/retrieval_pure_pq_index.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "calculator.h"
#include "rpcmem.h"

#include "dsp_demo/retrieval_pq_core.h"
#include "dsp_demo/retrieval_pure_pq_builder.h"
#include "dsp_demo/retrieval_sqmd_pure_pq.h"

#pragma weak fastrpc_mmap
#pragma weak fastrpc_munmap

struct PurePqIndex {
    RetrievalPqArtifacts artifacts;
    uint8_t *code_chunks[PURE_PQ_MAX_CODE_CHUNKS];
    int chunk_vector_counts[PURE_PQ_MAX_CODE_CHUNKS];
    int chunk_fds[PURE_PQ_MAX_CODE_CHUNKS];
    size_t chunk_bytes[PURE_PQ_MAX_CODE_CHUNKS];
    int chunk_count;
    int chunk_vectors;
    int domain_id;
    remote_handle64 handle;
    int dsp_loaded;
};

static void pure_pq_index_free_chunks(PurePqIndex *index) {
    int chunk;

    if (!index) {
        return;
    }
    for (chunk = 0; chunk < index->chunk_count; chunk++) {
        if (index->domain_id >= 0 && index->chunk_fds[chunk] >= 0 &&
            index->code_chunks[chunk] && fastrpc_munmap) {
            fastrpc_munmap(index->domain_id,
                           index->chunk_fds[chunk],
                           index->code_chunks[chunk],
                           index->chunk_bytes[chunk]);
        }
        if (index->code_chunks[chunk]) {
            rpcmem_free(index->code_chunks[chunk]);
        }
        index->code_chunks[chunk] = NULL;
        index->chunk_fds[chunk] = -1;
        index->chunk_vector_counts[chunk] = 0;
        index->chunk_bytes[chunk] = 0u;
    }
    index->chunk_count = 0;
    index->domain_id = -1;
}

static int pure_pq_index_validate_request(
        const PurePqPrepareRequest *request,
        const RetrievalPqArtifactMetadata *metadata) {
    if ((request->expected_num_vectors > 0 &&
         metadata->num_vectors != (uint64_t)request->expected_num_vectors) ||
        (request->expected_dim > 0 &&
         metadata->dim != (uint32_t)request->expected_dim) ||
        (request->m > 0 && metadata->m != (uint32_t)request->m) ||
        (request->ks > 0 && metadata->ks != (uint32_t)request->ks) ||
        (request->expected_layout_fingerprint &&
         memcmp(metadata->layout_fingerprint,
                request->expected_layout_fingerprint,
                RETRIEVAL_PQ_ARTIFACT_HASH_BYTES) != 0)) {
        return -1;
    }
    return 0;
}

int pure_pq_index_prepare(const PurePqPrepareRequest *request,
                          PurePqIndex **out_index) {
    PurePqIndex *index;
    int loaded;

    if (!request || !out_index || !request->artifact_prefix ||
        request->encode_chunk_vectors <= 0 ||
        request->encode_chunk_vectors > INT_MAX || request->mode < 0 ||
        request->mode > PURE_PQ_REBUILD) {
        return -1;
    }
    *out_index = NULL;
    index = (PurePqIndex *)calloc(1, sizeof(*index));
    if (!index) {
        return -1;
    }
    index->domain_id = -1;
    index->chunk_vectors = request->encode_chunk_vectors;
    loaded = retrieval_pq_artifacts_load(request->artifact_prefix,
                                         &index->artifacts) == 0;
    if (loaded && pure_pq_index_validate_request(
                      request, &index->artifacts.metadata) != 0) {
        retrieval_pq_artifacts_release(&index->artifacts);
        loaded = 0;
    }
    if (request->mode == PURE_PQ_REBUILD ||
        (request->mode == PURE_PQ_TRAIN_IF_MISSING && !loaded)) {
        PurePqBuildRequest build_request;

        retrieval_pq_artifacts_release(&index->artifacts);
        if (!request->base_fvecs_path || request->m <= 0 ||
            (request->ks != 16 && request->ks != 256) ||
            request->train_max <= 0 ||
            request->kmeans_iters <= 0 || request->memory_budget_bytes == 0u) {
            pure_pq_index_release(index);
            return -1;
        }
        memset(&build_request, 0, sizeof(build_request));
        build_request.base_fvecs_path = request->base_fvecs_path;
        build_request.order_map_path = request->order_map_path;
        build_request.artifact_prefix = request->artifact_prefix;
        build_request.expected_num_vectors = request->expected_num_vectors;
        build_request.expected_dim = request->expected_dim;
        build_request.m = request->m;
        build_request.ks = request->ks;
        build_request.train_max = request->train_max;
        build_request.kmeans_iters = request->kmeans_iters;
        build_request.encode_chunk_vectors = request->encode_chunk_vectors;
        build_request.residual_bits = request->residual_bits;
        build_request.memory_budget_bytes = request->memory_budget_bytes;
        build_request.layout_id = request->layout_id;
        build_request.layout_fingerprint =
            request->expected_layout_fingerprint;
        if (pure_pq_build_mobile_artifacts(&build_request, NULL) != 0 ||
            retrieval_pq_artifacts_load(request->artifact_prefix,
                                        &index->artifacts) != 0 ||
            pure_pq_index_validate_request(
                request, &index->artifacts.metadata) != 0) {
            pure_pq_index_release(index);
            return -1;
        }
        loaded = 1;
    }
    if (!loaded) {
        pure_pq_index_release(index);
        return -1;
    }
    *out_index = index;
    return 0;
}

int pure_pq_index_map_dsp(PurePqIndex *index,
                          remote_handle64 handle,
                          int domain_id) {
    int fds[PURE_PQ_MAX_CODE_CHUNKS];
    uint64_t num_vectors;
    uint64_t offset_vectors = 0u;
    int chunk_count;
    int chunk;
    int result;

    if (!index || index->dsp_loaded || index->chunk_count != 0 ||
        domain_id < 0 || !fastrpc_mmap) {
        return -1;
    }
    num_vectors = index->artifacts.metadata.num_vectors;
    chunk_count = (int)((num_vectors + (uint64_t)index->chunk_vectors - 1u) /
                        (uint64_t)index->chunk_vectors);
    if (chunk_count <= 0 || chunk_count > PURE_PQ_MAX_CODE_CHUNKS) {
        return -1;
    }
    index->domain_id = domain_id;
    index->handle = handle;
    for (chunk = 0; chunk < chunk_count; chunk++) {
        uint64_t remaining = num_vectors - offset_vectors;
        int vector_count = remaining > (uint64_t)index->chunk_vectors
                               ? index->chunk_vectors : (int)remaining;
        size_t bytes = (size_t)vector_count *
                       (size_t)index->artifacts.metadata.code_bytes;
        uint8_t *memory;
        int fd;

        if (bytes > (size_t)INT_MAX) {
            pure_pq_index_free_chunks(index);
            return -1;
        }
        memory = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                                         RPCMEM_DEFAULT_FLAGS,
                                         (int)bytes);
        if (!memory) {
            pure_pq_index_free_chunks(index);
            return -1;
        }
        memcpy(memory,
               index->artifacts.codes +
                   (size_t)offset_vectors *
                       (size_t)index->artifacts.metadata.code_bytes,
               bytes);
        fd = rpcmem_to_fd(memory);
        if (fd < 0 ||
            fastrpc_mmap(domain_id, fd, memory, 0, bytes,
                         FASTRPC_MAP_FD) != 0) {
            rpcmem_free(memory);
            pure_pq_index_free_chunks(index);
            return -1;
        }
        index->code_chunks[chunk] = memory;
        index->chunk_vector_counts[chunk] = vector_count;
        index->chunk_fds[chunk] = fd;
        index->chunk_bytes[chunk] = bytes;
        index->chunk_count = chunk + 1;
        fds[chunk] = fd;
        offset_vectors += (uint64_t)vector_count;
    }

    result = calculator_retrieval_pure_pq_load_dma(
        handle,
        fds, chunk_count,
        index->chunk_vector_counts, chunk_count,
        index->chunk_vectors,
        index->artifacts.codebook, (int)index->artifacts.codebook_values,
        (int)index->artifacts.metadata.num_vectors,
        (int)index->artifacts.metadata.dim,
        (int)index->artifacts.metadata.m,
        (int)index->artifacts.metadata.ks,
        (int)index->artifacts.metadata.subdim,
        (int)index->artifacts.metadata.code_bytes,
        index->artifacts.metadata.index_uuid,
        RETRIEVAL_PQ_ARTIFACT_UUID_BYTES);
    if (result != 0) {
        pure_pq_index_free_chunks(index);
        return -1;
    }
    index->dsp_loaded = 1;
    free(index->artifacts.codes);
    index->artifacts.codes = NULL;
    return 0;
}

int pure_pq_index_unmap_dsp(PurePqIndex *index) {
    int result = 0;

    if (!index) {
        return -1;
    }
    if (index->dsp_loaded) {
        result = calculator_retrieval_pure_pq_unload_dma(index->handle);
        if (result != 0) {
            return -1;
        }
        index->dsp_loaded = 0;
    }
    pure_pq_index_free_chunks(index);
    return 0;
}

const RetrievalPqArtifactMetadata *pure_pq_index_metadata(
        const PurePqIndex *index) {
    return index ? &index->artifacts.metadata : NULL;
}

int pure_pq_index_build_lut(const PurePqIndex *index,
                            const float *query,
                            float *out_lut,
                            size_t lut_count) {
    size_t expected;
    if (!index || !query || !out_lut) {
        return -1;
    }
    expected = (size_t)index->artifacts.metadata.m *
               (size_t)index->artifacts.metadata.ks;
    if (lut_count < expected) {
        return -1;
    }
    return retrieval_pq_build_lut(
        query, index->artifacts.codebook,
        (int)index->artifacts.metadata.m,
        (int)index->artifacts.metadata.ks,
        (int)index->artifacts.metadata.subdim,
        out_lut);
}

int pure_pq_index_score_ids(const PurePqIndex *index,
                            const int32_t *ids,
                            int count,
                            const float *lut,
                            float *out_distances) {
    const uint8_t *contiguous_chunks[1];
    int contiguous_counts[1];
    const uint8_t *const *chunks;
    const int *vector_counts;
    int chunk_count;
    int chunk_vectors;
    if (!index || !ids || count < 0 || !lut || !out_distances) {
        return -1;
    }
    if (index->chunk_count > 0) {
        chunks = (const uint8_t *const *)index->code_chunks;
        vector_counts = index->chunk_vector_counts;
        chunk_count = index->chunk_count;
        chunk_vectors = index->chunk_vectors;
    } else {
        if (!index->artifacts.codes) {
            return -1;
        }
        contiguous_chunks[0] = index->artifacts.codes;
        contiguous_counts[0] = (int)index->artifacts.metadata.num_vectors;
        chunks = contiguous_chunks;
        vector_counts = contiguous_counts;
        chunk_count = 1;
        chunk_vectors = contiguous_counts[0];
    }
    return retrieval_pq_adc_ids_batch(
        chunks, vector_counts, chunk_count, chunk_vectors,
        ids, count, lut,
        (int)index->artifacts.metadata.m,
        (int)index->artifacts.metadata.ks,
        out_distances);
}

int pure_pq_index_open_residual(
        PurePqIndex *index,
        const char *artifact_prefix,
        int residual_bits,
        int cache_slots,
        RetrievalPqResidualIndex **out_residual) {
    RetrievalPqResidualOpenRequest request;
    if (!index || !artifact_prefix || !out_residual ||
        !index->dsp_loaded || index->chunk_count <= 0) {
        return -1;
    }
    memset(&request, 0, sizeof(request));
    request.artifact_prefix = artifact_prefix;
    request.residual_bits = residual_bits;
    request.cache_slots = cache_slots;
    request.pq_metadata = &index->artifacts.metadata;
    request.codebook = index->artifacts.codebook;
    request.code_chunks = (const uint8_t *const *)index->code_chunks;
    request.chunk_vector_counts = index->chunk_vector_counts;
    request.chunk_count = index->chunk_count;
    request.chunk_vectors = index->chunk_vectors;
    return retrieval_pq_residual_index_open(&request, out_residual);
}

void pure_pq_index_release(PurePqIndex *index) {
    if (!index) {
        return;
    }
    pure_pq_index_unmap_dsp(index);
    retrieval_pq_artifacts_release(&index->artifacts);
    free(index);
}
