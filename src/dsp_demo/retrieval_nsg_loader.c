#include "dsp_demo/retrieval_nsg_loader.h"

#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define NSG_DISCARD_CHUNK_IDS 256u
#define NSG_DROP_CACHE_INTERVAL_MASK 0x3FFFF

static void nsg_set_error(char *error,
                          size_t error_size,
                          const char *format,
                          ...) {
    va_list args;

    if (!error || error_size == 0)
        return;
    va_start(args, format);
    vsnprintf(error, error_size, format, args);
    va_end(args);
}

int retrieval_nsg_load_fixed(const char *path,
                             int num_vectors,
                             int runtime_width,
                             int32_t *graph,
                             RetrievalNsgLoadStats *stats,
                             char *error,
                             size_t error_size) {
    RetrievalNsgLoadStats loaded;
    uint32_t discard[NSG_DISCARD_CHUNK_IDS];
    uint32_t header[2];
    FILE *file = NULL;
    int result = -1;
    int node;

    memset(&loaded, 0, sizeof(loaded));
    loaded.entry_point = -1;
    if (stats)
        *stats = loaded;
    if (error && error_size > 0)
        error[0] = '\0';

    if (!path || !graph || !stats || num_vectors <= 0 || runtime_width <= 0) {
        nsg_set_error(error, error_size, "invalid loader arguments");
        return -1;
    }

    file = fopen(path, "rb");
    if (!file) {
        nsg_set_error(error, error_size, "cannot open %s", path);
        return -1;
    }
    posix_fadvise(fileno(file), 0, 0, POSIX_FADV_SEQUENTIAL);

    if (fread(header, sizeof(header[0]), 2, file) != 2) {
        nsg_set_error(error, error_size, "truncated NSG header");
        goto cleanup;
    }
    if (header[0] == 0 || header[0] > (uint32_t)num_vectors) {
        nsg_set_error(error, error_size,
                      "invalid NSG width %u for N=%d",
                      header[0], num_vectors);
        goto cleanup;
    }
    if (header[1] >= (uint32_t)num_vectors) {
        nsg_set_error(error, error_size,
                      "invalid NSG entry point %u for N=%d",
                      header[1], num_vectors);
        goto cleanup;
    }
    loaded.file_width = header[0];
    loaded.entry_point = (int32_t)header[1];

    for (node = 0; node < num_vectors; node++) {
        int32_t *dst = graph + (size_t)node * (size_t)runtime_width;
        uint32_t degree;
        uint32_t keep;
        uint32_t index;

        if (fread(&degree, sizeof(degree), 1, file) != 1) {
            nsg_set_error(error, error_size,
                          "truncated degree at node %d", node);
            goto cleanup;
        }
        if (degree > loaded.file_width) {
            nsg_set_error(error, error_size,
                          "node %d degree %u exceeds header width %u",
                          node, degree, loaded.file_width);
            goto cleanup;
        }

        keep = degree < (uint32_t)runtime_width
                   ? degree
                   : (uint32_t)runtime_width;
        if (keep > 0 &&
            fread(dst, sizeof(dst[0]), keep, file) != keep) {
            nsg_set_error(error, error_size,
                          "truncated neighbor payload at node %d", node);
            goto cleanup;
        }

        for (index = 0; index < keep; index++) {
            int32_t neighbor = dst[index];
            uint32_t previous;

            if (neighbor < 0 || neighbor >= num_vectors) {
                nsg_set_error(error, error_size,
                              "node %d has invalid neighbor %d",
                              node, neighbor);
                goto cleanup;
            }
            if (neighbor == node) {
                nsg_set_error(error, error_size,
                              "node %d has a self edge", node);
                goto cleanup;
            }
            for (previous = 0; previous < index; previous++) {
                if (dst[previous] == neighbor) {
                    nsg_set_error(error, error_size,
                                  "node %d has duplicate neighbor %d",
                                  node, neighbor);
                    goto cleanup;
                }
            }
        }

        if (degree > keep) {
            uint32_t remaining = degree - keep;

            loaded.truncated_nodes++;
            loaded.truncated_edges += (uint64_t)remaining;
            while (remaining > 0) {
                uint32_t chunk = remaining < NSG_DISCARD_CHUNK_IDS
                                     ? remaining
                                     : NSG_DISCARD_CHUNK_IDS;

                if (fread(discard, sizeof(discard[0]), chunk, file) != chunk) {
                    nsg_set_error(error, error_size,
                                  "truncated excess payload at node %d", node);
                    goto cleanup;
                }
                for (index = 0; index < chunk; index++) {
                    if (discard[index] >= (uint32_t)num_vectors) {
                        nsg_set_error(error, error_size,
                                      "node %d has invalid excess neighbor %u",
                                      node, discard[index]);
                        goto cleanup;
                    }
                    if (discard[index] == (uint32_t)node) {
                        nsg_set_error(error, error_size,
                                      "node %d has an excess self edge", node);
                        goto cleanup;
                    }
                }
                remaining -= chunk;
            }
        }

        for (index = keep; index < (uint32_t)runtime_width; index++)
            dst[index] = -1;

        if ((node & NSG_DROP_CACHE_INTERVAL_MASK) ==
            NSG_DROP_CACHE_INTERVAL_MASK) {
            off_t offset = ftello(file);
            if (offset > 0)
                posix_fadvise(fileno(file), 0, offset, POSIX_FADV_DONTNEED);
        }
    }

    {
        int trailing = fgetc(file);
        if (trailing != EOF || ferror(file)) {
            nsg_set_error(error, error_size,
                          "unexpected data after %d NSG rows", num_vectors);
            goto cleanup;
        }
    }

    *stats = loaded;
    result = 0;

cleanup:
    posix_fadvise(fileno(file), 0, 0, POSIX_FADV_DONTNEED);
    fclose(file);
    return result;
}
