#ifndef RETRIEVAL_NSG_LOADER_H
#define RETRIEVAL_NSG_LOADER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RetrievalNsgLoadStats {
    uint32_t file_width;
    int32_t entry_point;
    uint64_t truncated_nodes;
    uint64_t truncated_edges;
} RetrievalNsgLoadStats;


int retrieval_nsg_load_fixed(const char *path,
                             int num_vectors,
                             int runtime_width,
                             int32_t *graph,
                             RetrievalNsgLoadStats *stats,
                             char *error,
                             size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
