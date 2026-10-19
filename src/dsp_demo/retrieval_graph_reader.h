#ifndef RETRIEVAL_GRAPH_READER_H
#define RETRIEVAL_GRAPH_READER_H
#include <stdint.h>


typedef struct RetrievalGraphReader {
    int (*read)(void *ctx, int source, int32_t *neighbors, int count);
    void *ctx;
} RetrievalGraphReader;
#endif
