#ifndef RETRIEVAL_INSERT_STREAM_H
#define RETRIEVAL_INSERT_STREAM_H
#include "retrieval_final_refine.h"
#include <stdint.h>


typedef struct RetrievalInsertStream RetrievalInsertStream;
typedef struct {
    uint64_t raw_rows, old_vector_decodes, current_vector_reads;
} RetrievalInsertStreamStats;
int retrieval_insert_stream_open(const char *raw_path,int n,int dim,
    RetrievalFinalRefineContext *encoded,RetrievalInsertStream **out);

int retrieval_insert_stream_begin(RetrievalInsertStream *stream,int id);
const float *retrieval_insert_stream_query(const RetrievalInsertStream *stream);
int retrieval_insert_stream_read(void *stream,int id,float *out);
void retrieval_insert_stream_stats(const RetrievalInsertStream *stream,
                                   RetrievalInsertStreamStats *out);
void retrieval_insert_stream_close(RetrievalInsertStream *stream);
#endif
