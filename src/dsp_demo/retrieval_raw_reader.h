#ifndef RETRIEVAL_RAW_READER_H
#define RETRIEVAL_RAW_READER_H
#include <stddef.h>
#include <stdint.h>
typedef struct RetrievalRawReader RetrievalRawReader;

int retrieval_raw_open(const char *path,int n,int dim,size_t cache_bytes_per_thread,
                       RetrievalRawReader **out);
int retrieval_raw_read(RetrievalRawReader *reader,int id,float *out);
int retrieval_raw_distances(void *reader,const int32_t *a,const int32_t *b,
                           int count,float *out);
void retrieval_raw_close(RetrievalRawReader *reader);
#endif
