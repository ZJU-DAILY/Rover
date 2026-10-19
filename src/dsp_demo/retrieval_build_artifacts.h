#ifndef RETRIEVAL_BUILD_ARTIFACTS_H
#define RETRIEVAL_BUILD_ARTIFACTS_H
#include "dsp_demo/retrieval_final_refine.h"
#include "dsp_demo/retrieval_raw_reader.h"

int retrieval_build_error4(const char *output,RetrievalRawReader *raw,int n,int dim,
    RetrievalRefineDecode decode,void *provider,const char *base_paths[3],
    const char *raw_or_codebook,int pq);
int retrieval_build_hash(const char *path,unsigned char hash[32]);
#endif
