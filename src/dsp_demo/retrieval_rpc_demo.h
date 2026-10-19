#ifndef RETRIEVAL_RPC_DEMO_H
#define RETRIEVAL_RPC_DEMO_H

#include <stdint.h>

#include "remote.h"

#include "imp/quantize_compress.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ResidualBlockCache ResidualBlockCache;


#ifndef MAX_L1_CHUNKS
#define MAX_L1_CHUNKS 16
#endif

typedef struct RetrievalDemoEncodedDataset {
    int num_vectors;
    int dim;
    int l1_bits;
    int l1_code_bytes;
    QCQuantParams l1_qparams;

    uint8_t *l1_codes;
    int l1_codes_uses_rpcmem;
    uint8_t *l1_chunks[MAX_L1_CHUNKS];
    int l1_chunk_fds[MAX_L1_CHUNKS];
    int l1_chunk_uses_rpcmem[MAX_L1_CHUNKS];
    int l1_chunk_count;
    int l1_chunk_vectors;
    QCResidualQuantParams *residual_qparams;
    uint8_t *residual_codes;
    int residual_bits;
    int residual_code_bytes;
} RetrievalDemoEncodedDataset;

typedef struct RetrievalDemoConfig {
    int entry_points;
    int expand_count;
    int pool_size;
    int top_k;
    int rerank_factor;
} RetrievalDemoConfig;

int retrieval_demo_encode_dataset_from_memory(const float *data,
                                              int num_vectors,
                                              int dim,
                                              RetrievalDemoEncodedDataset *dataset);
int retrieval_demo_encode_dataset_from_fvecs_streaming(const char *path,
                                                       int num_vectors,
                                                       int dim,
                                                       RetrievalDemoEncodedDataset *dataset);
void retrieval_demo_free_encoded_dataset(RetrievalDemoEncodedDataset *dataset);


int retrieval_demo_encode_dataset_streaming_to_disk(
    const char *fvecs_path,
    int num_vectors,
    int dim,
    const char *residual_path,
    const char *qparams_path,
    const int32_t *reverse_map,
    RetrievalDemoEncodedDataset *dataset);


int retrieval_demo_load_qparams_from_cache(
    const char *path,
    int num_vectors,
    int residual_bits,
    RetrievalDemoEncodedDataset *dataset);


int retrieval_demo_load_l1_and_qparams_from_disk(
    const char *l1_path,
    const char *qparams_path,
    int num_vectors,
    int dim,
    int residual_bits,
    RetrievalDemoEncodedDataset *dataset);


int retrieval_demo_load_residual_codes_from_block_file(
    const char *path,
    RetrievalDemoEncodedDataset *dataset);
float retrieval_demo_exact_distance_residual(const float *query,
                                             const RetrievalDemoEncodedDataset *dataset,
                                             int vec_idx);
float retrieval_demo_exact_distance_residual_cached(
    const float *query,
    const RetrievalDemoEncodedDataset *dataset,
    int vec_idx,
    ResidualBlockCache *cache);
int retrieval_demo_search_local(const RetrievalDemoEncodedDataset *dataset,
                                const int32_t *graph,
                                int graph_degree,
                                const float *query,
                                const RetrievalDemoConfig *config,
                                unsigned int seed,
                                int *result_ids,
                                float *result_dists);


int run_retrieval_search_rpc_demo(remote_handle64 handle, int domain_id);


int run_retrieval_search_rpc_demo_db(remote_handle64 handle, int domain_id);


int run_retrieval_search_rpc_demo_selected(remote_handle64 handle,
                                           int domain_id,
                                           const char *db_mode_env);


int run_retrieval_search_rpc_demo_sqmd(remote_handle64 handle, int domain_id);
int run_retrieval_search_rpc_demo_pure_pq(remote_handle64 handle,
                                          int domain_id);

#ifdef __cplusplus
}
#endif

#endif

int rover_run_build(remote_handle64 handle,int domain_id);
