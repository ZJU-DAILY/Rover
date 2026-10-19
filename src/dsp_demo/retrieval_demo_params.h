#ifndef RETRIEVAL_DEMO_PARAMS_H
#define RETRIEVAL_DEMO_PARAMS_H
#include "rover_limits.h"
#ifndef __hexagon__
#include "rover_config.h"
#undef DEMO_GT_K
#define DEMO_GT_K (rover_config.gt_k)
#undef DATA_N_BASE
#define DATA_N_BASE (rover_config.n)
#undef DATA_DIM
#define DATA_DIM (rover_config.dim)
#undef DATA_N_QUERIES
#define DATA_N_QUERIES (rover_config.nq)
#undef DATA_GRAPH_K
#define DATA_GRAPH_K (rover_config.degree)
#undef DATA_BASE_PATH
#define DATA_BASE_PATH (rover_config.base)
#undef DATA_QUERY_PATH
#define DATA_QUERY_PATH (rover_config.query)
#undef DATA_GT_PATH
#define DATA_GT_PATH (rover_config.gt)
#undef DATA_GRAPH_PATH
#define DATA_GRAPH_PATH (rover_config.graph)
#undef DATA_DIR
#define DATA_DIR (rover_config.index_dir)
#undef DATA_LAYOUT_NAME
#define DATA_LAYOUT_NAME (rover_config.layout)
#undef DATA_LAYOUT_FINGERPRINT_HEX
#define DATA_LAYOUT_FINGERPRINT_HEX (rover_config.layout_hash)
#undef DATA_GRAPH_FILENAME
#define DATA_GRAPH_FILENAME (rover_config.graph)
#undef DATA_GRAPH_INDEX_SCHEME_NAME
#define DATA_GRAPH_INDEX_SCHEME_NAME (rover_config.graph_format)
#undef DATA_ORDER_MAP_PATH
#define DATA_ORDER_MAP_PATH (rover_config.order_map)
#undef RETRIEVAL_DB_MODE
#define RETRIEVAL_DB_MODE (rover_config.search_mode)
#undef L1_CODES_CACHE_PATH
#define L1_CODES_CACHE_PATH (rover_config.l1)
#undef RESIDUAL_UFS_PATH
#define RESIDUAL_UFS_PATH (rover_config.residual)
#undef RESIDUAL_QPARAMS_CACHE_PATH
#define RESIDUAL_QPARAMS_CACHE_PATH (rover_config.qparams)
#undef RETRIEVAL_DEMO_L1_BITS
#define RETRIEVAL_DEMO_L1_BITS (rover_config.sq_bits)
#undef RETRIEVAL_DEMO_RESIDUAL_BITS
#define RETRIEVAL_DEMO_RESIDUAL_BITS (rover_config.residual_bits)
#undef RETRIEVAL_DEMO_USE_DISK
#define RETRIEVAL_DEMO_USE_DISK (rover_config.disk)
#undef SEARCH_TOP_K
#define SEARCH_TOP_K (rover_config.top_k)
#undef SEARCH_RERANK_FACTOR
#define SEARCH_RERANK_FACTOR (rover_config.first_factor)
#undef RESIDUAL_CACHE_SLOTS
#define RESIDUAL_CACHE_SLOTS (rover_config.cache_pages)
#undef PURE_PQ_RESIDUAL_CACHE_SLOTS
#define PURE_PQ_RESIDUAL_CACHE_SLOTS (rover_config.cache_pages)
#undef FINAL_REFINE_MODE
#define FINAL_REFINE_MODE (rover_config.refine_mode)
#undef FINAL_REFINE_FACTOR
#define FINAL_REFINE_FACTOR (rover_config.refine_factor)
#undef FINAL_REFINE_CACHE_SLOTS
#define FINAL_REFINE_CACHE_SLOTS (rover_config.refine_cache_pages)
#undef FINAL_REFINE_PATH
#define FINAL_REFINE_PATH (rover_config.error4)
#undef FINAL_REFINE_TRACE_PATH
#define FINAL_REFINE_TRACE_PATH (rover_config.trace_path)
#undef PURE_PQ_CODE_BITS
#define PURE_PQ_CODE_BITS (rover_config.pq_bits)
#undef PURE_PQ_M
#define PURE_PQ_M (rover_config.pq_m)
#undef PURE_PQ_COMPRESSION_X
#define PURE_PQ_COMPRESSION_X (rover_config.pq_compression)
#undef PURE_PQ_RESIDUAL_BITS
#define PURE_PQ_RESIDUAL_BITS (rover_config.residual_bits)
#undef PURE_PQ_CORRECTION_MODE
#define PURE_PQ_CORRECTION_MODE (rover_config.correction)
#undef PURE_PQ_TRAIN_MAX
#define PURE_PQ_TRAIN_MAX (rover_config.train_samples)
#undef PURE_PQ_KMEANS_ITERS
#define PURE_PQ_KMEANS_ITERS (rover_config.train_iters)
#undef PURE_PQ_ARTIFACT_PREFIX
#define PURE_PQ_ARTIFACT_PREFIX (rover_config.pq_prefix)
#undef PURE_PQ_DEVICE_RESULT_DIR
#define PURE_PQ_DEVICE_RESULT_DIR (rover_config.result_dir)
#undef PURE_PQ_DEVICE_RESULT_PATH
#define PURE_PQ_DEVICE_RESULT_PATH (rover_config.metrics)
#undef PURE_PQ_DATASET_NAME
#define PURE_PQ_DATASET_NAME (rover_config.dataset)
#undef RETRIEVAL_RPC_DEMO_QUERY_LIMIT
#define RETRIEVAL_RPC_DEMO_QUERY_LIMIT (rover_config.nq)
#undef ENTRY_POINT_SEED
#define ENTRY_POINT_SEED (rover_config.seed)
#undef PURE_PQ_SQMD_LANES
#define PURE_PQ_SQMD_LANES (rover_config.lanes)
#undef SQMD_CORRECTION_BIAS_PATH
#define SQMD_CORRECTION_BIAS_PATH (rover_config.bias)
#undef RETRIEVAL_DEMO_ENTRY_SWEEP_VALUES
#define RETRIEVAL_DEMO_ENTRY_SWEEP_VALUES {rover_config.entry_points}
#undef RETRIEVAL_DEMO_POOL_SWEEP_VALUES
#define RETRIEVAL_DEMO_POOL_SWEEP_VALUES {rover_config.pool}
#undef FINAL_REFINE_CANDIDATES
#define FINAL_REFINE_CANDIDATES ((rover_config.refine_all ? rover_config.pool : rover_config.refine_factor * rover_config.top_k))
#undef FINAL_REFINE_SHA256_BYTES
#define FINAL_REFINE_SHA256_BYTES (rover_config.error4_hash)
#undef PURE_PQ_CORRECTION_TAG
#define PURE_PQ_CORRECTION_TAG ((rover_config.correction ? "online" : "off"))
#else
extern int rover_dsp_l1_bits;
#undef RETRIEVAL_DEMO_L1_BITS
#define RETRIEVAL_DEMO_L1_BITS rover_dsp_l1_bits
#endif
#endif
