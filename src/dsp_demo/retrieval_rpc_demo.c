#include "dsp_demo/retrieval_rpc_demo.h"

#include "calculator.h"

#include "dsp_demo/retrieval_async_demo_common.h"
#include "dsp_demo/retrieval_demo_params.h"
#include "dsp_demo/retrieval_l1_int8_core.h"
#include "dsp_demo/retrieval_pq_core.h"
#include "dsp_demo/retrieval_db_common.h"
#include "dsp_demo/retrieval_demo_mode.h"
#include "dsp_demo/retrieval_demo_internal_types.h"
#include "dsp_demo/retrieval_sqmd_common.h"
#include "dsp_demo/retrieval_sqmd_pure_pq.h"
#include "dsp_demo/retrieval_pure_pq_index.h"
#include "dsp_demo/retrieval_nsg_loader.h"
#include "dsp_demo/retrieval_sqmd_seed.h"
#include "dsp_demo/retrieval_final_refine.h"
#include "dsp_demo/retrieval_build_artifacts.h"
#include <sys/resource.h>

#include <float.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sched.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <errno.h>

#include "rpcmem.h"
#pragma weak fastrpc_mmap
#pragma weak fastrpc_munmap

typedef struct DemoResult {
    int recall1_hits;
    int recall10_hits;
    int recall100_hits;
    unsigned long long phase1_us_sum;
    unsigned long long phase2_us_sum;
    unsigned long long total_us_sum;
    unsigned long long cpu_wait_us;
    unsigned long long dsp_invalidate_us;
    unsigned long long dsp_wait_poll_count;
    unsigned long long sqmd_work_items;
    unsigned long long sqmd_dedup_skips;
    unsigned long long sqmd_empty_turns;
    unsigned long long sqmd_dsp_compute_us;
    unsigned long long sqmd_cpu_correct_us;
    unsigned long long sqmd_cpu_wait_dsp_us;
    unsigned long long sqmd_dsp_wait_cpu_us;
    unsigned long long sqmd_dsp_invalidate_us;
    unsigned long long sqmd_lane_wait_us;
    unsigned long long sqmd_dsp_done_batches;
    unsigned long long sqmd_dsp_wait_poll_count;
    unsigned long long sqmd_abort_count;
    unsigned long long sqmd_source_jobs_created;
    unsigned long long sqmd_source_jobs_retired;
    unsigned long long sqmd_speculative_sources;
    unsigned long long sqmd_source_window_starvation;
    unsigned long long sqmd_inflight_sources_highwater;
    unsigned long long split_queries;
    unsigned long long split_candidates;
    unsigned long long query_u8_clipped_low;
    unsigned long long query_u8_clipped_high;
    double query_u8_error_sq;
    double query_u8_norm_sq;
} DemoResult;

typedef struct RetrievalSlot {
    int32_t *candidate_ids;
    float *candidate_dists;
    int32_t actual_count;
    const float *query;
    const int32_t *gt_row;
    int query_idx;
    int config_idx;
    ResidualBlockCache *res_cache;
    RetrievalFinalRefineContext *refinement;
} RetrievalSlot;

typedef enum RetrievalSlotState {
    RETRIEVAL_SLOT_FREE = 0,
    RETRIEVAL_SLOT_FILLING,
    RETRIEVAL_SLOT_READY,
    RETRIEVAL_SLOT_PROCESSING
} RetrievalSlotState;

typedef struct RetrievalPipeline {
    RetrievalSlot *slots;
    pthread_mutex_t mu;
    pthread_cond_t cond_ready;
    pthread_cond_t cond_free;
    int *state;
    int slot_count;
    int shutdown;
    int ret;
    int rerank_count;
    DemoResult *results;
    const RetrievalDemoConfig *configs;
} RetrievalPipeline;

typedef struct RetrievalRpcWorkerArgs {
    RetrievalPipeline *pipe;
    remote_handle64 handle;
    const RetrievalDemoConfig *config;
    int num_out;
    int slot_idx;
    int started;
    int ret;
} RetrievalRpcWorkerArgs;

typedef struct ResidualBlockCache ResidualBlockCache;

typedef struct DbCorrectionCtx {
    ExpansionBuffer *buf;
    volatile int     stop;
    const float *query_a;
    const float *query_b;
    const RetrievalDemoEncodedDataset *dataset;
    ResidualBlockCache *res_cache_a;
    ResidualBlockCache *res_cache_b;
    uint64_t corr_batches;
    uint64_t corr_candidates;
} DbCorrectionCtx;

typedef struct DbSharedBuffer {
    DbSharedLayout *layout;
    ExpansionBuffer *buf;
    DbSharedStats *stats;
    int uses_rpcmem;
    int fd;
    int mapped;
} DbSharedBuffer;

typedef struct SqmdSharedBuffer {
    SqmdSharedControl *ctrl;
    int uses_rpcmem;
    int fd;
    int mapped;
} SqmdSharedBuffer;

typedef struct RetrievalRpcDbWorkerArgs {
    RetrievalPipeline *pipe;
    remote_handle64    handle;
    const RetrievalDemoConfig *config;
    DbSharedBuffer    *db_shared;
    int num_out;
    int slot_idx_a;
    int slot_idx_b;
    int started;
    int ret;
} RetrievalRpcDbWorkerArgs;

typedef struct RetrievalRpcSqmdWorkerArgs {
    remote_handle64 handle;
    const float *query;
    SqmdSharedBuffer *sqmd_shared;
    int lane_count;
    int work_items_cap;
    int started;
    volatile int done;
    int ret;
} RetrievalRpcSqmdWorkerArgs;


typedef struct RetrievalRpcSqmdSessionArgs {
    remote_handle64 handle;
    SqmdSharedBuffer *sqmd_shared;
    int lane_count;
    int work_items_cap;
    int started;
    int query_gen;
    volatile int done;
    int ret;
} RetrievalRpcSqmdSessionArgs;

static RetrievalDemoEncodedDataset g_dataset;
static int32_t *g_graph = NULL;
static float *g_queries = NULL;
static int32_t *g_gt = NULL;
static int32_t g_nsg_entry_point = -1;
static int g_graph_uses_rpcmem = 0;
static int g_dma_graph_fd      = -1;

static int g_dma_l1_fds[MAX_L1_CHUNKS];
static int g_dma_l1_mapped_count = 0;
static int g_dma_graph_mapped  = 0;
static int g_dma_l1_mapped     = 0;
static int g_dma_domain_id     = -1;


static DbSharedBuffer   g_db_shared[DEMO_RPC_MAX_CONCURRENT];
static SqmdSharedBuffer g_sqmd_shared;
static float *g_sqmd_correction_bias = NULL;

static int32_t *g_reorder_map  = NULL;
static int32_t *g_reverse_map  = NULL;

#include "retrieval_rpc_demo_support.c"
#include "retrieval_rpc_demo_reorder.c"
#include "retrieval_rpc_demo_local_search.c"
#include "retrieval_rpc_demo_graph_index.c"
#include "retrieval_demo_mode.c"
#include "retrieval_sqmd_common.c"
#ifdef RETRIEVAL_ENABLE_FULL_BUILD
int rover_run_build(remote_handle64 handle,int domain_id);
#endif
#include "retrieval_rpc_demo_pipeline.c"
#include "retrieval_rpc_demo_pure_pq.c"

#ifdef RETRIEVAL_ENABLE_FULL_BUILD
#include "retrieval_rpc_demo_full_build.c"
#endif
