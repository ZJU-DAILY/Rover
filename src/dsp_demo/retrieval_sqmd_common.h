#ifndef RETRIEVAL_SQMD_COMMON_H
#define RETRIEVAL_SQMD_COMMON_H

#include <stdint.h>

#include "dsp_demo/retrieval_db_common.h"
#include "dsp_demo/retrieval_demo_internal_types.h"
#include "dsp_demo/retrieval_graph_reader.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RetrievalDemoEncodedDataset RetrievalDemoEncodedDataset;
typedef struct ResidualBlockCache ResidualBlockCache;

typedef struct sqmdCpuSearchState {
    DemoCandidate *pool;
    int pool_size;
    int pool_capacity;

    DemoFrontierEntry *frontier;
    int frontier_size;
    int frontier_capacity;

    uint32_t *visited;
    int visited_words;

    const float *query;
    const RetrievalDemoEncodedDataset *dataset;
    ResidualBlockCache *res_cache;
    RetrievalGraphReader graph_reader;

    int entry_points;
    int expand_count;
    int num_out;
    int num_vectors;
    int finished;

    uint64_t work_items;
    uint64_t dedup_skips;
    uint64_t exact_corrected;
    uint64_t empty_turns;
    uint64_t lane_wait_us;
    uint64_t dsp_done_batches;

    uint64_t source_jobs_created;
    uint64_t source_jobs_retired;
    uint64_t speculative_sources;
    uint64_t source_window_starvation;
    uint32_t inflight_sources_highwater;

    int abort_stage;
    int abort_rc;
    int abort_turn;
    int abort_lane;
    int abort_buffer;
    int abort_item;
    int abort_state;
    int abort_buf_turn;
    int abort_work_count;
    int abort_result_count;
    int abort_neighbor_id;
    int abort_source_id;
} sqmdCpuSearchState;

typedef enum SqmdStatus {
    SQMD_STATUS_OK = 0,
    SQMD_STATUS_BAD_PARAM = -1,
    SQMD_STATUS_NO_EMPTY_BUFFER = -2,
    SQMD_STATUS_TOO_MUCH_WORK = -3,
    SQMD_STATUS_TURN_INCOMPLETE = -4,
    SQMD_STATUS_TURN_TIMEOUT = -5
} SqmdStatus;

typedef struct SqmdDispatchSummary {
    int work_items;
    int dedup_skips;
    int invalid_skips;
    int buffers_submitted;
    int empty_turn;
} SqmdDispatchSummary;

typedef int (*SqmdDistanceFn)(const SqmdWorkItem *work,
                              float *coarse_dist,
                              void *ctx);
typedef float (*SqmdCorrectionFn)(int neighbor_id,
                                  int source_id,
                                  float coarse_dist,
                                  void *ctx);
typedef float (*SqmdResidualDeltaFn)(int neighbor_id,
                                     int source_id,
                                     void *ctx);
typedef void (*SqmdWaitFn)(void *ctx);
typedef uint64_t (*SqmdTimeFn)(void *ctx);

void sqmd_shared_control_init(SqmdSharedControl *ctrl, int lane_count);
void sqmd_shared_control_init_lanes(SqmdLane *lanes, int lane_count);

int sqmd_lane_find_empty_buffer(SqmdLane *lane);
int sqmd_process_lane_buffer(SqmdLaneBuffer *buf,
                             int expected_turn_id,
                             int num_vectors,
                             SqmdDistanceFn distance_fn,
                             void *ctx);
int sqmd_process_lane_buffer_with_cap(SqmdLaneBuffer *buf,
                                      int expected_turn_id,
                                      int num_vectors,
                                      int work_items_cap,
                                      SqmdDistanceFn distance_fn,
                                      void *ctx);
int sqmd_process_lane_buffer_payload_with_cap(SqmdLaneBuffer *buf,
                                              int expected_turn_id,
                                              int num_vectors,
                                              int work_items_cap,
                                              SqmdDistanceFn distance_fn,
                                              void *ctx);

int sqmd_dispatch_neighbors_to_lanes(const int32_t *neighbors,
                                     int neighbor_count,
                                     int num_vectors,
                                     int source_id,
                                     uint32_t *visited,
                                     SqmdLane *lanes,
                                     int lane_count,
                                     int turn_id,
                                     SqmdDispatchSummary *summary);

int sqmd_cpu_state_init(sqmdCpuSearchState *st,
                        int num_vectors,
                        int pool_capacity,
                        int entry_points,
                        int expand_count,
                        const float *query,
                        const RetrievalDemoEncodedDataset *dataset,
                        ResidualBlockCache *res_cache);
void sqmd_cpu_state_cleanup(sqmdCpuSearchState *st);
int sqmd_cpu_state_seed(sqmdCpuSearchState *st,
                        const int *seed_ids,
                        const float *seed_dists,
                        int seed_count);
int sqmd_cpu_state_pop_source(sqmdCpuSearchState *st, int *source_id);
int sqmd_cpu_state_apply_results(sqmdCpuSearchState *st,
                                 SqmdLaneBuffer *buf,
                                 int expected_turn_id);
int sqmd_cpu_state_apply_results_with_correction(sqmdCpuSearchState *st,
                                                 SqmdLaneBuffer *buf,
                                                 int expected_turn_id,
                                                 SqmdCorrectionFn correction_fn,
                                                 void *ctx);
int sqmd_cpu_state_dispatch_next_turn(sqmdCpuSearchState *st,
                                      const int32_t *graph,
                                      int graph_degree,
                                      int num_vectors,
                                      SqmdLane *lanes,
                                      int lane_count,
                                      int turn_id,
                                      SqmdDispatchSummary *summary,
                                      int *source_id);
int sqmd_shared_control_dispatch_next_turn(sqmdCpuSearchState *st,
                                           SqmdSharedControl *ctrl,
                                           const int32_t *graph,
                                           int graph_degree,
                                           int num_vectors,
                                           int turn_id,
                                           SqmdDispatchSummary *summary,
                                           int *source_id);
int sqmd_shared_control_collect_turn(sqmdCpuSearchState *st,
                                     SqmdSharedControl *ctrl,
                                     int expected_turn_id,
                                     SqmdCorrectionFn correction_fn,
                                     void *ctx);
int sqmd_shared_control_collect_turn_timed(sqmdCpuSearchState *st,
                                           SqmdSharedControl *ctrl,
                                           int expected_turn_id,
                                           SqmdCorrectionFn correction_fn,
                                           void *correction_ctx,
                                           SqmdTimeFn time_fn,
                                           void *time_ctx);
int sqmd_cpu_state_run_shared_control(sqmdCpuSearchState *st,
                                      SqmdSharedControl *ctrl,
                                      const int32_t *graph,
                                      int graph_degree,
                                      int num_vectors,
                                      SqmdCorrectionFn correction_fn,
                                      void *correction_ctx,
                                      SqmdWaitFn wait_fn,
                                      void *wait_ctx,
                                      SqmdTimeFn time_fn,
                                      void *time_ctx,
                                      int *out_ids,
                                      float *out_dists,
                                      int out_cap);
int sqmd_cpu_state_run_source_pipeline(sqmdCpuSearchState *st,
                                       SqmdSharedControl *ctrl,
                                       int max_inflight_sources,
                                       const int32_t *graph,
                                       int graph_degree,
                                       int num_vectors,
                                       SqmdCorrectionFn correction_fn,
                                       void *correction_ctx,
                                       SqmdWaitFn wait_fn,
                                       void *wait_ctx,
                                       SqmdTimeFn time_fn,
                                       void *time_ctx,
                                       int *out_ids,
                                       float *out_dists,
                                       int out_cap);
int sqmd_cpu_state_run_source_pipeline_split(
    sqmdCpuSearchState *st,
    SqmdSharedControl *ctrl,
    int max_inflight_sources,
    const int32_t *graph,
    int graph_degree,
    int num_vectors,
    SqmdResidualDeltaFn residual_delta_fn,
    void *residual_delta_ctx,
    SqmdWaitFn wait_fn,
    void *wait_ctx,
    SqmdTimeFn time_fn,
    void *time_ctx,
    int *out_ids,
    float *out_dists,
    int out_cap);
int sqmd_cpu_state_run_fake_dsp(const int32_t *graph,
                                int graph_degree,
                                int num_vectors,
                                int pool_capacity,
                                int entry_points,
                                int expand_count,
                                const int *seed_ids,
                                const float *seed_dists,
                                int seed_count,
                                int lane_count,
                                SqmdDistanceFn distance_fn,
                                void *distance_ctx,
                                SqmdCorrectionFn correction_fn,
                                void *correction_ctx,
                                int *out_ids,
                                float *out_dists,
                                int out_cap);

#ifdef __cplusplus
}
#endif

#endif
