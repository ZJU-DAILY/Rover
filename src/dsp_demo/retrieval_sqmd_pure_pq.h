#ifndef RETRIEVAL_SQMD_PURE_PQ_H
#define RETRIEVAL_SQMD_PURE_PQ_H
#include "dsp_demo/retrieval_graph_reader.h"

#include <stdint.h>
#include <stddef.h>
#ifdef RETRIEVAL_ENABLE_FULL_BUILD
#include "dsp_demo/retrieval_build_pause.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PurePqSearchIndexView {
    const int32_t *graph;
    int graph_degree;
    int num_vectors;
    const uint8_t *const *code_chunks;
    const int *chunk_vector_counts;
    int chunk_count;
    int chunk_vectors;
    const float *codebook;
    int dim;
    int m;
    int ks;
    int subdim;
} PurePqSearchIndexView;

typedef struct PurePqSearchRequest {
    const float *query;
    const int *entry_ids;
    int entry_count;
    int expand_count;
    int pool_capacity;
    int num_out;
} PurePqSearchRequest;

typedef struct PurePqSearchStats {
    uint64_t pq_scored_candidates;
    uint64_t corrected_candidates;
    uint64_t expanded_nodes;
    uint64_t visited_dedup_skips;
} PurePqSearchStats;

typedef enum PurePqCorrectionMode {
    PURE_PQ_CORRECTION_OFF = 0,
    PURE_PQ_CORRECTION_ONLINE = 1
} PurePqCorrectionMode;

typedef int (*PurePqDistanceEvaluatorFn)(void *context,
                                         int vector_id,
                                         float coarse_distance,
                                         float *out_distance);

typedef struct PurePqDistancePolicy {
    PurePqCorrectionMode mode;
    PurePqDistanceEvaluatorFn evaluator;
    void *context;
} PurePqDistancePolicy;

#define PURE_PQ_SQMD_MAX_LANES 8
#define PURE_PQ_MAX_CODE_CHUNKS 64
#define PURE_PQ_SQMD_BUFFERS_PER_LANE 2
#define PURE_PQ_SQMD_MAX_WORK_ITEMS 32
#define PURE_PQ_SQMD_CACHE_LINE_BYTES 128u
#define PURE_PQ_SQMD_ALIGN_UP(value, alignment) \
    ((((size_t)(value) + (size_t)(alignment) - 1u) / \
      (size_t)(alignment)) * (size_t)(alignment))

typedef enum PurePqSqmdBufferState {
    PURE_PQ_SQMD_BUFFER_EMPTY = 0,
    PURE_PQ_SQMD_BUFFER_CPU_READY = 1,
    PURE_PQ_SQMD_BUFFER_DSP_DONE = 2
} PurePqSqmdBufferState;

typedef enum PurePqSqmdControlState {
    PURE_PQ_SQMD_CONTROL_IDLE = 0,
    PURE_PQ_SQMD_CONTROL_RUNNING = 1,
    PURE_PQ_SQMD_CONTROL_DONE = 2,
    PURE_PQ_SQMD_CONTROL_SHUTDOWN = 3,
    PURE_PQ_SQMD_CONTROL_ABORT = 4
} PurePqSqmdControlState;

typedef struct PurePqSqmdWorkItem {
    int32_t neighbor_id;
    int32_t source_id;
} PurePqSqmdWorkItem;

typedef struct PurePqSqmdResultItem {
    int32_t neighbor_id;
    int32_t source_id;
    float distance;
} PurePqSqmdResultItem;

typedef struct __attribute__((aligned(PURE_PQ_SQMD_CACHE_LINE_BYTES)))
PurePqSqmdLaneBuffer {
    volatile int32_t state;
    int32_t turn_id;
    int32_t query_gen;
    int32_t work_count;
    int32_t result_count;
    PurePqSqmdWorkItem work[PURE_PQ_SQMD_MAX_WORK_ITEMS];
    PurePqSqmdResultItem result[PURE_PQ_SQMD_MAX_WORK_ITEMS];
} PurePqSqmdLaneBuffer;

typedef struct PurePqSqmdLane {
    PurePqSqmdLaneBuffer buffers[PURE_PQ_SQMD_BUFFERS_PER_LANE];
    volatile int32_t dsp_active_buffer;
#ifdef RETRIEVAL_ENABLE_FULL_BUILD
    volatile int32_t cpu_fill_buffer __attribute__((aligned(PURE_PQ_SQMD_CACHE_LINE_BYTES)));
#else
    volatile int32_t cpu_fill_buffer;
#endif
} PurePqSqmdLane;
#ifdef RETRIEVAL_ENABLE_FULL_BUILD
_Static_assert(offsetof(PurePqSqmdLane,cpu_fill_buffer) - offsetof(PurePqSqmdLane,dsp_active_buffer) >= PURE_PQ_SQMD_CACHE_LINE_BYTES,
               "CPU FIFO cursor must not share a DSP-written cache line");
#endif

typedef struct __attribute__((aligned(PURE_PQ_SQMD_CACHE_LINE_BYTES)))
PurePqQueryLutHeader {
    volatile int32_t query_gen;
    volatile int32_t lut_gen;
    int32_t m;
    int32_t ks;
    int32_t lut_count;
    uint8_t index_uuid[16];
    uint8_t reserved[PURE_PQ_SQMD_CACHE_LINE_BYTES - 5u * sizeof(int32_t) - 16u];
} PurePqQueryLutHeader;

typedef struct PurePqSqmdStats {
    volatile uint64_t work_items;
    volatile uint64_t dsp_compute_us;
    volatile uint64_t dsp_wait_cpu_us;
    volatile uint64_t dsp_wait_polls;
    volatile uint64_t abort_count;
} PurePqSqmdStats;

typedef struct PurePqSqmdSharedControl {
    volatile int32_t control_state;
    int32_t lane_count;
    int32_t turn_id;
    volatile int32_t query_gen;
    uint8_t reserved[PURE_PQ_SQMD_CACHE_LINE_BYTES - 4u * sizeof(int32_t)];
    PurePqSqmdLane lanes[PURE_PQ_SQMD_MAX_LANES];
    PurePqSqmdStats stats;
#ifdef RETRIEVAL_ENABLE_FULL_BUILD
    RetrievalBuildPause build_pause;
#endif
} PurePqSqmdSharedControl;

typedef struct PurePqSqmdAdcContext {
    const uint8_t *const *code_chunks;
    const int *chunk_vector_counts;
    int chunk_count;
    int chunk_vectors;
    int num_vectors;
    int m;
    int ks;
    const uint8_t *index_uuid;
    const PurePqQueryLutHeader *lut_header;
    const float *lut;
    int use_hvx;
    void *hvx_gather_buffer;
} PurePqSqmdAdcContext;

typedef struct PurePqCpuSearchState PurePqCpuSearchState;

typedef struct PurePqSqmdDispatchSummary {
    int source_id;
    int work_items;
    int dedup_skips;
    int buffers_submitted;
    int empty_turn;
} PurePqSqmdDispatchSummary;

#define PURE_PQ_SQMD_LANE_BUFFER_RAW_BYTES \
    (5u * sizeof(int32_t) + \
     (size_t)PURE_PQ_SQMD_MAX_WORK_ITEMS * \
         (sizeof(PurePqSqmdWorkItem) + sizeof(PurePqSqmdResultItem)))
#define PURE_PQ_SQMD_LANE_BUFFER_BYTES \
    PURE_PQ_SQMD_ALIGN_UP(PURE_PQ_SQMD_LANE_BUFFER_RAW_BYTES, \
                          PURE_PQ_SQMD_CACHE_LINE_BYTES)

_Static_assert(sizeof(PurePqSqmdWorkItem) == 8u,
               "Pure-PQ SQMD work layout changed");
_Static_assert(sizeof(PurePqSqmdResultItem) == 12u,
               "Pure-PQ SQMD result layout changed");
_Static_assert(sizeof(PurePqSqmdLaneBuffer) ==
                   PURE_PQ_SQMD_LANE_BUFFER_BYTES,
               "Pure-PQ SQMD lane buffer must be cache-line isolated");
_Static_assert(sizeof(PurePqQueryLutHeader) ==
                   PURE_PQ_SQMD_CACHE_LINE_BYTES,
               "Pure-PQ LUT header must occupy one cache line");

void pure_pq_sqmd_control_init(PurePqSqmdSharedControl *control,
                               int lane_count);


int pure_pq_sqmd_process_buffer_payload(PurePqSqmdLaneBuffer *buffer,
                                        int expected_turn_id,
                                        int expected_query_gen,
                                        const PurePqSqmdAdcContext *context);

int pure_pq_sqmd_process_buffer(PurePqSqmdLaneBuffer *buffer,
                                int expected_turn_id,
                                int expected_query_gen,
                                const PurePqSqmdAdcContext *context);

int pure_pq_cpu_search_reset(PurePqCpuSearchState *,int,int,int);

int pure_pq_cpu_search_create(int num_vectors,
                              int pool_capacity,
                              int expand_count,
                              PurePqCpuSearchState **out_state);

int pure_pq_cpu_search_seed(PurePqCpuSearchState *state,
                            const int *ids,
                            const float *distances,
                            int count);

int pure_pq_cpu_search_set_graph_reader(PurePqCpuSearchState *state,
                                       const RetrievalGraphReader *reader);

int pure_pq_cpu_search_set_distance_policy(
    PurePqCpuSearchState *state,
    const PurePqDistancePolicy *policy);

int pure_pq_cpu_search_dispatch(PurePqCpuSearchState *state,
                                const int32_t *graph,
                                int graph_degree,
                                PurePqSqmdSharedControl *control,
                                int turn_id,
                                int query_gen,
                                PurePqSqmdDispatchSummary *summary);

int pure_pq_cpu_search_collect(PurePqCpuSearchState *state,
                               PurePqSqmdSharedControl *control,
                               int turn_id,
                               int query_gen);

typedef struct PurePqSourcePipeline {
    int window, next_turn, pending, head, high_water;
    int turns[PURE_PQ_SQMD_BUFFERS_PER_LANE];
    uint64_t jobs, retired, work_items;
} PurePqSourcePipeline;

int pure_pq_source_pipeline_init(PurePqSourcePipeline *pipeline, int window);

int pure_pq_source_pipeline_step(PurePqSourcePipeline *pipeline,
                                 PurePqCpuSearchState *state,
                                 const int32_t *graph, int graph_degree,
                                 PurePqSqmdSharedControl *control,
                                 int query_gen);

int pure_pq_cpu_search_finish(PurePqCpuSearchState *state,
                              int *out_ids,
                              float *out_distances,
                              int out_capacity);

void pure_pq_cpu_search_destroy(PurePqCpuSearchState *state);

int retrieval_pure_pq_search_scalar(const PurePqSearchIndexView *index,
                                    const PurePqSearchRequest *request,
                                    int *out_ids,
                                    float *out_distances,
                                    int out_capacity,
                                    PurePqSearchStats *out_stats);

int retrieval_pure_pq_search_scalar_with_policy(
    const PurePqSearchIndexView *index,
    const PurePqSearchRequest *request,
    const PurePqDistancePolicy *policy,
    int *out_ids,
    float *out_distances,
    int out_capacity,
    PurePqSearchStats *out_stats);

#ifdef __cplusplus
}
#endif

#endif
