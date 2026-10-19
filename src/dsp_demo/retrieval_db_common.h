#ifndef RETRIEVAL_DB_COMMON_H
#define RETRIEVAL_DB_COMMON_H

#include <stdint.h>
#include <stddef.h>
#ifdef RETRIEVAL_ENABLE_FULL_BUILD
#include "dsp_demo/retrieval_build_pause.h"
#endif
#include "dsp_demo/retrieval_demo_params.h"


#define DB_MAX_EXPAND  SEARCH_EXPAND_COUNT

typedef enum DbBufState {
    DB_STATE_EMPTY    = 0,
    DB_STATE_DSP_DONE = 1,
    DB_STATE_CPU_DONE = 2
} DbBufState;

typedef struct ExpansionBuffer {
    volatile int32_t state;
    int32_t  count;
    int32_t  finished;
    int32_t  _pad;
    int32_t  neighbor_ids[DB_MAX_EXPAND];
    float    coarse_dists[DB_MAX_EXPAND];
    float    corrected_dists[DB_MAX_EXPAND];
} ExpansionBuffer;

#define DB_EXPANSION_BUFFER_EXPECTED_BYTES \
    ((size_t)(DB_MAX_EXPAND) * (sizeof(int32_t) + 2u * sizeof(float)) + \
     4u * sizeof(int32_t))

_Static_assert(offsetof(ExpansionBuffer, state) == 0u,
               "ExpansionBuffer.state must be at offset 0");
_Static_assert(offsetof(ExpansionBuffer, count) == sizeof(int32_t),
               "ExpansionBuffer.count must follow state");
_Static_assert(offsetof(ExpansionBuffer, finished) == 2u * sizeof(int32_t),
               "ExpansionBuffer.finished must follow count");
_Static_assert(sizeof(ExpansionBuffer) == DB_EXPANSION_BUFFER_EXPECTED_BYTES,
               "ExpansionBuffer size changed unexpectedly");

typedef struct DbSharedStats {
    volatile uint64_t dsp_wait_cpu_done_us;
    volatile uint64_t dsp_invalidate_us;
    volatile uint64_t dsp_wait_poll_count;
} DbSharedStats;

typedef struct DbSharedLayout {
    ExpansionBuffer buf[2];
    DbSharedStats   stats;
} DbSharedLayout;

typedef enum SqmdBufState {
    SQMD_BUF_EMPTY     = 0,
    SQMD_BUF_CPU_READY = 1,
    SQMD_BUF_DSP_DONE  = 2,
    SQMD_BUF_DSP_ABORT = 3
} SqmdBufState;

typedef enum SqmdCtrlState {
    SQMD_CTRL_RUNNING = 0,
    SQMD_CTRL_DONE    = 1,
    SQMD_CTRL_ABORT   = 2,

    SQMD_CTRL_PREPARING = 3
} SqmdCtrlState;

typedef enum SqmdAbortStage {
    SQMD_ABORT_NONE = 0,
    SQMD_ABORT_DISPATCH = 1,
    SQMD_ABORT_COLLECT = 2,
    SQMD_ABORT_CTRL = 3,
    SQMD_ABORT_APPLY_HEADER = 4,
    SQMD_ABORT_APPLY_RESULT = 5,
    SQMD_ABORT_DSP_PROCESS = 6,
    SQMD_ABORT_TURN_WAIT_TIMEOUT = 7
} SqmdAbortStage;


typedef enum SqmdSessionState {
    SQMD_SESSION_IDLE     = 0,
    SQMD_SESSION_RUNNING  = 1,
    SQMD_SESSION_SHUTDOWN = 2,
    SQMD_SESSION_ABORT    = 3
} SqmdSessionState;

typedef struct SqmdWorkItem {
    int32_t neighbor_id;
    int32_t source_id;
} SqmdWorkItem;

typedef struct SqmdResultItem {
    int32_t neighbor_id;
    int32_t source_id;
    float coarse_dist;
} SqmdResultItem;


#define SQMD_CACHE_LINE_BYTES 128u
#define SQMD_ALIGN_UP_BYTES(n, a) \
    ((((size_t)(n) + (size_t)(a) - 1u) / (size_t)(a)) * (size_t)(a))

typedef struct __attribute__((aligned(SQMD_CACHE_LINE_BYTES))) SqmdLaneBuffer {
    volatile int32_t state;
    int32_t turn_id;
    int32_t work_count;
    int32_t result_count;
    SqmdWorkItem work[SQMD_MAX_WORK_ITEMS];
    SqmdResultItem result[SQMD_MAX_WORK_ITEMS];
} SqmdLaneBuffer;

typedef struct SqmdLane {
    SqmdLaneBuffer buf[SQMD_BUFFERS_PER_LANE];
    volatile int32_t dsp_active_buf;

    volatile int32_t cpu_fill_buf __attribute__((aligned(SQMD_CACHE_LINE_BYTES)));
} SqmdLane;

typedef struct SqmdSharedStats {
    volatile uint64_t sqmd_work_items;
    volatile uint64_t sqmd_dedup_skips;
    volatile uint64_t sqmd_empty_turns;
    volatile uint64_t sqmd_dsp_compute_us;
    volatile uint64_t sqmd_cpu_correct_us;
    volatile uint64_t sqmd_cpu_wait_dsp_us;
    volatile uint64_t sqmd_dsp_wait_cpu_us;
    volatile uint64_t sqmd_dsp_invalidate_us;
    volatile uint64_t sqmd_lane_wait_us;
    volatile uint64_t sqmd_dsp_done_batches;
    volatile uint64_t sqmd_dsp_wait_poll_count;
    volatile uint64_t sqmd_abort_count;
    volatile int32_t sqmd_abort_stage;
    volatile int32_t sqmd_abort_rc;
    volatile int32_t sqmd_abort_turn;
    volatile int32_t sqmd_abort_lane;
    volatile int32_t sqmd_abort_buffer;
    volatile int32_t sqmd_abort_item;
    volatile int32_t sqmd_abort_state;
    volatile int32_t sqmd_abort_buf_turn;
    volatile int32_t sqmd_abort_work_count;
    volatile int32_t sqmd_abort_result_count;
    volatile int32_t sqmd_abort_neighbor_id;
    volatile int32_t sqmd_abort_source_id;

    volatile uint64_t sqmd_source_jobs_created;
    volatile uint64_t sqmd_source_jobs_retired;
    volatile uint64_t sqmd_speculative_sources;
    volatile uint64_t sqmd_source_window_starvation;
    volatile uint64_t sqmd_inflight_sources_highwater;
} SqmdSharedStats;

#define SQMD_CONTROL_HEADER_BYTES 128u
#define SQMD_CONTROL_HEADER_FIELD_BYTES (4u * sizeof(int32_t))


typedef struct __attribute__((aligned(SQMD_CACHE_LINE_BYTES))) SqmdQuerySlot {
    volatile int32_t query_gen;
    int32_t          query_dim;
    int32_t          contract_id;
    int32_t          reserved;
    uint8_t          query_u8[RETRIEVAL_ASYNC_DEMO_MAX_DIM];
    float            query[RETRIEVAL_ASYNC_DEMO_MAX_DIM];
} SqmdQuerySlot;

#define SQMD_QUERY_CONTRACT_LEGACY 0
#define SQMD_QUERY_CONTRACT_L1_ALIGNED_U8 0x12020001


typedef struct __attribute__((aligned(SQMD_CACHE_LINE_BYTES))) SqmdStartupSlot {
    volatile int32_t cancel;
    volatile int32_t acknowledged;
    uint8_t _pad[SQMD_CACHE_LINE_BYTES - 2u * sizeof(int32_t)];
} SqmdStartupSlot;

typedef struct SqmdSharedControl {
    volatile int32_t ctrl_state;
    int32_t lane_count;
    int32_t turn_id;
    volatile int32_t session_state;
    uint8_t _ctrl_cache_pad[SQMD_CONTROL_HEADER_BYTES -
                             SQMD_CONTROL_HEADER_FIELD_BYTES];
    SqmdLane lanes[SQMD_DSP_LANES];
    SqmdSharedStats stats;
    SqmdQuerySlot query_slot;
    SqmdStartupSlot startup_slot;
#ifdef RETRIEVAL_ENABLE_FULL_BUILD
    RetrievalBuildPause build_pause;
#endif
} SqmdSharedControl;

#define SQMD_WORK_ITEM_EXPECTED_BYTES \
    (2u * sizeof(int32_t))
#define SQMD_RESULT_ITEM_EXPECTED_BYTES \
    (2u * sizeof(int32_t) + sizeof(float))

#define SQMD_LANE_BUFFER_RAW_BYTES \
    (4u * sizeof(int32_t) + \
     (size_t)SQMD_MAX_WORK_ITEMS * \
         (SQMD_WORK_ITEM_EXPECTED_BYTES + SQMD_RESULT_ITEM_EXPECTED_BYTES))
#define SQMD_LANE_BUFFER_EXPECTED_BYTES \
    SQMD_ALIGN_UP_BYTES(SQMD_LANE_BUFFER_RAW_BYTES, SQMD_CACHE_LINE_BYTES)

#define SQMD_LANE_EXPECTED_BYTES \
    ((size_t)SQMD_BUFFERS_PER_LANE * SQMD_LANE_BUFFER_EXPECTED_BYTES + \
     2u * SQMD_CACHE_LINE_BYTES)
_Static_assert(offsetof(SqmdLane,cpu_fill_buf) - offsetof(SqmdLane,dsp_active_buf) >= SQMD_CACHE_LINE_BYTES,
               "CPU FIFO cursor must not share a DSP-written cache line");
#define SQMD_SHARED_STATS_EXPECTED_BYTES \
    (17u * sizeof(uint64_t) + 12u * sizeof(int32_t))

#define SQMD_QUERY_SLOT_RAW_BYTES \
    (4u * sizeof(int32_t) + \
     (size_t)RETRIEVAL_ASYNC_DEMO_MAX_DIM * sizeof(uint8_t) + \
     (size_t)RETRIEVAL_ASYNC_DEMO_MAX_DIM * sizeof(float))
#define SQMD_QUERY_SLOT_EXPECTED_BYTES \
    SQMD_ALIGN_UP_BYTES(SQMD_QUERY_SLOT_RAW_BYTES, SQMD_CACHE_LINE_BYTES)
#define SQMD_STARTUP_SLOT_EXPECTED_BYTES SQMD_CACHE_LINE_BYTES

#ifdef RETRIEVAL_ENABLE_FULL_BUILD
#define SQMD_BUILD_PAUSE_BYTES RETRIEVAL_BUILD_PAUSE_BYTES
#else
#define SQMD_BUILD_PAUSE_BYTES 0u
#endif
#define SQMD_SHARED_CONTROL_EXPECTED_BYTES \
    SQMD_ALIGN_UP_BYTES( \
        SQMD_ALIGN_UP_BYTES(SQMD_CONTROL_HEADER_BYTES + \
                                (size_t)SQMD_DSP_LANES * \
                                    SQMD_LANE_EXPECTED_BYTES + \
                                sizeof(SqmdSharedStats), \
                            SQMD_CACHE_LINE_BYTES) + \
            SQMD_QUERY_SLOT_EXPECTED_BYTES + \
            SQMD_STARTUP_SLOT_EXPECTED_BYTES + SQMD_BUILD_PAUSE_BYTES, \
        SQMD_CACHE_LINE_BYTES)

_Static_assert(sizeof(SqmdWorkItem) == SQMD_WORK_ITEM_EXPECTED_BYTES,
               "SqmdWorkItem size changed unexpectedly");
_Static_assert(sizeof(SqmdResultItem) == SQMD_RESULT_ITEM_EXPECTED_BYTES,
               "SqmdResultItem size changed unexpectedly");
_Static_assert(offsetof(SqmdLaneBuffer, state) == 0u,
               "SqmdLaneBuffer.state must be at offset 0");
_Static_assert(offsetof(SqmdLaneBuffer, work) == 4u * sizeof(int32_t),
               "SqmdLaneBuffer.work must follow header");
_Static_assert(offsetof(SqmdLaneBuffer, result) ==
                   4u * sizeof(int32_t) +
                       (size_t)SQMD_MAX_WORK_ITEMS * sizeof(SqmdWorkItem),
               "SqmdLaneBuffer.result must follow work array");
_Static_assert(sizeof(SqmdLaneBuffer) == SQMD_LANE_BUFFER_EXPECTED_BYTES,
               "SqmdLaneBuffer size changed unexpectedly");
_Static_assert(sizeof(SqmdLane) == SQMD_LANE_EXPECTED_BYTES,
               "SqmdLane size changed unexpectedly");
_Static_assert(offsetof(SqmdSharedStats, sqmd_work_items) == 0u,
               "SqmdSharedStats.work_items must be first");
_Static_assert(offsetof(SqmdSharedStats, sqmd_abort_stage) ==
                   12u * sizeof(uint64_t),
               "SqmdSharedStats legacy abort offsets changed unexpectedly");
_Static_assert(offsetof(SqmdSharedStats, sqmd_source_jobs_created) ==
                   12u * sizeof(uint64_t) + 12u * sizeof(int32_t),
               "SqmdSharedStats pipeline counters must follow legacy fields");
_Static_assert(sizeof(SqmdSharedStats) == SQMD_SHARED_STATS_EXPECTED_BYTES,
               "SqmdSharedStats size changed unexpectedly");
_Static_assert(offsetof(SqmdSharedControl, ctrl_state) == 0u,
               "SqmdSharedControl.ctrl_state must be at offset 0");
_Static_assert(SQMD_CONTROL_HEADER_BYTES >= SQMD_CONTROL_HEADER_FIELD_BYTES,
               "SqmdSharedControl control header pad underflow");
_Static_assert(offsetof(SqmdSharedControl, lanes) == SQMD_CONTROL_HEADER_BYTES,
               "SqmdSharedControl.lanes must follow isolated control header");
_Static_assert(sizeof(SqmdQuerySlot) == SQMD_QUERY_SLOT_EXPECTED_BYTES,
               "SqmdQuerySlot size changed unexpectedly");
_Static_assert(offsetof(SqmdSharedControl, query_slot) % SQMD_CACHE_LINE_BYTES == 0u,
               "SqmdSharedControl.query_slot must be cache-line aligned");
_Static_assert(sizeof(SqmdStartupSlot) == SQMD_STARTUP_SLOT_EXPECTED_BYTES,
               "SqmdStartupSlot must occupy exactly one cache line");
_Static_assert(offsetof(SqmdSharedControl, startup_slot) %
                       SQMD_CACHE_LINE_BYTES == 0u,
               "SqmdSharedControl.startup_slot must be cache-line aligned");
_Static_assert(sizeof(SqmdSharedControl) == SQMD_SHARED_CONTROL_EXPECTED_BYTES,
               "SqmdSharedControl size changed unexpectedly");


#ifdef __hexagon__
#define DB_WRITE_BARRIER()  __asm__ __volatile__("barrier" ::: "memory")
#define DB_READ_BARRIER()   __asm__ __volatile__("barrier" ::: "memory")
#else
#define DB_WRITE_BARRIER()  __sync_synchronize()
#define DB_READ_BARRIER()   __sync_synchronize()
#endif

#endif
