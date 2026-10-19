#include "dsp_demo/retrieval_sqmd_common.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

static void sqmd_heap_sift_up(DemoCandidate *pool, int i) {
    DemoCandidate tmp = pool[i];

    while (i > 0) {
        int parent = (i - 1) / 2;
        if (tmp.distance <= pool[parent].distance)
            break;
        pool[i] = pool[parent];
        i = parent;
    }
    pool[i] = tmp;
}

static void sqmd_heap_sift_down(DemoCandidate *pool, int size, int i) {
    DemoCandidate tmp = pool[i];

    while (1) {
        int child = 2 * i + 1;
        if (child >= size)
            break;
        if (child + 1 < size &&
            pool[child + 1].distance > pool[child].distance)
            child++;
        if (tmp.distance >= pool[child].distance)
            break;
        pool[i] = pool[child];
        i = child;
    }
    pool[i] = tmp;
}

static void sqmd_heap_sort(DemoCandidate *pool, int size) {
    int i;

    for (i = size - 1; i > 0; i--) {
        DemoCandidate tmp = pool[0];
        pool[0] = pool[i];
        pool[i] = tmp;
        sqmd_heap_sift_down(pool, i, 0);
    }
}

static int sqmd_candidate_insert(DemoCandidate *pool,
                                 int *pool_size,
                                 int pool_capacity,
                                 int id,
                                 float distance) {
    int size;

    if (!pool || !pool_size || pool_capacity <= 0)
        return 0;

    size = *pool_size;
    if (size < pool_capacity) {
        pool[size].id = id;
        pool[size].distance = distance;
        pool[size].expanded = 0;
        sqmd_heap_sift_up(pool, size);
        *pool_size = size + 1;
        return 1;
    }

    if (distance < pool[0].distance) {
        pool[0].id = id;
        pool[0].distance = distance;
        pool[0].expanded = 0;
        sqmd_heap_sift_down(pool, size, 0);
        return 1;
    }

    return 0;
}

static void sqmd_frontier_sift_up(DemoFrontierEntry *f, int i) {
    DemoFrontierEntry tmp = f[i];

    while (i > 0) {
        int parent = (i - 1) / 2;
        if (tmp.distance >= f[parent].distance)
            break;
        f[i] = f[parent];
        i = parent;
    }
    f[i] = tmp;
}

static void sqmd_frontier_sift_down(DemoFrontierEntry *f, int size, int i) {
    DemoFrontierEntry tmp = f[i];

    while (1) {
        int child = 2 * i + 1;
        if (child >= size)
            break;
        if (child + 1 < size &&
            f[child + 1].distance < f[child].distance)
            child++;
        if (tmp.distance <= f[child].distance)
            break;
        f[i] = f[child];
        i = child;
    }
    f[i] = tmp;
}

static int sqmd_frontier_push(DemoFrontierEntry *f,
                              int *f_size,
                              int f_capacity,
                              int id,
                              float distance) {
    int pos;

    if (!f || !f_size || *f_size >= f_capacity)
        return 0;

    pos = *f_size;
    f[pos].id = id;
    f[pos].distance = distance;
    sqmd_frontier_sift_up(f, pos);
    *f_size = pos + 1;
    return 1;
}

static int sqmd_frontier_pop(DemoFrontierEntry *f, int *f_size) {
    int top_id;

    if (!f || !f_size || *f_size <= 0)
        return -1;

    top_id = f[0].id;
    *f_size -= 1;
    if (*f_size > 0) {
        f[0] = f[*f_size];
        sqmd_frontier_sift_down(f, *f_size, 0);
    }
    return top_id;
}

static void sqmd_reset_lane_buffer(SqmdLaneBuffer *buf) {
    int state;

    if (!buf)
        return;

    state = buf->state;
    memset(buf, 0, sizeof(*buf));
    buf->state = state;
}

void sqmd_shared_control_init_lanes(SqmdLane *lanes, int lane_count) {
    int lane;
    int buf_idx;

    if (!lanes || lane_count <= 0)
        return;

    if (lane_count > SQMD_DSP_LANES)
        lane_count = SQMD_DSP_LANES;

    for (lane = 0; lane < lane_count; lane++) {
        for (buf_idx = 0; buf_idx < SQMD_BUFFERS_PER_LANE; buf_idx++) {
            memset(&lanes[lane].buf[buf_idx],
                   0,
                   sizeof(lanes[lane].buf[buf_idx]));
            lanes[lane].buf[buf_idx].state = SQMD_BUF_EMPTY;
        }
        lanes[lane].dsp_active_buf = -1;
        lanes[lane].cpu_fill_buf = 0;
    }
}

void sqmd_shared_control_init(SqmdSharedControl *ctrl, int lane_count) {
    if (!ctrl)
        return;

    memset(ctrl, 0, sizeof(*ctrl));
    if (lane_count < 1)
        lane_count = 1;
    if (lane_count > SQMD_DSP_LANES)
        lane_count = SQMD_DSP_LANES;

    ctrl->ctrl_state = SQMD_CTRL_RUNNING;
    ctrl->lane_count = lane_count;
    ctrl->turn_id = 0;
    sqmd_shared_control_init_lanes(ctrl->lanes, lane_count);
}

int sqmd_lane_find_empty_buffer(SqmdLane *lane) {
    int start;
    int offset;

    if (!lane)
        return SQMD_STATUS_BAD_PARAM;

    start = lane->cpu_fill_buf;
    if (start < 0 || start >= SQMD_BUFFERS_PER_LANE)
        start = 0;

    for (offset = 0; offset < SQMD_BUFFERS_PER_LANE; offset++) {
        int idx = (start + offset) % SQMD_BUFFERS_PER_LANE;
        if (lane->buf[idx].state == SQMD_BUF_EMPTY)
            return idx;
    }

    return SQMD_STATUS_NO_EMPTY_BUFFER;
}

int sqmd_process_lane_buffer_payload_with_cap(SqmdLaneBuffer *buf,
                                              int expected_turn_id,
                                              int num_vectors,
                                              int work_items_cap,
                                              SqmdDistanceFn distance_fn,
                                              void *ctx) {
    int i;

    if (!buf || !distance_fn || buf->state != SQMD_BUF_CPU_READY ||
        buf->turn_id != expected_turn_id ||
        work_items_cap <= 0 || work_items_cap > SQMD_MAX_WORK_ITEMS ||
        buf->work_count < 0)
        return SQMD_STATUS_BAD_PARAM;
    if (buf->work_count > work_items_cap)
        return SQMD_STATUS_TOO_MUCH_WORK;

    for (i = 0; i < buf->work_count; i++) {
        SqmdWorkItem *work = &buf->work[i];
        SqmdResultItem *result = &buf->result[i];
        float coarse_dist = 0.0f;

        if (work->neighbor_id < 0 || work->neighbor_id >= num_vectors)
            return SQMD_STATUS_BAD_PARAM;
        if (distance_fn(work, &coarse_dist, ctx) != 0)
            return SQMD_STATUS_BAD_PARAM;

        result->neighbor_id = work->neighbor_id;
        result->source_id = work->source_id;
        result->coarse_dist = coarse_dist;
    }

    buf->result_count = buf->work_count;
    return SQMD_STATUS_OK;
}

int sqmd_process_lane_buffer_with_cap(SqmdLaneBuffer *buf,
                                      int expected_turn_id,
                                      int num_vectors,
                                      int work_items_cap,
                                      SqmdDistanceFn distance_fn,
                                      void *ctx) {
    int rc = sqmd_process_lane_buffer_payload_with_cap(buf,
                                                       expected_turn_id,
                                                       num_vectors,
                                                       work_items_cap,
                                                       distance_fn,
                                                       ctx);

    if (rc != SQMD_STATUS_OK)
        return rc;
    DB_WRITE_BARRIER();
    buf->state = SQMD_BUF_DSP_DONE;
    return SQMD_STATUS_OK;
}

int sqmd_process_lane_buffer(SqmdLaneBuffer *buf,
                             int expected_turn_id,
                             int num_vectors,
                             SqmdDistanceFn distance_fn,
                             void *ctx) {
    return sqmd_process_lane_buffer_with_cap(buf,
                                             expected_turn_id,
                                             num_vectors,
                                             SQMD_MAX_WORK_ITEMS,
                                             distance_fn,
                                             ctx);
}

int sqmd_dispatch_neighbors_to_lanes(const int32_t *neighbors,
                                     int neighbor_count,
                                     int num_vectors,
                                     int source_id,
                                     uint32_t *visited,
                                     SqmdLane *lanes,
                                     int lane_count,
                                     int turn_id,
                                     SqmdDispatchSummary *summary) {
    int work_ids[SQMD_WORK_NODES_PER_TURN * SQMD_MAX_WORK_ITEMS];
    int work_count = 0;
    int invalid_skips = 0;
    int dedup_skips = 0;
    int buffers_needed;
    int lane;
    int next_work;

    if (summary)
        memset(summary, 0, sizeof(*summary));

    if (!neighbors || neighbor_count < 0 || num_vectors <= 0 || !visited ||
        !lanes || lane_count <= 0 || lane_count > SQMD_DSP_LANES)
        return SQMD_STATUS_BAD_PARAM;

    if (neighbor_count >
        SQMD_WORK_NODES_PER_TURN * SQMD_MAX_WORK_ITEMS)
        return SQMD_STATUS_TOO_MUCH_WORK;

    for (next_work = 0; next_work < neighbor_count; next_work++) {
        int neighbor_id = neighbors[next_work];
        int duplicate_in_batch = 0;
        int i;

        if (neighbor_id < 0 || neighbor_id >= num_vectors ||
            neighbor_id == source_id) {
            invalid_skips++;
            continue;
        }
        for (i = 0; i < work_count; i++) {
            if (work_ids[i] == neighbor_id) {
                duplicate_in_batch = 1;
                break;
            }
        }
        if (duplicate_in_batch || VISITED_TEST(visited, neighbor_id)) {
            dedup_skips++;
            continue;
        }

        work_ids[work_count++] = neighbor_id;
    }

    if (summary) {
        summary->work_items = work_count;
        summary->dedup_skips = dedup_skips;
        summary->invalid_skips = invalid_skips;
        summary->empty_turn = (work_count == 0);
    }

    if (work_count == 0)
        return SQMD_STATUS_OK;

    buffers_needed = (work_count < lane_count) ? work_count : lane_count;
    for (lane = 0; lane < buffers_needed; lane++) {
        if (sqmd_lane_find_empty_buffer(&lanes[lane]) < 0)
            return SQMD_STATUS_NO_EMPTY_BUFFER;
    }

    for (next_work = 0; next_work < work_count; next_work++)
        VISITED_SET(visited, work_ids[next_work]);

    next_work = 0;
    for (lane = 0; lane < buffers_needed; lane++) {
        SqmdLane *cur_lane = &lanes[lane];
        int buf_idx = sqmd_lane_find_empty_buffer(cur_lane);
        SqmdLaneBuffer *buf = &cur_lane->buf[buf_idx];
        int remaining = work_count - next_work;
        int lanes_left = buffers_needed - lane;
        int shard_count = (remaining + lanes_left - 1) / lanes_left;
        int i;

        sqmd_reset_lane_buffer(buf);
        buf->turn_id = turn_id;
        buf->work_count = shard_count;
        buf->result_count = 0;

        for (i = 0; i < shard_count; i++) {
            buf->work[i].neighbor_id = work_ids[next_work++];
            buf->work[i].source_id = source_id;
        }

        DB_WRITE_BARRIER();
        buf->state = SQMD_BUF_CPU_READY;
        cur_lane->cpu_fill_buf = (buf_idx + 1) % SQMD_BUFFERS_PER_LANE;

        if (summary)
            summary->buffers_submitted++;
    }

    return SQMD_STATUS_OK;
}

int sqmd_cpu_state_init(sqmdCpuSearchState *st,
                        int num_vectors,
                        int pool_capacity,
                        int entry_points,
                        int expand_count,
                        const float *query,
                        const RetrievalDemoEncodedDataset *dataset,
                        ResidualBlockCache *res_cache) {
    size_t frontier_capacity;

    if (!st || num_vectors <= 0 || pool_capacity <= 0 ||
        entry_points <= 0 || expand_count <= 0)
        return SQMD_STATUS_BAD_PARAM;

    memset(st, 0, sizeof(*st));
    frontier_capacity = (size_t)entry_points +
                        (size_t)pool_capacity * (size_t)expand_count;
    if (frontier_capacity > (size_t)INT32_MAX)
        return SQMD_STATUS_BAD_PARAM;

    st->pool = (DemoCandidate *)calloc((size_t)pool_capacity,
                                       sizeof(*st->pool));
    st->frontier = (DemoFrontierEntry *)calloc(frontier_capacity,
                                              sizeof(*st->frontier));
    st->visited_words = VISITED_BITMAP_WORDS(num_vectors);
    st->visited = (uint32_t *)calloc((size_t)st->visited_words,
                                    sizeof(*st->visited));
    if (!st->pool || !st->frontier || !st->visited) {
        sqmd_cpu_state_cleanup(st);
        return SQMD_STATUS_BAD_PARAM;
    }

    st->pool_capacity = pool_capacity;
    st->frontier_capacity = (int)frontier_capacity;
    st->query = query;
    st->dataset = dataset;
    st->res_cache = res_cache;
    st->entry_points = entry_points;
    st->expand_count = expand_count;
    st->num_vectors = num_vectors;
    return SQMD_STATUS_OK;
}

void sqmd_cpu_state_cleanup(sqmdCpuSearchState *st) {
    if (!st)
        return;

    free(st->pool);
    free(st->frontier);
    free(st->visited);
    st->pool = NULL;
    st->frontier = NULL;
    st->visited = NULL;
    st->pool_size = 0;
    st->pool_capacity = 0;
    st->frontier_size = 0;
    st->frontier_capacity = 0;
    st->visited_words = 0;
    st->num_vectors = 0;
}

int sqmd_cpu_state_seed(sqmdCpuSearchState *st,
                        const int *seed_ids,
                        const float *seed_dists,
                        int seed_count) {
    int i;

    if (!st || !st->pool || !st->frontier || !st->visited ||
        !seed_ids || !seed_dists || seed_count < 0)
        return SQMD_STATUS_BAD_PARAM;

    for (i = 0; i < seed_count; i++) {
        int id = seed_ids[i];
        if (id < 0 || id >= st->num_vectors)
            return SQMD_STATUS_BAD_PARAM;
        if (VISITED_TEST(st->visited, id))
            continue;
        VISITED_SET(st->visited, id);
        if (sqmd_candidate_insert(st->pool,
                                  &st->pool_size,
                                  st->pool_capacity,
                                  id,
                                  seed_dists[i]))
            sqmd_frontier_push(st->frontier,
                               &st->frontier_size,
                               st->frontier_capacity,
                               id,
                               seed_dists[i]);
    }

    return SQMD_STATUS_OK;
}

static int sqmd_cpu_state_try_peek_source(sqmdCpuSearchState *st,
                                          int *source_id) {
    if (!st || !source_id)
        return SQMD_STATUS_BAD_PARAM;

    while (st->frontier_size > 0) {
        if (st->pool_size >= st->pool_capacity &&
            st->frontier[0].distance > st->pool[0].distance) {
            sqmd_frontier_pop(st->frontier, &st->frontier_size);
            continue;
        }
        break;
    }

    if (st->frontier_size <= 0) {
        *source_id = -1;
        return 0;
    }

    *source_id = st->frontier[0].id;
    return 1;
}

int sqmd_cpu_state_pop_source(sqmdCpuSearchState *st, int *source_id) {
    int rc;

    rc = sqmd_cpu_state_try_peek_source(st, source_id);
    if (rc <= 0) {
        if (rc == 0)
            st->finished = 1;
        return rc;
    }

    *source_id = sqmd_frontier_pop(st->frontier, &st->frontier_size);
    return (*source_id >= 0) ? 1 : SQMD_STATUS_BAD_PARAM;
}

static float sqmd_default_correction(int neighbor_id,
                                     int source_id,
                                     float coarse_dist,
                                     void *ctx) {
    (void)neighbor_id;
    (void)source_id;
    (void)ctx;
    return coarse_dist;
}

static void sqmd_cpu_state_record_abort(sqmdCpuSearchState *st,
                                        int stage,
                                        int rc,
                                        int turn_id,
                                        int lane,
                                        int buffer,
                                        int item,
                                        const SqmdLaneBuffer *buf,
                                        int neighbor_id,
                                        int source_id) {
    if (!st)
        return;

    st->abort_stage = stage;
    st->abort_rc = rc;
    st->abort_turn = turn_id;
    st->abort_lane = lane;
    st->abort_buffer = buffer;
    st->abort_item = item;
    st->abort_neighbor_id = neighbor_id;
    st->abort_source_id = source_id;

    if (buf) {
        st->abort_state = buf->state;
        st->abort_buf_turn = buf->turn_id;
        st->abort_work_count = buf->work_count;
        st->abort_result_count = buf->result_count;
    } else {
        st->abort_state = -1;
        st->abort_buf_turn = -1;
        st->abort_work_count = -1;
        st->abort_result_count = -1;
    }
}

static int sqmd_cpu_state_capture_dsp_abort(sqmdCpuSearchState *st,
                                            const SqmdSharedControl *ctrl) {
    int lane;
    int buffer;

    if (!st || !ctrl || ctrl->lane_count <= 0 ||
        ctrl->lane_count > SQMD_DSP_LANES)
        return 0;

    for (lane = 0; lane < ctrl->lane_count; lane++) {
        for (buffer = 0; buffer < SQMD_BUFFERS_PER_LANE; buffer++) {
            const SqmdLaneBuffer *buf = &ctrl->lanes[lane].buf[buffer];
            int neighbor_id = -1;
            int source_id = -1;
            int rc;

            if (buf->state != SQMD_BUF_DSP_ABORT)
                continue;
            DB_READ_BARRIER();
            if (buf->state != SQMD_BUF_DSP_ABORT)
                continue;

            rc = buf->result_count;
            if (rc >= 0)
                rc = SQMD_STATUS_BAD_PARAM;
            if (buf->work_count > 0 &&
                buf->work_count <= SQMD_MAX_WORK_ITEMS) {
                neighbor_id = buf->work[0].neighbor_id;
                source_id = buf->work[0].source_id;
            }
            sqmd_cpu_state_record_abort(st,
                                        SQMD_ABORT_DSP_PROCESS,
                                        rc,
                                        buf->turn_id,
                                        lane,
                                        buffer,
                                        -1,
                                        buf,
                                        neighbor_id,
                                        source_id);
            return 1;
        }
    }
    return 0;
}

static void sqmd_cpu_state_clear_abort(sqmdCpuSearchState *st) {
    sqmd_cpu_state_record_abort(st,
                                SQMD_ABORT_NONE,
                                SQMD_STATUS_OK,
                                0,
                                -1,
                                -1,
                                -1,
                                NULL,
                                -1,
                                -1);
}

static void sqmd_stats_copy_abort_from_cpu(SqmdSharedStats *stats,
                                           const sqmdCpuSearchState *st) {
    if (!stats || !st)
        return;

    stats->sqmd_abort_stage = st->abort_stage;
    stats->sqmd_abort_rc = st->abort_rc;
    stats->sqmd_abort_turn = st->abort_turn;
    stats->sqmd_abort_lane = st->abort_lane;
    stats->sqmd_abort_buffer = st->abort_buffer;
    stats->sqmd_abort_item = st->abort_item;
    stats->sqmd_abort_state = st->abort_state;
    stats->sqmd_abort_buf_turn = st->abort_buf_turn;
    stats->sqmd_abort_work_count = st->abort_work_count;
    stats->sqmd_abort_result_count = st->abort_result_count;
    stats->sqmd_abort_neighbor_id = st->abort_neighbor_id;
    stats->sqmd_abort_source_id = st->abort_source_id;
}

int sqmd_cpu_state_apply_results_with_correction(sqmdCpuSearchState *st,
                                                 SqmdLaneBuffer *buf,
                                                 int expected_turn_id,
                                                 SqmdCorrectionFn correction_fn,
                                                 void *ctx) {
    int i;

    if (!st || !buf)
        return SQMD_STATUS_BAD_PARAM;
    if (buf->state != SQMD_BUF_DSP_DONE ||
        buf->turn_id != expected_turn_id ||
        buf->result_count < 0 ||
        buf->result_count > SQMD_MAX_WORK_ITEMS) {
        sqmd_cpu_state_record_abort(st,
                                    SQMD_ABORT_APPLY_HEADER,
                                    SQMD_STATUS_BAD_PARAM,
                                    expected_turn_id,
                                    -1,
                                    -1,
                                    -1,
                                    buf,
                                    -1,
                                    -1);
        return SQMD_STATUS_BAD_PARAM;
    }
    if (!correction_fn)
        correction_fn = sqmd_default_correction;

    for (i = 0; i < buf->result_count; i++) {
        int id = buf->result[i].neighbor_id;
        int source_id = buf->result[i].source_id;
        float distance;

        if (id < 0 || id >= st->num_vectors ||
            source_id < 0 || source_id >= st->num_vectors) {
            sqmd_cpu_state_record_abort(st,
                                        SQMD_ABORT_APPLY_RESULT,
                                        SQMD_STATUS_BAD_PARAM,
                                        expected_turn_id,
                                        -1,
                                        -1,
                                        i,
                                        buf,
                                        id,
                                        source_id);
            return SQMD_STATUS_BAD_PARAM;
        }
        distance = correction_fn(id,
                                 source_id,
                                 buf->result[i].coarse_dist,
                                 ctx);
        if (sqmd_candidate_insert(st->pool,
                                  &st->pool_size,
                                  st->pool_capacity,
                                  id,
                                  distance))
            sqmd_frontier_push(st->frontier,
                               &st->frontier_size,
                               st->frontier_capacity,
                               id,
                               distance);
        st->exact_corrected++;
    }

    st->dsp_done_batches++;
    buf->work_count = 0;
    buf->result_count = 0;
    DB_WRITE_BARRIER();
    buf->state = SQMD_BUF_EMPTY;
    return SQMD_STATUS_OK;
}

int sqmd_cpu_state_apply_results(sqmdCpuSearchState *st,
                                 SqmdLaneBuffer *buf,
                                 int expected_turn_id) {
    return sqmd_cpu_state_apply_results_with_correction(st,
                                                        buf,
                                                        expected_turn_id,
                                                        NULL,
                                                        NULL);
}

int sqmd_cpu_state_dispatch_next_turn(sqmdCpuSearchState *st,
                                      const int32_t *graph,
                                      int graph_degree,
                                      int num_vectors,
                                      SqmdLane *lanes,
                                      int lane_count,
                                      int turn_id,
                                      SqmdDispatchSummary *summary,
                                      int *source_id) {
    const int32_t *neighbors;
    SqmdDispatchSummary local_summary;
    int popped_id = -1;
    int expand_count;
    int rc;

    if (!st || !graph || graph_degree <= 0 || !lanes || !source_id)
        return SQMD_STATUS_BAD_PARAM;

    rc = sqmd_cpu_state_pop_source(st, &popped_id);
    if (rc <= 0) {
        *source_id = -1;
        if (summary)
            memset(summary, 0, sizeof(*summary));
        return rc;
    }
    if (popped_id < 0 || popped_id >= num_vectors)
        return SQMD_STATUS_BAD_PARAM;

    neighbors = graph + (size_t)popped_id * (size_t)graph_degree;
    expand_count = (st->expand_count < graph_degree)
                       ? st->expand_count : graph_degree;
    memset(&local_summary, 0, sizeof(local_summary));
    rc = sqmd_dispatch_neighbors_to_lanes(neighbors,
                                          expand_count,
                                          num_vectors,
                                          popped_id,
                                          st->visited,
                                          lanes,
                                          lane_count,
                                          turn_id,
                                          &local_summary);
    if (rc != SQMD_STATUS_OK)
        return rc;

    st->work_items += (uint64_t)local_summary.work_items;
    st->dedup_skips += (uint64_t)local_summary.dedup_skips;
    if (local_summary.empty_turn)
        st->empty_turns++;

    if (summary)
        *summary = local_summary;
    *source_id = popped_id;
    return 1;
}

static int sqmd_shared_control_turn_complete(const SqmdSharedControl *ctrl,
                                             int turn_id) {
    int lane;
    int buf_idx;

    if (!ctrl)
        return 0;

    for (lane = 0; lane < ctrl->lane_count; lane++) {
        for (buf_idx = 0; buf_idx < SQMD_BUFFERS_PER_LANE; buf_idx++) {
            const SqmdLaneBuffer *buf = &ctrl->lanes[lane].buf[buf_idx];

            if (buf->turn_id == turn_id &&
                (buf->state == SQMD_BUF_CPU_READY ||
                 buf->state == SQMD_BUF_DSP_DONE ||
                 buf->state == SQMD_BUF_DSP_ABORT))
                return 0;
        }
    }

    return 1;
}

int sqmd_shared_control_dispatch_next_turn(sqmdCpuSearchState *st,
                                           SqmdSharedControl *ctrl,
                                           const int32_t *graph,
                                           int graph_degree,
                                           int num_vectors,
                                           int turn_id,
                                           SqmdDispatchSummary *summary,
                                           int *source_id) {
    int rc;

    if (!st || !ctrl || !source_id || ctrl->lane_count <= 0 ||
        ctrl->lane_count > SQMD_DSP_LANES)
        return SQMD_STATUS_BAD_PARAM;
    if (turn_id > 1 &&
        !sqmd_shared_control_turn_complete(ctrl, turn_id - 1))
        return SQMD_STATUS_TURN_INCOMPLETE;

    ctrl->turn_id = turn_id;
    rc = sqmd_cpu_state_dispatch_next_turn(st,
                                           graph,
                                           graph_degree,
                                           num_vectors,
                                           ctrl->lanes,
                                           ctrl->lane_count,
                                           turn_id,
                                           summary,
                                           source_id);
    if (rc < 0)
        ctrl->ctrl_state = SQMD_CTRL_ABORT;
    return rc;
}

int sqmd_shared_control_collect_turn_timed(sqmdCpuSearchState *st,
                                           SqmdSharedControl *ctrl,
                                           int expected_turn_id,
                                           SqmdCorrectionFn correction_fn,
                                           void *correction_ctx,
                                           SqmdTimeFn time_fn,
                                           void *time_ctx) {
    int lane;
    int buf_idx;
    int saw_pending = 0;

    if (!st || !ctrl || expected_turn_id <= 0 ||
        ctrl->lane_count <= 0 || ctrl->lane_count > SQMD_DSP_LANES)
        return SQMD_STATUS_BAD_PARAM;

    if (sqmd_cpu_state_capture_dsp_abort(st, ctrl)) {
        ctrl->ctrl_state = SQMD_CTRL_ABORT;
        return st->abort_rc;
    }

    for (lane = 0; lane < ctrl->lane_count; lane++) {
        for (buf_idx = 0; buf_idx < SQMD_BUFFERS_PER_LANE; buf_idx++) {
            SqmdLaneBuffer *buf = &ctrl->lanes[lane].buf[buf_idx];

            if (buf->turn_id != expected_turn_id)
                continue;
            if (buf->state == SQMD_BUF_DSP_DONE) {
                DB_READ_BARRIER();
                uint64_t t0 = time_fn ? time_fn(time_ctx) : 0u;
                int rc = sqmd_cpu_state_apply_results_with_correction(
                    st,
                    buf,
                    expected_turn_id,
                    correction_fn,
                    correction_ctx);
                if (time_fn)
                    ctrl->stats.sqmd_cpu_correct_us +=
                        time_fn(time_ctx) - t0;
                if (rc != SQMD_STATUS_OK) {
                    int stage = st->abort_stage ? st->abort_stage
                                                : SQMD_ABORT_COLLECT;
                    int item = st->abort_item;
                    int neighbor_id = st->abort_neighbor_id;
                    int source_id = st->abort_source_id;

                    sqmd_cpu_state_record_abort(st,
                                                stage,
                                                rc,
                                                expected_turn_id,
                                                lane,
                                                buf_idx,
                                                item,
                                                buf,
                                                neighbor_id,
                                                source_id);
                    ctrl->ctrl_state = SQMD_CTRL_ABORT;
                    return rc;
                }
            } else if (buf->state == SQMD_BUF_CPU_READY) {
                saw_pending = 1;
            }
        }
    }

    if (saw_pending)
        return 0;
    return 1;
}

int sqmd_shared_control_collect_turn(sqmdCpuSearchState *st,
                                     SqmdSharedControl *ctrl,
                                     int expected_turn_id,
                                     SqmdCorrectionFn correction_fn,
                                     void *ctx) {
    return sqmd_shared_control_collect_turn_timed(st,
                                                  ctrl,
                                                  expected_turn_id,
                                                  correction_fn,
                                                  ctx,
                                                  NULL,
                                                  NULL);
}

static int sqmd_source_pipeline_check_protocol(sqmdCpuSearchState *st,
                                               SqmdSharedControl *ctrl,
                                               int captured_query_gen,
                                               int source_sequence,
                                               int source_id);
static void sqmd_source_pipeline_merge_stats(SqmdSharedStats *stats,
                                             const sqmdCpuSearchState *st);

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
                                      int out_cap) {
    int turn_id = 1;
    int captured_query_gen = 0;
    int out_count;
    int i;

    if (st)
        sqmd_cpu_state_clear_abort(st);

    if (!st || !ctrl || !graph || graph_degree <= 0 || num_vectors <= 0 ||
        !out_ids || !out_dists || out_cap <= 0 ||
        ctrl->lane_count <= 0 || ctrl->lane_count > SQMD_DSP_LANES) {
        sqmd_cpu_state_record_abort(st,
                                    SQMD_ABORT_DISPATCH,
                                    SQMD_STATUS_BAD_PARAM,
                                    0,
                                    -1,
                                    -1,
                                    -1,
                                    NULL,
                                    -1,
                                    -1);
        if (ctrl)
            ctrl->ctrl_state = SQMD_CTRL_ABORT;
        return -1;
    }

    captured_query_gen = ctrl->query_slot.query_gen;
    if (sqmd_source_pipeline_check_protocol(st,
                                            ctrl,
                                            captured_query_gen,
                                            0,
                                            -1) != SQMD_STATUS_OK)
        goto abort;

    while (!st->finished) {
        SqmdDispatchSummary summary;
        int source_id = -1;
        int dispatched;
        uint64_t turn_done_start;
        uint64_t turn_done_target;
        uint64_t turn_wait_start = 0u;

        if (sqmd_source_pipeline_check_protocol(st,
                                                ctrl,
                                                captured_query_gen,
                                                turn_id,
                                                -1) != SQMD_STATUS_OK)
            goto abort;

        dispatched = sqmd_shared_control_dispatch_next_turn(st,
                                                            ctrl,
                                                            graph,
                                                            graph_degree,
                                                            num_vectors,
                                                            turn_id,
                                                            &summary,
                                                            &source_id);
        if (dispatched < 0) {
            if (st->abort_stage == SQMD_ABORT_NONE)
                sqmd_cpu_state_record_abort(st,
                                            SQMD_ABORT_DISPATCH,
                                            dispatched,
                                            turn_id,
                                            -1,
                                            -1,
                                            -1,
                                            NULL,
                                            source_id,
                                            -1);
            goto abort;
        }
        if (dispatched == 0)
            break;

        turn_done_start = st->dsp_done_batches;
        turn_done_target =
            turn_done_start + (uint64_t)summary.buffers_submitted;
        if (time_fn)
            turn_wait_start = time_fn(time_ctx);

        while (st->dsp_done_batches < turn_done_target) {
            int collected;

            if (sqmd_source_pipeline_check_protocol(st,
                                                    ctrl,
                                                    captured_query_gen,
                                                    turn_id,
                                                    source_id) !=
                SQMD_STATUS_OK)
                goto abort;

            collected = sqmd_shared_control_collect_turn_timed(
                st,
                ctrl,
                turn_id,
                correction_fn,
                correction_ctx,
                time_fn,
                time_ctx);
            if (collected < 0) {
                if (st->abort_stage == SQMD_ABORT_NONE)
                    sqmd_cpu_state_record_abort(st,
                                                SQMD_ABORT_COLLECT,
                                                collected,
                                                turn_id,
                                                -1,
                                                -1,
                                                -1,
                                                NULL,
                                                source_id,
                                                -1);
                goto abort;
            }
            if (sqmd_source_pipeline_check_protocol(st,
                                                    ctrl,
                                                    captured_query_gen,
                                                    turn_id,
                                                    source_id) !=
                SQMD_STATUS_OK)
                goto abort;
            if (st->dsp_done_batches >= turn_done_target)
                break;
            if (time_fn && SQMD_TURN_WAIT_TIMEOUT_US > 0u &&
                time_fn(time_ctx) - turn_wait_start >
                    SQMD_TURN_WAIT_TIMEOUT_US) {
                sqmd_cpu_state_record_abort(st,
                                            SQMD_ABORT_TURN_WAIT_TIMEOUT,
                                            SQMD_STATUS_TURN_TIMEOUT,
                                            turn_id,
                                            -1,
                                            -1,
                                            -1,
                                            NULL,
                                            source_id,
                                            -1);
                goto abort;
            }
            if (wait_fn) {
                uint64_t wait_t0 = time_fn ? time_fn(time_ctx) : 0u;

                wait_fn(wait_ctx);
                if (time_fn)
                    ctrl->stats.sqmd_cpu_wait_dsp_us +=
                        time_fn(time_ctx) - wait_t0;
            }
        }

        turn_id++;
    }

    if (sqmd_source_pipeline_check_protocol(st,
                                            ctrl,
                                            captured_query_gen,
                                            ctrl->turn_id,
                                            -1) != SQMD_STATUS_OK)
        goto abort;

    sqmd_heap_sort(st->pool, st->pool_size);
    out_count = (st->pool_size < out_cap) ? st->pool_size : out_cap;
    for (i = 0; i < out_count; i++) {
        out_ids[i] = st->pool[i].id;
        out_dists[i] = st->pool[i].distance;
    }

    if (sqmd_source_pipeline_check_protocol(st,
                                            ctrl,
                                            captured_query_gen,
                                            ctrl->turn_id,
                                            -1) != SQMD_STATUS_OK)
        goto abort;

    ctrl->stats.sqmd_work_items += st->work_items;
    ctrl->stats.sqmd_dedup_skips += st->dedup_skips;
    ctrl->stats.sqmd_empty_turns += st->empty_turns;
    ctrl->stats.sqmd_lane_wait_us += st->lane_wait_us;
    ctrl->stats.sqmd_dsp_done_batches += st->dsp_done_batches;
    DB_WRITE_BARRIER();
    ctrl->ctrl_state = SQMD_CTRL_DONE;
    DB_WRITE_BARRIER();
    return out_count;

abort:
    st->finished = 0;
    sqmd_source_pipeline_merge_stats(&ctrl->stats, st);
    ctrl->stats.sqmd_abort_count++;
    if (ctrl->stats.sqmd_abort_stage == SQMD_ABORT_NONE ||
        st->abort_stage != SQMD_ABORT_CTRL)
        sqmd_stats_copy_abort_from_cpu(&ctrl->stats, st);
    DB_WRITE_BARRIER();
    ctrl->ctrl_state = SQMD_CTRL_ABORT;
    DB_WRITE_BARRIER();
    return -1;
}


#define SQMD_SOURCE_WINDOW_MAX 2
#define SQMD_SOURCE_ACCEPTED_MAX \
    (SQMD_WORK_NODES_PER_TURN * SQMD_MAX_WORK_ITEMS)

typedef struct SqmdSourceShardRef {
    int lane;
    int buffer;
    int accepted_offset;
    int expected_count;
} SqmdSourceShardRef;

typedef struct SqmdSourceJob {
    int source_sequence;
    int source_id;
    int accepted_count;
    int shard_count;
    int accepted_ids[SQMD_SOURCE_ACCEPTED_MAX];
    float residual_delta[SQMD_SOURCE_ACCEPTED_MAX];
    int residual_delta_ready;
    SqmdSourceShardRef shard[SQMD_DSP_LANES];
} SqmdSourceJob;

static int sqmd_shared_control_buffers_empty(const SqmdSharedControl *ctrl) {
    int lane;
    int buffer;

    if (!ctrl || ctrl->lane_count <= 0 ||
        ctrl->lane_count > SQMD_DSP_LANES)
        return 0;

    for (lane = 0; lane < ctrl->lane_count; lane++) {
        for (buffer = 0; buffer < SQMD_BUFFERS_PER_LANE; buffer++) {
            if (ctrl->lanes[lane].buf[buffer].state != SQMD_BUF_EMPTY)
                return 0;
        }
    }
    return 1;
}

static int sqmd_source_pipeline_control_ok(const SqmdSharedControl *ctrl,
                                           int captured_query_gen) {
    if (!ctrl)
        return 0;
    if (ctrl->ctrl_state != SQMD_CTRL_RUNNING ||
        ctrl->session_state == SQMD_SESSION_ABORT ||
        ctrl->session_state == SQMD_SESSION_SHUTDOWN)
        return 0;
    return ctrl->query_slot.query_gen == captured_query_gen;
}

static int sqmd_source_pipeline_check_protocol(sqmdCpuSearchState *st,
                                               SqmdSharedControl *ctrl,
                                               int captured_query_gen,
                                               int source_sequence,
                                               int source_id) {
    DB_READ_BARRIER();
    if (sqmd_cpu_state_capture_dsp_abort(st, ctrl))
        return st->abort_rc;
    if (sqmd_source_pipeline_control_ok(ctrl, captured_query_gen))
        return SQMD_STATUS_OK;

    sqmd_cpu_state_record_abort(st,
                                SQMD_ABORT_CTRL,
                                SQMD_STATUS_BAD_PARAM,
                                source_sequence,
                                -1,
                                -1,
                                -1,
                                NULL,
                                -1,
                                source_id);
    return SQMD_STATUS_BAD_PARAM;
}

static void sqmd_source_job_clear_prepared(SqmdSharedControl *ctrl,
                                           const SqmdSourceJob *job) {
    int ordinal;

    if (!ctrl || !job)
        return;
    for (ordinal = 0; ordinal < job->shard_count; ordinal++) {
        const SqmdSourceShardRef *ref = &job->shard[ordinal];
        SqmdLaneBuffer *buf = &ctrl->lanes[ref->lane].buf[ref->buffer];

        if (buf->state == SQMD_BUF_EMPTY)
            sqmd_reset_lane_buffer(buf);
    }
}


static int sqmd_source_pipeline_admit_one(sqmdCpuSearchState *st,
                                          SqmdSharedControl *ctrl,
                                          const int32_t *graph,
                                          int graph_degree,
                                          int num_vectors,
                                          int source_sequence,
                                          int captured_query_gen,
                                          int current_inflight,
                                          SqmdSourceJob *job) {
    const int32_t *neighbors;
    int accepted_ids[SQMD_SOURCE_ACCEPTED_MAX];
    int32_t row_snapshot[SQMD_SOURCE_ACCEPTED_MAX];
    int accepted_count = 0;
    int dedup_skips = 0;
    int source_id = -1;
    int expand_count;
    int buffers_needed;
    int popped_id;
    int ordinal;
    int next_work;
    int rc;

    if (!st || !ctrl || !graph || graph_degree <= 0 || num_vectors <= 0 ||
        source_sequence <= 0 || !job)
        return SQMD_STATUS_BAD_PARAM;

    memset(job, 0, sizeof(*job));
    rc = sqmd_cpu_state_try_peek_source(st, &source_id);
    if (rc <= 0)
        return rc;
    if (source_id < 0 || source_id >= num_vectors)
        return SQMD_STATUS_BAD_PARAM;

    expand_count = (st->expand_count < graph_degree)
                       ? st->expand_count : graph_degree;
    if (expand_count < 0 || expand_count > SQMD_SOURCE_ACCEPTED_MAX)
        return SQMD_STATUS_TOO_MUCH_WORK;
    neighbors = graph + (size_t)source_id * (size_t)graph_degree;

    if (st->graph_reader.read) {
        if (st->graph_reader.read(st->graph_reader.ctx, source_id,
                                  row_snapshot, expand_count) != 0)
            return SQMD_STATUS_BAD_PARAM;
        neighbors = row_snapshot;
    }

    for (next_work = 0; next_work < expand_count; next_work++) {
        int neighbor_id = neighbors[next_work];
        int duplicate_in_batch = 0;
        int i;

        if (neighbor_id < 0 || neighbor_id >= num_vectors ||
            neighbor_id == source_id) {
            continue;
        }
        for (i = 0; i < accepted_count; i++) {
            if (accepted_ids[i] == neighbor_id) {
                duplicate_in_batch = 1;
                break;
            }
        }
        if (duplicate_in_batch || VISITED_TEST(st->visited, neighbor_id)) {
            dedup_skips++;
            continue;
        }
        accepted_ids[accepted_count++] = neighbor_id;
    }

    job->source_sequence = source_sequence;
    job->source_id = source_id;
    job->accepted_count = accepted_count;
    if (accepted_count > 0) {
        memcpy(job->accepted_ids,
               accepted_ids,
               (size_t)accepted_count * sizeof(job->accepted_ids[0]));
    }

    buffers_needed = (accepted_count < ctrl->lane_count)
                         ? accepted_count : ctrl->lane_count;
    job->shard_count = buffers_needed;


    next_work = 0;
    for (ordinal = 0; ordinal < buffers_needed; ordinal++) {
        SqmdLane *lane = &ctrl->lanes[ordinal];
        SqmdSourceShardRef *ref = &job->shard[ordinal];
        int remaining = accepted_count - next_work;
        int lanes_left = buffers_needed - ordinal;
        int shard_count = (remaining + lanes_left - 1) / lanes_left;
        int buffer = sqmd_lane_find_empty_buffer(lane);

        if (buffer < 0)
            return buffer;

        ref->lane = ordinal;
        ref->buffer = buffer;
        ref->accepted_offset = next_work;
        ref->expected_count = shard_count;
        next_work += shard_count;
    }


    for (ordinal = 0; ordinal < buffers_needed; ordinal++) {
        const SqmdSourceShardRef *ref = &job->shard[ordinal];
        SqmdLaneBuffer *buf = &ctrl->lanes[ref->lane].buf[ref->buffer];
        int i;

        if (buf->state != SQMD_BUF_EMPTY)
            return SQMD_STATUS_NO_EMPTY_BUFFER;
        sqmd_reset_lane_buffer(buf);
        buf->turn_id = source_sequence;
        buf->work_count = ref->expected_count;
        buf->result_count = 0;
        for (i = 0; i < ref->expected_count; i++) {
            buf->work[i].neighbor_id =
                accepted_ids[ref->accepted_offset + i];
            buf->work[i].source_id = source_id;
        }
    }


    if (sqmd_source_pipeline_check_protocol(st,
                                            ctrl,
                                            captured_query_gen,
                                            source_sequence,
                                            source_id) != SQMD_STATUS_OK) {
        sqmd_source_job_clear_prepared(ctrl, job);
        return SQMD_STATUS_BAD_PARAM;
    }


    popped_id = sqmd_frontier_pop(st->frontier, &st->frontier_size);
    if (popped_id != source_id)
        return SQMD_STATUS_BAD_PARAM;

    for (next_work = 0; next_work < accepted_count; next_work++)
        VISITED_SET(st->visited, accepted_ids[next_work]);

    st->work_items += (uint64_t)accepted_count;
    st->dedup_skips += (uint64_t)dedup_skips;
    st->source_jobs_created++;
    ctrl->turn_id = source_sequence;

    for (ordinal = 0; ordinal < job->shard_count; ordinal++) {
        const SqmdSourceShardRef *ref = &job->shard[ordinal];
        SqmdLane *lane = &ctrl->lanes[ref->lane];

        lane->cpu_fill_buf =
            (ref->buffer + 1) % SQMD_BUFFERS_PER_LANE;
    }

    if (current_inflight > 0)
        st->speculative_sources++;
    if ((uint32_t)(current_inflight + 1) > st->inflight_sources_highwater)
        st->inflight_sources_highwater = (uint32_t)(current_inflight + 1);


    if (accepted_count == 0)
        return 1;

    DB_WRITE_BARRIER();
    for (ordinal = 0; ordinal < job->shard_count; ordinal++) {
        const SqmdSourceShardRef *ref = &job->shard[ordinal];
        ctrl->lanes[ref->lane].buf[ref->buffer].state = SQMD_BUF_CPU_READY;
    }
    return 1;
}


static int sqmd_source_job_ready(sqmdCpuSearchState *st,
                                 SqmdSharedControl *ctrl,
                                 const SqmdSourceJob *job) {
    int ordinal;

    for (ordinal = 0; ordinal < job->shard_count; ordinal++) {
        const SqmdSourceShardRef *ref = &job->shard[ordinal];
        SqmdLaneBuffer *buf = &ctrl->lanes[ref->lane].buf[ref->buffer];

        if (buf->state == SQMD_BUF_DSP_ABORT) {
            if (sqmd_cpu_state_capture_dsp_abort(st, ctrl))
                return st->abort_rc;
            return SQMD_STATUS_BAD_PARAM;
        }

        if (buf->turn_id != job->source_sequence) {
            sqmd_cpu_state_record_abort(st,
                                        SQMD_ABORT_APPLY_HEADER,
                                        SQMD_STATUS_BAD_PARAM,
                                        job->source_sequence,
                                        ref->lane,
                                        ref->buffer,
                                        -1,
                                        buf,
                                        -1,
                                        job->source_id);
            return SQMD_STATUS_BAD_PARAM;
        }
        if (buf->state == SQMD_BUF_CPU_READY)
            return 0;
        if (buf->state != SQMD_BUF_DSP_DONE) {
            sqmd_cpu_state_record_abort(st,
                                        SQMD_ABORT_APPLY_HEADER,
                                        SQMD_STATUS_BAD_PARAM,
                                        job->source_sequence,
                                        ref->lane,
                                        ref->buffer,
                                        -1,
                                        buf,
                                        -1,
                                        job->source_id);
            return SQMD_STATUS_BAD_PARAM;
        }
    }

    DB_READ_BARRIER();
    for (ordinal = 0; ordinal < job->shard_count; ordinal++) {
        const SqmdSourceShardRef *ref = &job->shard[ordinal];
        SqmdLaneBuffer *buf = &ctrl->lanes[ref->lane].buf[ref->buffer];
        int i;

        if (buf->state == SQMD_BUF_DSP_ABORT) {
            if (sqmd_cpu_state_capture_dsp_abort(st, ctrl))
                return st->abort_rc;
            return SQMD_STATUS_BAD_PARAM;
        }

        if (buf->state != SQMD_BUF_DSP_DONE ||
            buf->turn_id != job->source_sequence ||
            buf->work_count != ref->expected_count ||
            buf->result_count != ref->expected_count) {
            sqmd_cpu_state_record_abort(st,
                                        SQMD_ABORT_APPLY_HEADER,
                                        SQMD_STATUS_BAD_PARAM,
                                        job->source_sequence,
                                        ref->lane,
                                        ref->buffer,
                                        -1,
                                        buf,
                                        -1,
                                        job->source_id);
            return SQMD_STATUS_BAD_PARAM;
        }

        for (i = 0; i < ref->expected_count; i++) {
            int expected_id = job->accepted_ids[ref->accepted_offset + i];

            if (buf->work[i].neighbor_id != expected_id ||
                buf->work[i].source_id != job->source_id ||
                buf->result[i].neighbor_id != expected_id ||
                buf->result[i].source_id != job->source_id) {
                sqmd_cpu_state_record_abort(
                    st,
                    SQMD_ABORT_APPLY_RESULT,
                    SQMD_STATUS_BAD_PARAM,
                    job->source_sequence,
                    ref->lane,
                    ref->buffer,
                    i,
                    buf,
                    buf->result[i].neighbor_id,
                    buf->result[i].source_id);
                return SQMD_STATUS_BAD_PARAM;
            }
        }
    }
    return 1;
}

static int sqmd_source_job_compute_residual_delta(
        SqmdSourceJob *job,
        SqmdResidualDeltaFn residual_delta_fn,
        void *residual_delta_ctx) {
    int i;

    if (!job || !residual_delta_fn)
        return SQMD_STATUS_BAD_PARAM;
    for (i = 0; i < job->accepted_count; i++) {
        float delta = residual_delta_fn(job->accepted_ids[i],
                                        job->source_id,
                                        residual_delta_ctx);
        if (!isfinite(delta))
            return SQMD_STATUS_BAD_PARAM;
        job->residual_delta[i] = delta;
    }
    job->residual_delta_ready = 1;
    return SQMD_STATUS_OK;
}

static int sqmd_source_job_commit(sqmdCpuSearchState *st,
                                  SqmdSharedControl *ctrl,
                                  const SqmdSourceJob *job,
                                  int captured_query_gen,
                                  SqmdCorrectionFn correction_fn,
                                  void *correction_ctx,
                                  SqmdTimeFn time_fn,
                                  void *time_ctx) {
    float corrected_distances[SQMD_SOURCE_ACCEPTED_MAX];
    uint64_t correction_start = time_fn ? time_fn(time_ctx) : 0u;
    int ordinal;

    if (!correction_fn)
        correction_fn = sqmd_default_correction;


    for (ordinal = 0; ordinal < job->shard_count; ordinal++) {
        const SqmdSourceShardRef *ref = &job->shard[ordinal];
        SqmdLaneBuffer *buf = &ctrl->lanes[ref->lane].buf[ref->buffer];
        int i;

        for (i = 0; i < ref->expected_count; i++) {
            const SqmdResultItem *result = &buf->result[i];
            float distance;

            if (job->residual_delta_ready) {
                distance = result->coarse_dist +
                    job->residual_delta[ref->accepted_offset + i];
                if (distance < 0.0f && distance > -1.0e-4f)
                    distance = 0.0f;
            } else {
                distance = correction_fn(result->neighbor_id,
                                         result->source_id,
                                         result->coarse_dist,
                                         correction_ctx);
            }

            if (!isfinite(distance) || distance < 0.0f) {
                sqmd_cpu_state_record_abort(st,
                                            SQMD_ABORT_APPLY_RESULT,
                                            SQMD_STATUS_BAD_PARAM,
                                            job->source_sequence,
                                            ref->lane,
                                            ref->buffer,
                                            i,
                                            buf,
                                            result->neighbor_id,
                                            result->source_id);
                return SQMD_STATUS_BAD_PARAM;
            }
            corrected_distances[ref->accepted_offset + i] = distance;
        }
    }


    if (sqmd_source_pipeline_check_protocol(st,
                                            ctrl,
                                            captured_query_gen,
                                            job->source_sequence,
                                            job->source_id) != SQMD_STATUS_OK)
        return SQMD_STATUS_BAD_PARAM;


    for (ordinal = 0; ordinal < job->shard_count; ordinal++) {
        const SqmdSourceShardRef *ref = &job->shard[ordinal];
        const SqmdLaneBuffer *buf =
            &ctrl->lanes[ref->lane].buf[ref->buffer];

        if (buf->state == SQMD_BUF_DSP_ABORT) {
            if (sqmd_cpu_state_capture_dsp_abort(st, ctrl))
                return st->abort_rc;
            return SQMD_STATUS_BAD_PARAM;
        }

        if (buf->state != SQMD_BUF_DSP_DONE ||
            buf->turn_id != job->source_sequence ||
            buf->work_count != ref->expected_count ||
            buf->result_count != ref->expected_count) {
            sqmd_cpu_state_record_abort(st,
                                        SQMD_ABORT_APPLY_HEADER,
                                        SQMD_STATUS_BAD_PARAM,
                                        job->source_sequence,
                                        ref->lane,
                                        ref->buffer,
                                        -1,
                                        buf,
                                        -1,
                                        job->source_id);
            return SQMD_STATUS_BAD_PARAM;
        }
    }


    for (ordinal = 0; ordinal < job->shard_count; ordinal++) {
        const SqmdSourceShardRef *ref = &job->shard[ordinal];
        int i;

        for (i = 0; i < ref->expected_count; i++) {
            int accepted_index = ref->accepted_offset + i;
            int id = job->accepted_ids[accepted_index];
            float distance = corrected_distances[accepted_index];

            if (sqmd_candidate_insert(st->pool,
                                      &st->pool_size,
                                      st->pool_capacity,
                                      id,
                                      distance)) {
                sqmd_frontier_push(st->frontier,
                                   &st->frontier_size,
                                   st->frontier_capacity,
                                   id,
                                   distance);
            }
            st->exact_corrected++;
        }
    }

    for (ordinal = 0; ordinal < job->shard_count; ordinal++) {
        const SqmdSourceShardRef *ref = &job->shard[ordinal];
        SqmdLaneBuffer *buf = &ctrl->lanes[ref->lane].buf[ref->buffer];

        st->dsp_done_batches++;
        buf->work_count = 0;
        buf->result_count = 0;
    }
    DB_WRITE_BARRIER();
    for (ordinal = 0; ordinal < job->shard_count; ordinal++) {
        const SqmdSourceShardRef *ref = &job->shard[ordinal];

        ctrl->lanes[ref->lane].buf[ref->buffer].state = SQMD_BUF_EMPTY;
    }

    if (time_fn)
        ctrl->stats.sqmd_cpu_correct_us +=
            time_fn(time_ctx) - correction_start;
    if (job->accepted_count == 0)
        st->empty_turns++;
    st->source_jobs_retired++;
    return SQMD_STATUS_OK;
}

static void sqmd_source_pipeline_merge_stats(SqmdSharedStats *stats,
                                             const sqmdCpuSearchState *st) {
    if (!stats || !st)
        return;

    stats->sqmd_work_items += st->work_items;
    stats->sqmd_dedup_skips += st->dedup_skips;
    stats->sqmd_empty_turns += st->empty_turns;
    stats->sqmd_lane_wait_us += st->lane_wait_us;
    stats->sqmd_dsp_done_batches += st->dsp_done_batches;
    stats->sqmd_source_jobs_created += st->source_jobs_created;
    stats->sqmd_source_jobs_retired += st->source_jobs_retired;
    stats->sqmd_speculative_sources += st->speculative_sources;
    stats->sqmd_source_window_starvation += st->source_window_starvation;
    if (stats->sqmd_inflight_sources_highwater <
        (uint64_t)st->inflight_sources_highwater) {
        stats->sqmd_inflight_sources_highwater =
            (uint64_t)st->inflight_sources_highwater;
    }
}

static int sqmd_cpu_state_run_source_pipeline_impl(sqmdCpuSearchState *st,
                                       SqmdSharedControl *ctrl,
                                       int max_inflight_sources,
                                       const int32_t *graph,
                                       int graph_degree,
                                       int num_vectors,
                                       SqmdCorrectionFn correction_fn,
                                       void *correction_ctx,
                                       SqmdResidualDeltaFn residual_delta_fn,
                                       void *residual_delta_ctx,
                                       SqmdWaitFn wait_fn,
                                       void *wait_ctx,
                                       SqmdTimeFn time_fn,
                                       void *time_ctx,
                                       int *out_ids,
                                       float *out_dists,
                                       int out_cap) {
    SqmdSourceJob jobs[SQMD_SOURCE_WINDOW_MAX];
    int64_t next_sequence = 1;
    uint64_t head_wait_start = 0u;
    int head_wait_sequence = 0;
    int captured_query_gen = 0;
    int starvation_active = 0;
    int job_count = 0;
    int out_count;
    int i;


    if (max_inflight_sources == 1 && !residual_delta_fn) {
        return sqmd_cpu_state_run_shared_control(st,
                                                 ctrl,
                                                 graph,
                                                 graph_degree,
                                                 num_vectors,
                                                 correction_fn,
                                                 correction_ctx,
                                                 wait_fn,
                                                 wait_ctx,
                                                 time_fn,
                                                 time_ctx,
                                                 out_ids,
                                                 out_dists,
                                                 out_cap);
    }

    memset(jobs, 0, sizeof(jobs));
    if (st)
        sqmd_cpu_state_clear_abort(st);

    if (!st || !ctrl || !graph || graph_degree <= 0 || num_vectors <= 0 ||
        !out_ids || !out_dists || out_cap <= 0 ||
        (max_inflight_sources != 1 && max_inflight_sources != 2) ||
        ctrl->lane_count <= 0 || ctrl->lane_count > SQMD_DSP_LANES) {
        sqmd_cpu_state_record_abort(st,
                                    SQMD_ABORT_DISPATCH,
                                    SQMD_STATUS_BAD_PARAM,
                                    0,
                                    -1,
                                    -1,
                                    -1,
                                    NULL,
                                    -1,
                                    -1);
        if (ctrl) {
            DB_WRITE_BARRIER();
            ctrl->ctrl_state = SQMD_CTRL_ABORT;
            DB_WRITE_BARRIER();
        }
        return -1;
    }

    st->finished = 0;
    captured_query_gen = ctrl->query_slot.query_gen;

    while (!st->finished) {
        int frontier_empty = 0;

        if (sqmd_source_pipeline_check_protocol(
                st,
                ctrl,
                captured_query_gen,
                job_count > 0 ? jobs[0].source_sequence : ctrl->turn_id,
                job_count > 0 ? jobs[0].source_id : -1) != SQMD_STATUS_OK) {
            goto abort;
        }

        while (job_count < max_inflight_sources) {
            int admitted;
            int delta_rc = SQMD_STATUS_OK;
            uint64_t delta_start = 0u;

            if (sqmd_source_pipeline_check_protocol(
                    st,
                    ctrl,
                    captured_query_gen,
                    job_count > 0
                        ? jobs[0].source_sequence : ctrl->turn_id,
                    job_count > 0
                        ? jobs[0].source_id : -1) != SQMD_STATUS_OK) {
                goto abort;
            }

            if (next_sequence > INT_MAX) {
                sqmd_cpu_state_record_abort(st,
                                            SQMD_ABORT_DISPATCH,
                                            SQMD_STATUS_BAD_PARAM,
                                            INT_MAX,
                                            -1,
                                            -1,
                                            -1,
                                            NULL,
                                            -1,
                                            -1);
                goto abort;
            }

            admitted = sqmd_source_pipeline_admit_one(
                st,
                ctrl,
                graph,
                graph_degree,
                num_vectors,
                (int)next_sequence,
                captured_query_gen,
                job_count,
                &jobs[job_count]);
            if (admitted < 0) {
                if (st->abort_stage == SQMD_ABORT_NONE) {
                    sqmd_cpu_state_record_abort(st,
                                                SQMD_ABORT_DISPATCH,
                                                admitted,
                                                (int)next_sequence,
                                                -1,
                                                -1,
                                                -1,
                                                NULL,
                                                -1,
                                                -1);
                }
                goto abort;
            }
            if (admitted == 0) {
                frontier_empty = 1;
                break;
            }

            if (residual_delta_fn) {
                delta_start = time_fn ? time_fn(time_ctx) : 0u;
                delta_rc = sqmd_source_job_compute_residual_delta(
                    &jobs[job_count], residual_delta_fn,
                    residual_delta_ctx);
                if (time_fn)
                    ctrl->stats.sqmd_cpu_correct_us +=
                        time_fn(time_ctx) - delta_start;
            }
            if (delta_rc != SQMD_STATUS_OK) {
                sqmd_cpu_state_record_abort(st,
                                            SQMD_ABORT_APPLY_RESULT,
                                            SQMD_STATUS_BAD_PARAM,
                                            (int)next_sequence,
                                            -1,
                                            -1,
                                            -1,
                                            NULL,
                                            -1,
                                            jobs[job_count].source_id);
                goto abort;
            }

            next_sequence++;
            job_count++;
            starvation_active = 0;
        }

        if (job_count == 0) {
            if (!frontier_empty) {
                int ignored_source = -1;
                int peek_rc = sqmd_cpu_state_try_peek_source(st,
                                                             &ignored_source);
                if (peek_rc < 0)
                    goto abort;
                frontier_empty = (peek_rc == 0);
            }
            if (frontier_empty) {
                if (sqmd_source_pipeline_check_protocol(
                        st,
                        ctrl,
                        captured_query_gen,
                        ctrl->turn_id,
                        -1) != SQMD_STATUS_OK) {
                    goto abort;
                }
                if (!sqmd_shared_control_buffers_empty(ctrl)) {
                    sqmd_cpu_state_record_abort(st,
                                                SQMD_ABORT_COLLECT,
                                                SQMD_STATUS_BAD_PARAM,
                                                ctrl->turn_id,
                                                -1,
                                                -1,
                                                -1,
                                                NULL,
                                                -1,
                                                -1);
                    goto abort;
                }
                st->finished = 1;
                break;
            }
            continue;
        }

        if (job_count < max_inflight_sources && frontier_empty) {
            if (!starvation_active) {
                st->source_window_starvation++;
                starvation_active = 1;
            }
        } else {
            starvation_active = 0;
        }

        {
            int ready = sqmd_source_job_ready(st, ctrl, &jobs[0]);

            if (ready < 0)
                goto abort;
            if (ready > 0) {
                if (sqmd_source_pipeline_check_protocol(
                        st,
                        ctrl,
                        captured_query_gen,
                        jobs[0].source_sequence,
                        jobs[0].source_id) != SQMD_STATUS_OK) {
                    goto abort;
                }
                int commit_rc = sqmd_source_job_commit(st,
                                                       ctrl,
                                                       &jobs[0],
                                                       captured_query_gen,
                                                       correction_fn,
                                                       correction_ctx,
                                                       time_fn,
                                                       time_ctx);
                if (commit_rc != SQMD_STATUS_OK)
                    goto abort;

                if (job_count == 2)
                    jobs[0] = jobs[1];
                memset(&jobs[job_count - 1], 0, sizeof(jobs[0]));
                job_count--;
                head_wait_sequence = 0;
                head_wait_start = 0u;
                continue;
            }
        }

        if (sqmd_source_pipeline_check_protocol(st,
                                                ctrl,
                                                captured_query_gen,
                                                jobs[0].source_sequence,
                                                jobs[0].source_id) !=
            SQMD_STATUS_OK) {
            goto abort;
        }

        if (time_fn && head_wait_sequence != jobs[0].source_sequence) {
            head_wait_sequence = jobs[0].source_sequence;
            head_wait_start = time_fn(time_ctx);
        }
        if (time_fn && SQMD_TURN_WAIT_TIMEOUT_US > 0u &&
            time_fn(time_ctx) - head_wait_start >
                SQMD_TURN_WAIT_TIMEOUT_US) {
            sqmd_cpu_state_record_abort(st,
                                        SQMD_ABORT_TURN_WAIT_TIMEOUT,
                                        SQMD_STATUS_TURN_TIMEOUT,
                                        jobs[0].source_sequence,
                                        -1,
                                        -1,
                                        -1,
                                        NULL,
                                        -1,
                                        jobs[0].source_id);
            goto abort;
        }

        if (wait_fn) {
            uint64_t wait_start = time_fn ? time_fn(time_ctx) : 0u;

            wait_fn(wait_ctx);
            if (time_fn)
                ctrl->stats.sqmd_cpu_wait_dsp_us +=
                    time_fn(time_ctx) - wait_start;
        }
    }

    if (sqmd_source_pipeline_check_protocol(st,
                                            ctrl,
                                            captured_query_gen,
                                            ctrl->turn_id,
                                            -1) != SQMD_STATUS_OK)
        goto abort;

    sqmd_heap_sort(st->pool, st->pool_size);
    out_count = (st->pool_size < out_cap) ? st->pool_size : out_cap;
    for (i = 0; i < out_count; i++) {
        out_ids[i] = st->pool[i].id;
        out_dists[i] = st->pool[i].distance;
    }

    if (sqmd_source_pipeline_check_protocol(st,
                                            ctrl,
                                            captured_query_gen,
                                            ctrl->turn_id,
                                            -1) != SQMD_STATUS_OK)
        goto abort;

    sqmd_source_pipeline_merge_stats(&ctrl->stats, st);

    DB_WRITE_BARRIER();
    ctrl->ctrl_state = SQMD_CTRL_DONE;
    DB_WRITE_BARRIER();
    return out_count;

abort:
    st->finished = 0;
    sqmd_source_pipeline_merge_stats(&ctrl->stats, st);
    ctrl->stats.sqmd_abort_count++;
    if (ctrl->stats.sqmd_abort_stage == SQMD_ABORT_NONE ||
        st->abort_stage != SQMD_ABORT_CTRL) {
        sqmd_stats_copy_abort_from_cpu(&ctrl->stats, st);
    }
    DB_WRITE_BARRIER();
    ctrl->ctrl_state = SQMD_CTRL_ABORT;
    DB_WRITE_BARRIER();
    return -1;
}


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
                                       int out_cap) {
    return sqmd_cpu_state_run_source_pipeline_impl(
        st, ctrl, max_inflight_sources, graph, graph_degree, num_vectors,
        correction_fn, correction_ctx, NULL, NULL, wait_fn, wait_ctx,
        time_fn, time_ctx, out_ids, out_dists, out_cap);
}

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
        int out_cap) {
    if (!residual_delta_fn)
        return -1;
    return sqmd_cpu_state_run_source_pipeline_impl(
        st, ctrl, max_inflight_sources, graph, graph_degree, num_vectors,
        NULL, NULL, residual_delta_fn, residual_delta_ctx, wait_fn, wait_ctx,
        time_fn, time_ctx, out_ids, out_dists, out_cap);
}

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
                                int out_cap) {
    sqmdCpuSearchState st;
    SqmdSharedControl ctrl;
    int turn_id = 1;
    int ret = -1;
    int out_count = 0;

    if (!graph || graph_degree <= 0 || num_vectors <= 0 ||
        !seed_ids || !seed_dists || seed_count <= 0 ||
        lane_count <= 0 || lane_count > SQMD_DSP_LANES ||
        !distance_fn || !out_ids || !out_dists || out_cap <= 0)
        return -1;

    sqmd_shared_control_init(&ctrl, lane_count);
    if (sqmd_cpu_state_init(&st,
                            num_vectors,
                            pool_capacity,
                            entry_points,
                            expand_count,
                            NULL,
                            NULL,
                            NULL) != SQMD_STATUS_OK)
        return -1;
    if (sqmd_cpu_state_seed(&st, seed_ids, seed_dists, seed_count) !=
        SQMD_STATUS_OK)
        goto cleanup;

    while (!st.finished) {
        SqmdDispatchSummary summary;
        int source_id = -1;
        int lane;
        int dispatched;

        dispatched = sqmd_shared_control_dispatch_next_turn(&st,
                                                            &ctrl,
                                                            graph,
                                                            graph_degree,
                                                            num_vectors,
                                                            turn_id,
                                                            &summary,
                                                            &source_id);
        if (dispatched < 0)
            goto cleanup;
        if (dispatched == 0)
            break;

        for (lane = 0; lane < lane_count; lane++) {
            int buf_idx;
            for (buf_idx = 0; buf_idx < SQMD_BUFFERS_PER_LANE; buf_idx++) {
                SqmdLaneBuffer *buf = &ctrl.lanes[lane].buf[buf_idx];

                if (buf->state != SQMD_BUF_CPU_READY)
                    continue;
                if (sqmd_process_lane_buffer(buf,
                                             turn_id,
                                             num_vectors,
                                             distance_fn,
                                             distance_ctx) != SQMD_STATUS_OK)
                    goto cleanup;
            }
        }
        if (sqmd_shared_control_collect_turn(&st,
                                             &ctrl,
                                             turn_id,
                                             correction_fn,
                                             correction_ctx) != 1)
            goto cleanup;
        turn_id++;
    }

    sqmd_heap_sort(st.pool, st.pool_size);
    out_count = (st.pool_size < out_cap) ? st.pool_size : out_cap;
    {
        int i;
        for (i = 0; i < out_count; i++) {
            out_ids[i] = st.pool[i].id;
            out_dists[i] = st.pool[i].distance;
        }
    }
    ret = out_count;

cleanup:
    sqmd_cpu_state_cleanup(&st);
    return ret;
}
