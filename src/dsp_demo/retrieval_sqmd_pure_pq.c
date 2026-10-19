#include "dsp_demo/retrieval_sqmd_pure_pq.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "dsp_demo/retrieval_pq_core.h"

typedef struct PurePqCandidate {
    int id;
    float distance;
} PurePqCandidate;

struct PurePqCpuSearchState {
    PurePqCandidate *pool;
    PurePqCandidate *frontier;
    uint32_t *visited;
    int pool_size;
    int pool_capacity;
    int frontier_size;
    int num_vectors;
    int vector_capacity;
    int expand_count;
    PurePqDistancePolicy distance_policy;
    RetrievalGraphReader graph_reader;
    uint64_t corrected_candidates;
};

static int pure_pq_validate_distance_policy(
        const PurePqDistancePolicy *policy) {
    if (!policy || policy->mode == PURE_PQ_CORRECTION_OFF) {
        return 0;
    }
    return policy->mode == PURE_PQ_CORRECTION_ONLINE && policy->evaluator
               ? 0 : -1;
}

static int pure_pq_evaluate_distance(const PurePqDistancePolicy *policy,
                                     int vector_id,
                                     float coarse_distance,
                                     float *out_distance) {
    float distance = coarse_distance;
    if (!out_distance || !isfinite(coarse_distance) ||
        pure_pq_validate_distance_policy(policy) != 0) {
        return -1;
    }
    if (policy && policy->mode == PURE_PQ_CORRECTION_ONLINE &&
        (policy->evaluator(policy->context, vector_id,
                           coarse_distance, &distance) != 0 ||
         !isfinite(distance))) {
        return -1;
    }
    *out_distance = distance;
    return 0;
}

static int pure_pq_graph_row_count(const int32_t *neighbors,
                                   int slot_count,
                                   int num_vectors) {
    int count = slot_count;
    int index;

    if (!neighbors || slot_count < 0 || num_vectors <= 0) {
        return -1;
    }
    for (index = 0; index < slot_count; index++) {
        int id = neighbors[index];

        if (id == -1) {
            int suffix;

            count = index;
            for (suffix = index + 1; suffix < slot_count; suffix++) {
                if (neighbors[suffix] != -1) {
                    return -1;
                }
            }
            break;
        }
        if (id < 0 || id >= num_vectors) {
            return -1;
        }
    }
    return count;
}

static void pure_pq_memory_barrier(void) {
#ifdef __hexagon__
    __asm__ __volatile__("barrier" ::: "memory");
#else
    __sync_synchronize();
#endif
}

static int pure_pq_order(float left_distance, int left_id,
                         float right_distance, int right_id) {
    if (left_distance < right_distance) {
        return -1;
    }
    if (left_distance > right_distance) {
        return 1;
    }
    if (left_id < right_id) {
        return -1;
    }
    if (left_id > right_id) {
        return 1;
    }
    return 0;
}

static int pure_pq_candidate_compare(const void *left, const void *right) {
    const PurePqCandidate *left_candidate = (const PurePqCandidate *)left;
    const PurePqCandidate *right_candidate = (const PurePqCandidate *)right;
    return pure_pq_order(left_candidate->distance, left_candidate->id,
                         right_candidate->distance, right_candidate->id);
}

static void pure_pq_pool_sift_up(PurePqCandidate *pool, int position) {
    PurePqCandidate candidate = pool[position];

    while (position > 0) {
        int parent = (position - 1) / 2;
        if (pure_pq_order(candidate.distance, candidate.id,
                          pool[parent].distance, pool[parent].id) <= 0) {
            break;
        }
        pool[position] = pool[parent];
        position = parent;
    }
    pool[position] = candidate;
}

static void pure_pq_pool_sift_down(PurePqCandidate *pool, int size,
                                   int position) {
    PurePqCandidate candidate = pool[position];

    for (;;) {
        int child = position * 2 + 1;
        if (child >= size) {
            break;
        }
        if (child + 1 < size &&
            pure_pq_order(pool[child + 1].distance, pool[child + 1].id,
                          pool[child].distance, pool[child].id) > 0) {
            child++;
        }
        if (pure_pq_order(candidate.distance, candidate.id,
                          pool[child].distance, pool[child].id) >= 0) {
            break;
        }
        pool[position] = pool[child];
        position = child;
    }
    pool[position] = candidate;
}

static int pure_pq_pool_insert(PurePqCandidate *pool, int *size, int capacity,
                               int id, float distance) {
    if (*size < capacity) {
        pool[*size].id = id;
        pool[*size].distance = distance;
        pure_pq_pool_sift_up(pool, *size);
        (*size)++;
        return 1;
    }
    if (pure_pq_order(distance, id, pool[0].distance, pool[0].id) < 0) {
        pool[0].id = id;
        pool[0].distance = distance;
        pure_pq_pool_sift_down(pool, *size, 0);
        return 1;
    }
    return 0;
}

static void pure_pq_frontier_sift_up(PurePqCandidate *frontier,
                                      int position) {
    PurePqCandidate candidate = frontier[position];

    while (position > 0) {
        int parent = (position - 1) / 2;
        if (pure_pq_order(candidate.distance, candidate.id,
                          frontier[parent].distance, frontier[parent].id) >= 0) {
            break;
        }
        frontier[position] = frontier[parent];
        position = parent;
    }
    frontier[position] = candidate;
}

static void pure_pq_frontier_sift_down(PurePqCandidate *frontier, int size,
                                        int position) {
    PurePqCandidate candidate = frontier[position];

    for (;;) {
        int child = position * 2 + 1;
        if (child >= size) {
            break;
        }
        if (child + 1 < size &&
            pure_pq_order(frontier[child + 1].distance,
                          frontier[child + 1].id,
                          frontier[child].distance,
                          frontier[child].id) < 0) {
            child++;
        }
        if (pure_pq_order(candidate.distance, candidate.id,
                          frontier[child].distance,
                          frontier[child].id) <= 0) {
            break;
        }
        frontier[position] = frontier[child];
        position = child;
    }
    frontier[position] = candidate;
}

static void pure_pq_frontier_push(PurePqCandidate *frontier, int *size,
                                  int id, float distance) {
    frontier[*size].id = id;
    frontier[*size].distance = distance;
    pure_pq_frontier_sift_up(frontier, *size);
    (*size)++;
}

static PurePqCandidate pure_pq_frontier_pop(PurePqCandidate *frontier,
                                             int *size) {
    PurePqCandidate result = frontier[0];
    (*size)--;
    if (*size > 0) {
        frontier[0] = frontier[*size];
        pure_pq_frontier_sift_down(frontier, *size, 0);
    }
    return result;
}

static int pure_pq_pool_contains(const PurePqCandidate *pool, int size,
                                 int id, float distance) {
    int index;
    for (index = 0; index < size; index++) {
        if (pool[index].id == id && pool[index].distance == distance) {
            return 1;
        }
    }
    return 0;
}

static int pure_pq_validate_search(const PurePqSearchIndexView *index,
                                   const PurePqSearchRequest *request,
                                   int *out_ids, float *out_distances,
                                   int out_capacity) {
    int chunk;

    if (!index || !request || !out_ids || !out_distances ||
        out_capacity <= 0 || !index->graph || index->graph_degree <= 0 ||
        index->num_vectors <= 0 || !index->code_chunks ||
        !index->chunk_vector_counts || index->chunk_count <= 0 ||
        index->chunk_vectors <= 0 || !index->codebook || index->dim <= 0 ||
        index->m <= 0 || (index->ks != 16 && index->ks != 256) ||
        index->subdim <= 0 ||
        index->m * index->subdim != index->dim || !request->query ||
        !request->entry_ids || request->entry_count <= 0 ||
        request->expand_count <= 0 || request->pool_capacity <= 0 ||
        request->num_out <= 0 || request->num_out > request->pool_capacity) {
        return -1;
    }
    for (chunk = 0; chunk < index->chunk_count; chunk++) {
        if (!index->code_chunks[chunk] ||
            index->chunk_vector_counts[chunk] <= 0 ||
            index->chunk_vector_counts[chunk] > index->chunk_vectors) {
            return -1;
        }
    }
    return 0;
}

void pure_pq_sqmd_control_init(PurePqSqmdSharedControl *control,
                               int lane_count) {
    int lane;
    int buffer_index;

    if (!control) {
        return;
    }
    if (lane_count < 1) {
        lane_count = 1;
    }
    if (lane_count > PURE_PQ_SQMD_MAX_LANES) {
        lane_count = PURE_PQ_SQMD_MAX_LANES;
    }
    memset(control, 0, sizeof(*control));
    control->control_state = PURE_PQ_SQMD_CONTROL_IDLE;
    control->lane_count = lane_count;
    for (lane = 0; lane < PURE_PQ_SQMD_MAX_LANES; lane++) {
        for (buffer_index = 0;
             buffer_index < PURE_PQ_SQMD_BUFFERS_PER_LANE;
             buffer_index++) {
            control->lanes[lane].buffers[buffer_index].state =
                PURE_PQ_SQMD_BUFFER_EMPTY;
        }
        control->lanes[lane].dsp_active_buffer = -1;
        control->lanes[lane].cpu_fill_buffer = 0;
    }
}

int pure_pq_sqmd_process_buffer_payload(PurePqSqmdLaneBuffer *buffer,
                                int expected_turn_id,
                                int expected_query_gen,
                                const PurePqSqmdAdcContext *context) {
    int32_t ids[PURE_PQ_SQMD_MAX_WORK_ITEMS];
    float distances[PURE_PQ_SQMD_MAX_WORK_ITEMS];
    int item;

    if (!buffer || !context || !context->code_chunks ||
        !context->chunk_vector_counts || !context->index_uuid ||
        !context->lut_header || !context->lut || context->chunk_count <= 0 ||
        context->chunk_vectors <= 0 || context->num_vectors <= 0 ||
        context->m <= 0 ||
        (context->ks != 16 && context->ks != 256) ||
        buffer->state != PURE_PQ_SQMD_BUFFER_CPU_READY ||
        buffer->turn_id != expected_turn_id ||
        buffer->query_gen != expected_query_gen ||
        buffer->work_count < 0 ||
        buffer->work_count > PURE_PQ_SQMD_MAX_WORK_ITEMS ||
        context->lut_header->query_gen != expected_query_gen ||
        context->lut_header->lut_gen != expected_query_gen ||
        context->lut_header->m != context->m ||
        context->lut_header->ks != context->ks ||
        context->lut_header->lut_count != context->m * context->ks ||
        memcmp(context->lut_header->index_uuid,
               context->index_uuid, 16) != 0) {
        return -1;
    }
    for (item = 0; item < buffer->work_count; item++) {
        if (buffer->work[item].neighbor_id < 0 ||
            buffer->work[item].neighbor_id >= context->num_vectors ||
            buffer->work[item].source_id < 0 ||
            buffer->work[item].source_id >= context->num_vectors) {
            return -1;
        }
        ids[item] = buffer->work[item].neighbor_id;
    }
    if (context->use_hvx) {
        if (retrieval_pq_adc_ids_batch_hvx(
                context->code_chunks,
                context->chunk_vector_counts,
                context->chunk_count,
                context->chunk_vectors,
                ids, buffer->work_count,
                context->lut, context->m, context->ks,
                context->hvx_gather_buffer, distances) != 0) {
            return -1;
        }
    } else if (retrieval_pq_adc_ids_batch_scalar(
                   context->code_chunks,
                   context->chunk_vector_counts,
                   context->chunk_count,
                   context->chunk_vectors,
                   ids, buffer->work_count,
                   context->lut, context->m, context->ks,
                   distances) != 0) {
        return -1;
    }
    for (item = 0; item < buffer->work_count; item++) {
        if (!isfinite(distances[item])) {
            return -1;
        }
        buffer->result[item].neighbor_id = buffer->work[item].neighbor_id;
        buffer->result[item].source_id = buffer->work[item].source_id;
        buffer->result[item].distance = distances[item];
    }
    buffer->result_count = buffer->work_count;
    return 0;
}


int pure_pq_sqmd_process_buffer(PurePqSqmdLaneBuffer *buffer,
                                int expected_turn_id,
                                int expected_query_gen,
                                const PurePqSqmdAdcContext *context) {
    int rc = pure_pq_sqmd_process_buffer_payload(buffer, expected_turn_id,
                                                expected_query_gen, context);
    if (rc == 0) {
        pure_pq_memory_barrier();
        buffer->state = PURE_PQ_SQMD_BUFFER_DSP_DONE;
    }
    return rc;
}

int pure_pq_cpu_search_create(int num_vectors,
                              int pool_capacity,
                              int expand_count,
                              PurePqCpuSearchState **out_state) {
    PurePqCpuSearchState *state;
    size_t visited_words;

    if (!out_state || num_vectors <= 0 || pool_capacity <= 0 ||
        expand_count <= 0) {
        return -1;
    }
    *out_state = NULL;
    state = (PurePqCpuSearchState *)calloc(1, sizeof(*state));
    if (!state) {
        return -1;
    }
    visited_words = ((size_t)num_vectors + 31u) / 32u;
    state->pool = (PurePqCandidate *)malloc(
        (size_t)pool_capacity * sizeof(*state->pool));
    state->frontier = (PurePqCandidate *)malloc(
        (size_t)num_vectors * sizeof(*state->frontier));
    state->visited = (uint32_t *)calloc(visited_words,
                                        sizeof(*state->visited));
    if (!state->pool || !state->frontier || !state->visited) {
        pure_pq_cpu_search_destroy(state);
        return -1;
    }
    state->pool_capacity = pool_capacity;
    state->num_vectors = num_vectors;
    state->vector_capacity = num_vectors;
    state->expand_count = expand_count;
    *out_state = state;
    return 0;
}

int pure_pq_cpu_search_reset(PurePqCpuSearchState *state, int num_vectors,
                              int pool_capacity, int expand_count) {
    if (!state || num_vectors<=0 || num_vectors>state->vector_capacity ||
        state->pool_capacity != pool_capacity || state->expand_count != expand_count)
        return -1;
    memset(state->visited, 0, (((size_t)num_vectors + 31u) / 32u) * sizeof(uint32_t));
    state->num_vectors = num_vectors;
    state->pool_size = 0; state->frontier_size = 0;
    state->corrected_candidates = 0;
    memset(&state->distance_policy, 0, sizeof(state->distance_policy));
    memset(&state->graph_reader,0,sizeof(state->graph_reader));
    return 0;
}

int pure_pq_cpu_search_seed(PurePqCpuSearchState *state,
                            const int *ids,
                            const float *distances,
                            int count) {
    int index;
    if (!state || !ids || !distances || count <= 0) {
        return -1;
    }
    for (index = 0; index < count; index++) {
        int id = ids[index];
        float distance;
        if (id < 0 || id >= state->num_vectors ||
            !isfinite(distances[index])) {
            return -1;
        }
        if (state->visited[id >> 5] & (1u << (id & 31))) {
            continue;
        }
        state->visited[id >> 5] |= 1u << (id & 31);
        if (pure_pq_evaluate_distance(&state->distance_policy, id,
                                      distances[index], &distance) != 0) {
            return -1;
        }
        if (state->distance_policy.mode == PURE_PQ_CORRECTION_ONLINE) {
            state->corrected_candidates++;
        }
        if (pure_pq_pool_insert(state->pool, &state->pool_size,
                                state->pool_capacity, id, distance)) {
            pure_pq_frontier_push(state->frontier, &state->frontier_size,
                                  id, distance);
        }
    }
    return state->pool_size > 0 ? 0 : -1;
}

int pure_pq_cpu_search_set_distance_policy(
        PurePqCpuSearchState *state,
        const PurePqDistancePolicy *policy) {
    if (!state || pure_pq_validate_distance_policy(policy) != 0 ||
        state->pool_size != 0 || state->frontier_size != 0) {
        return -1;
    }
    memset(&state->distance_policy, 0, sizeof(state->distance_policy));
    if (policy) {
        state->distance_policy = *policy;
    }
    return 0;
}

int pure_pq_cpu_search_set_graph_reader(PurePqCpuSearchState *state,
                                       const RetrievalGraphReader *reader) {
    if (!state || state->pool_size || state->frontier_size) return -1;
    memset(&state->graph_reader, 0, sizeof(state->graph_reader));
    if (reader) state->graph_reader = *reader;
    return 0;
}

static int pure_pq_find_empty_buffer(PurePqSqmdLane *lane) {
    int offset;
    int start = lane->cpu_fill_buffer;
    if (start < 0 || start >= PURE_PQ_SQMD_BUFFERS_PER_LANE) {
        start = 0;
    }
    for (offset = 0; offset < PURE_PQ_SQMD_BUFFERS_PER_LANE; offset++) {
        int index = (start + offset) % PURE_PQ_SQMD_BUFFERS_PER_LANE;
        if (lane->buffers[index].state == PURE_PQ_SQMD_BUFFER_EMPTY) {
            return index;
        }
    }
    return -1;
}

int pure_pq_cpu_search_dispatch(PurePqCpuSearchState *state,
                                const int32_t *graph,
                                int graph_degree,
                                PurePqSqmdSharedControl *control,
                                int turn_id,
                                int query_gen,
                                PurePqSqmdDispatchSummary *summary) {
    PurePqCandidate source;
    const int32_t *neighbors;
    int32_t row_snapshot[PURE_PQ_SQMD_MAX_LANES * PURE_PQ_SQMD_MAX_WORK_ITEMS];
    PurePqSqmdLaneBuffer *buffers[PURE_PQ_SQMD_MAX_LANES];
    int buffer_indices[PURE_PQ_SQMD_MAX_LANES];
    int expand_count;
    int valid_neighbor_count;
    int lane;
    int neighbor_index;
    int next_lane = 0;
    int found_source = 0;

    if (!state || !graph || graph_degree <= 0 || !control || !summary ||
        control->lane_count <= 0 ||
        control->lane_count > PURE_PQ_SQMD_MAX_LANES || turn_id <= 0 ||
        query_gen <= 0) {
        return -1;
    }
    memset(summary, 0, sizeof(*summary));
    summary->source_id = -1;
    while (state->frontier_size > 0) {
        if (state->pool_size == state->pool_capacity &&
            pure_pq_order(state->frontier[0].distance,
                          state->frontier[0].id,
                          state->pool[0].distance,
                          state->pool[0].id) > 0) {
            return 0;
        }
        source = pure_pq_frontier_pop(state->frontier,
                                      &state->frontier_size);
        if (pure_pq_pool_contains(state->pool, state->pool_size,
                                  source.id, source.distance)) {
            found_source = 1;
            break;
        }
    }
    if (!found_source) {
        return 0;
    }
    summary->source_id = source.id;
    neighbors = graph + (size_t)source.id * (size_t)graph_degree;
    expand_count = state->expand_count < graph_degree
                       ? state->expand_count : graph_degree;
    if (state->graph_reader.read) {
        if (expand_count > (int)(sizeof(row_snapshot)/sizeof(row_snapshot[0])) ||
            state->graph_reader.read(state->graph_reader.ctx, source.id,
                                     row_snapshot, expand_count) != 0) return -1;
        neighbors = row_snapshot;
    }
    valid_neighbor_count = pure_pq_graph_row_count(neighbors,
                                                    expand_count,
                                                    state->num_vectors);
    if (valid_neighbor_count < 0) {
        return -1;
    }
    for (lane = 0; lane < control->lane_count; lane++) {
        int buffer_index = pure_pq_find_empty_buffer(&control->lanes[lane]);
        if (buffer_index < 0) {
            return -1;
        }
        buffer_indices[lane] = buffer_index;
        buffers[lane] = &control->lanes[lane].buffers[buffer_index];
        memset(buffers[lane], 0, sizeof(*buffers[lane]));
        buffers[lane]->state = PURE_PQ_SQMD_BUFFER_EMPTY;
        buffers[lane]->turn_id = turn_id;
        buffers[lane]->query_gen = query_gen;
    }
    for (neighbor_index = 0;
         neighbor_index < valid_neighbor_count;
         neighbor_index++) {
        int id = neighbors[neighbor_index];
        PurePqSqmdLaneBuffer *buffer;
        int item;
        if (state->visited[id >> 5] & (1u << (id & 31))) {
            summary->dedup_skips++;
            continue;
        }
        state->visited[id >> 5] |= 1u << (id & 31);
        buffer = buffers[next_lane];
        item = buffer->work_count++;
        if (item >= PURE_PQ_SQMD_MAX_WORK_ITEMS) {
            return -1;
        }
        buffer->work[item].neighbor_id = id;
        buffer->work[item].source_id = source.id;
        summary->work_items++;
        next_lane = (next_lane + 1) % control->lane_count;
    }
    control->turn_id = turn_id;
    control->query_gen = query_gen;
    pure_pq_memory_barrier();
    for (lane = 0; lane < control->lane_count; lane++) {
        if (buffers[lane]->work_count > 0) {
            control->lanes[lane].cpu_fill_buffer =
                (buffer_indices[lane] + 1) % PURE_PQ_SQMD_BUFFERS_PER_LANE;
            buffers[lane]->state = PURE_PQ_SQMD_BUFFER_CPU_READY;
            summary->buffers_submitted++;
        }
    }
    summary->empty_turn = summary->work_items == 0;
    return 1;
}

int pure_pq_cpu_search_collect(PurePqCpuSearchState *state,
                               PurePqSqmdSharedControl *control,
                               int turn_id,
                               int query_gen) {
    PurePqCandidate results[PURE_PQ_SQMD_MAX_LANES *
                            PURE_PQ_SQMD_MAX_WORK_ITEMS];
    PurePqSqmdLaneBuffer *completed[PURE_PQ_SQMD_MAX_LANES];
    int completed_count = 0;
    int result_count = 0;
    int lane;
    int buffer_index;
    int index;

    if (!state || !control || turn_id <= 0 || query_gen <= 0 ||
        control->lane_count <= 0 || control->lane_count > PURE_PQ_SQMD_MAX_LANES) {
        return -1;
    }

    for (lane = 0; lane < control->lane_count; lane++) {
        for (buffer_index = 0; buffer_index < PURE_PQ_SQMD_BUFFERS_PER_LANE; buffer_index++) {
            PurePqSqmdLaneBuffer *buffer = &control->lanes[lane].buffers[buffer_index];
            if (buffer->turn_id != turn_id || buffer->work_count == 0) continue;
            if (buffer->query_gen != query_gen) return -1;
            if (buffer->state == PURE_PQ_SQMD_BUFFER_CPU_READY) return 0;
            if (buffer->state != PURE_PQ_SQMD_BUFFER_DSP_DONE ||
                buffer->work_count < 0 ||
                buffer->work_count > PURE_PQ_SQMD_MAX_WORK_ITEMS ||
                buffer->result_count != buffer->work_count) return -1;
        }
    }
    for (lane = 0; lane < control->lane_count; lane++) {
        for (buffer_index = 0;
             buffer_index < PURE_PQ_SQMD_BUFFERS_PER_LANE;
             buffer_index++) {
            PurePqSqmdLaneBuffer *buffer =
                &control->lanes[lane].buffers[buffer_index];
            if (buffer->turn_id != turn_id || buffer->work_count == 0) {
                continue;
            }
            if (buffer->state == PURE_PQ_SQMD_BUFFER_CPU_READY) {
                return 0;
            }
            if (buffer->state != PURE_PQ_SQMD_BUFFER_DSP_DONE ||
                buffer->query_gen != query_gen ||
                buffer->result_count != buffer->work_count ||
                buffer->result_count < 0 ||
                buffer->result_count > PURE_PQ_SQMD_MAX_WORK_ITEMS) {
                return -1;
            }
            if (completed_count >= PURE_PQ_SQMD_MAX_LANES) return -1;
            completed[completed_count++] = buffer;
            for (index = 0; index < buffer->result_count; index++) {
                PurePqSqmdResultItem *item = &buffer->result[index];
                if (item->neighbor_id != buffer->work[index].neighbor_id ||
                    item->source_id != buffer->work[index].source_id ||
                    item->neighbor_id < 0 ||
                    item->neighbor_id >= state->num_vectors ||
                    item->source_id < 0 ||
                    item->source_id >= state->num_vectors ||
                    !isfinite(item->distance)) {
                    return -1;
                }
                results[result_count].id = item->neighbor_id;
                if (pure_pq_evaluate_distance(
                        &state->distance_policy, item->neighbor_id,
                        item->distance,
                        &results[result_count].distance) != 0) {
                    return -1;
                }
                if (state->distance_policy.mode ==
                    PURE_PQ_CORRECTION_ONLINE) {
                    state->corrected_candidates++;
                }
                result_count++;
            }
        }
    }
    qsort(results, (size_t)result_count, sizeof(results[0]),
          pure_pq_candidate_compare);
    for (index = 0; index < result_count; index++) {
        if (pure_pq_pool_insert(state->pool, &state->pool_size,
                                state->pool_capacity,
                                results[index].id,
                                results[index].distance)) {
            pure_pq_frontier_push(state->frontier, &state->frontier_size,
                                  results[index].id,
                                  results[index].distance);
        }
    }
    for (index = 0; index < completed_count; index++) {
        completed[index]->work_count = 0;
        completed[index]->result_count = 0;
        pure_pq_memory_barrier();
        completed[index]->state = PURE_PQ_SQMD_BUFFER_EMPTY;
    }
    return 1;
}

int pure_pq_source_pipeline_init(PurePqSourcePipeline *pipeline, int window) {
    if (!pipeline || window < 1 || window > PURE_PQ_SQMD_BUFFERS_PER_LANE) return -1;
    memset(pipeline, 0, sizeof(*pipeline));
    pipeline->window = window;
    pipeline->next_turn = 1;
    return 0;
}


int pure_pq_source_pipeline_step(PurePqSourcePipeline *pipeline,
                                 PurePqCpuSearchState *state,
                                 const int32_t *graph, int graph_degree,
                                 PurePqSqmdSharedControl *control,
                                 int query_gen) {
    if (!pipeline || !state || !control || pipeline->window < 1 ||
        pipeline->window > PURE_PQ_SQMD_BUFFERS_PER_LANE) return -1;
    while (pipeline->pending < pipeline->window) {
        PurePqSqmdDispatchSummary summary;
        int rc = pure_pq_cpu_search_dispatch(state, graph, graph_degree, control,
                                             pipeline->next_turn, query_gen, &summary);
        if (rc < 0) return -1;
        if (rc == 0) break;
        pipeline->turns[(pipeline->head + pipeline->pending) %
                         PURE_PQ_SQMD_BUFFERS_PER_LANE] = pipeline->next_turn++;
        pipeline->pending++;
        pipeline->jobs++;
        pipeline->work_items += (uint64_t)summary.work_items;
        if (pipeline->pending > pipeline->high_water) pipeline->high_water = pipeline->pending;
    }
    if (!pipeline->pending) return 1;
    {
        int rc = pure_pq_cpu_search_collect(state, control,
                         pipeline->turns[pipeline->head], query_gen);
        if (rc < 0) return -1;
        if (rc == 0) return 0;
    }
    pipeline->head = (pipeline->head + 1) % PURE_PQ_SQMD_BUFFERS_PER_LANE;
    pipeline->pending--;
    pipeline->retired++;
    return 0;
}

int pure_pq_cpu_search_finish(PurePqCpuSearchState *state,
                              int *out_ids,
                              float *out_distances,
                              int out_capacity) {
    int count;
    int index;
    if (!state || !out_ids || !out_distances || out_capacity <= 0) {
        return -1;
    }
    qsort(state->pool, (size_t)state->pool_size, sizeof(state->pool[0]),
          pure_pq_candidate_compare);
    count = state->pool_size < out_capacity ? state->pool_size : out_capacity;
    for (index = 0; index < count; index++) {
        out_ids[index] = state->pool[index].id;
        out_distances[index] = state->pool[index].distance;
    }
    return count;
}

void pure_pq_cpu_search_destroy(PurePqCpuSearchState *state) {
    if (!state) {
        return;
    }
    free(state->pool);
    free(state->frontier);
    free(state->visited);
    free(state);
}

int retrieval_pure_pq_search_scalar_with_policy(
        const PurePqSearchIndexView *index,
        const PurePqSearchRequest *request,
        const PurePqDistancePolicy *policy,
        int *out_ids,
        float *out_distances,
        int out_capacity,
        PurePqSearchStats *out_stats) {
    PurePqCandidate *pool = NULL;
    PurePqCandidate *frontier = NULL;
    uint32_t *visited = NULL;
    float *lut = NULL;
    int32_t *batch_ids = NULL;
    float *batch_distances = NULL;
    int visited_words;
    int pool_size = 0;
    int frontier_size = 0;
    int expand_count;
    PurePqSearchStats stats;
    int result_count = -1;
    int entry_index;

    memset(&stats, 0, sizeof(stats));
    if (out_stats) {
        memset(out_stats, 0, sizeof(*out_stats));
    }
    if (pure_pq_validate_search(index, request, out_ids, out_distances,
                                out_capacity) != 0 ||
        pure_pq_validate_distance_policy(policy) != 0) {
        return -1;
    }
    expand_count = request->expand_count < index->graph_degree
                       ? request->expand_count : index->graph_degree;
    visited_words = (index->num_vectors + 31) / 32;
    pool = (PurePqCandidate *)malloc(
        (size_t)request->pool_capacity * sizeof(*pool));
    frontier = (PurePqCandidate *)malloc(
        (size_t)index->num_vectors * sizeof(*frontier));
    visited = (uint32_t *)calloc((size_t)visited_words, sizeof(*visited));
    lut = (float *)malloc((size_t)index->m * (size_t)index->ks * sizeof(*lut));
    batch_ids = (int32_t *)malloc((size_t)expand_count * sizeof(*batch_ids));
    batch_distances = (float *)malloc(
        (size_t)expand_count * sizeof(*batch_distances));
    if (!pool || !frontier || !visited || !lut || !batch_ids ||
        !batch_distances ||
        retrieval_pq_build_lut(request->query, index->codebook,
                               index->m, index->ks, index->subdim, lut) != 0) {
        goto done;
    }

    for (entry_index = 0; entry_index < request->entry_count; entry_index++) {
        int id = request->entry_ids[entry_index];
        float distance;
        const int32_t score_id = id;
        if (id < 0 || id >= index->num_vectors) {
            goto done;
        }
        if (visited[id >> 5] & (1u << (id & 31))) {
            stats.visited_dedup_skips++;
            continue;
        }
        visited[id >> 5] |= 1u << (id & 31);
        if (retrieval_pq_adc_ids_batch(index->code_chunks,
                                       index->chunk_vector_counts,
                                       index->chunk_count,
                                       index->chunk_vectors,
                                       &score_id, 1, lut,
                                       index->m, index->ks, &distance) != 0 ||
            !isfinite(distance) ||
            pure_pq_evaluate_distance(policy, id, distance, &distance) != 0) {
            goto done;
        }
        stats.pq_scored_candidates++;
        if (policy && policy->mode == PURE_PQ_CORRECTION_ONLINE) {
            stats.corrected_candidates++;
        }
        if (pure_pq_pool_insert(pool, &pool_size, request->pool_capacity,
                                id, distance)) {
            pure_pq_frontier_push(frontier, &frontier_size, id, distance);
        }
    }

    while (frontier_size > 0) {
        PurePqCandidate source;
        const int32_t *neighbors;
        int batch_count = 0;
        int valid_neighbor_count;
        int neighbor_index;

        if (pool_size == request->pool_capacity &&
            pure_pq_order(frontier[0].distance, frontier[0].id,
                          pool[0].distance, pool[0].id) > 0) {
            break;
        }
        source = pure_pq_frontier_pop(frontier, &frontier_size);
        if (!pure_pq_pool_contains(pool, pool_size,
                                   source.id, source.distance)) {
            continue;
        }
        neighbors = index->graph +
            (size_t)source.id * (size_t)index->graph_degree;
        valid_neighbor_count = pure_pq_graph_row_count(neighbors,
                                                        expand_count,
                                                        index->num_vectors);
        if (valid_neighbor_count < 0) {
            goto done;
        }
        for (neighbor_index = 0; neighbor_index < valid_neighbor_count;
             neighbor_index++) {
            int id = neighbors[neighbor_index];
            if (visited[id >> 5] & (1u << (id & 31))) {
                stats.visited_dedup_skips++;
                continue;
            }
            visited[id >> 5] |= 1u << (id & 31);
            batch_ids[batch_count++] = id;
        }
        stats.expanded_nodes++;
        if (batch_count == 0) {
            continue;
        }
        if (retrieval_pq_adc_ids_batch(index->code_chunks,
                                       index->chunk_vector_counts,
                                       index->chunk_count,
                                       index->chunk_vectors,
                                       batch_ids, batch_count, lut,
                                       index->m, index->ks,
                                       batch_distances) != 0) {
            goto done;
        }
        stats.pq_scored_candidates += (uint64_t)batch_count;
        for (neighbor_index = 0; neighbor_index < batch_count;
             neighbor_index++) {
            int id = batch_ids[neighbor_index];
            float distance = batch_distances[neighbor_index];
            if (!isfinite(distance) ||
                pure_pq_evaluate_distance(policy, id, distance,
                                          &distance) != 0) {
                goto done;
            }
            if (policy && policy->mode == PURE_PQ_CORRECTION_ONLINE) {
                stats.corrected_candidates++;
            }
            if (pure_pq_pool_insert(pool, &pool_size,
                                    request->pool_capacity,
                                    id, distance)) {
                pure_pq_frontier_push(frontier, &frontier_size,
                                      id, distance);
            }
        }
    }

    qsort(pool, (size_t)pool_size, sizeof(*pool), pure_pq_candidate_compare);
    result_count = request->num_out;
    if (result_count > pool_size) {
        result_count = pool_size;
    }
    if (result_count > out_capacity) {
        result_count = out_capacity;
    }
    for (entry_index = 0; entry_index < result_count; entry_index++) {
        out_ids[entry_index] = pool[entry_index].id;
        out_distances[entry_index] = pool[entry_index].distance;
    }
    if (out_stats) {
        *out_stats = stats;
    }

done:
    free(pool);
    free(frontier);
    free(visited);
    free(lut);
    free(batch_ids);
    free(batch_distances);
    return result_count;
}

int retrieval_pure_pq_search_scalar(const PurePqSearchIndexView *index,
                                    const PurePqSearchRequest *request,
                                    int *out_ids,
                                    float *out_distances,
                                    int out_capacity,
                                    PurePqSearchStats *out_stats) {
    return retrieval_pure_pq_search_scalar_with_policy(
        index, request, NULL, out_ids, out_distances,
        out_capacity, out_stats);
}
