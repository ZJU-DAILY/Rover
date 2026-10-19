typedef struct RetrievalAsyncDemoTask {
    int in_use;
    int task_id;
    int status;
    int num_candidates;
    int dim;
    float scale;
    float zero_point;
    float query[RETRIEVAL_ASYNC_DEMO_MAX_DIM];
    int candidate_ids[RETRIEVAL_ASYNC_DEMO_MAX_BATCH];
    uint8_t compressed_batch[RETRIEVAL_ASYNC_DEMO_MAX_BATCH * RETRIEVAL_ASYNC_DEMO_MAX_DIM];
    float distances[RETRIEVAL_ASYNC_DEMO_MAX_BATCH];
} RetrievalAsyncDemoTask;

typedef struct RetrievalAsyncDemoService {
    int initialized;
    int shutdown;
    int next_task_id;
    int queue[RETRIEVAL_ASYNC_DEMO_MAX_TASKS];
    int queue_head;
    int queue_tail;
    int queue_count;
    qurt_mutex_t mutex;
    qurt_cond_t cond_nonempty;
    qurt_cond_t cond_nonfull;
    qurt_thread_t workers[RETRIEVAL_ASYNC_DEMO_NUM_WORKERS];
    void *worker_stacks[RETRIEVAL_ASYNC_DEMO_NUM_WORKERS];
    RetrievalAsyncDemoTask tasks[RETRIEVAL_ASYNC_DEMO_MAX_TASKS];
} RetrievalAsyncDemoService;

static RetrievalAsyncDemoService g_retrieval_async_demo_service;
static int g_retrieval_async_demo_open_count = 0;

static void retrieval_async_demo_task_reset(RetrievalAsyncDemoTask *task) {
    if (!task) {
        return;
    }
    task->in_use = 0;
    task->task_id = 0;
}

static int retrieval_async_demo_find_free_slot_locked(RetrievalAsyncDemoService *svc) {
    int i = 0;

    for (i = 0; i < RETRIEVAL_ASYNC_DEMO_MAX_TASKS; i++) {
        if (!svc->tasks[i].in_use) {
            return i;
        }
    }
    return -1;
}

static RetrievalAsyncDemoTask *retrieval_async_demo_find_task_locked(RetrievalAsyncDemoService *svc,
                                                                     int task_id,
                                                                     int *slot_idx) {
    int i = 0;

    for (i = 0; i < RETRIEVAL_ASYNC_DEMO_MAX_TASKS; i++) {
        if (svc->tasks[i].in_use && svc->tasks[i].task_id == task_id) {
            if (slot_idx) {
                *slot_idx = i;
            }
            return &svc->tasks[i];
        }
    }
    return NULL;
}

static void retrieval_async_demo_worker(void *arg) {
    RetrievalAsyncDemoService *svc = &g_retrieval_async_demo_service;
    int worker_id = (int)(uintptr_t)arg;
    uint8_t query_u8[RETRIEVAL_ASYNC_DEMO_MAX_DIM];

    (void)worker_id;
    FARF(HIGH, "retrieval_async_demo worker %d started", worker_id);

    while (1) {
        int slot_idx = -1;
        RetrievalAsyncDemoTask *task = NULL;

        qurt_mutex_lock(&svc->mutex);
        while (!svc->shutdown && svc->queue_count == 0) {
            qurt_cond_wait(&svc->cond_nonempty, &svc->mutex);
        }

        if (svc->shutdown) {
            qurt_mutex_unlock(&svc->mutex);
            break;
        }

        slot_idx = svc->queue[svc->queue_head];
        svc->queue_head = (svc->queue_head + 1) % RETRIEVAL_ASYNC_DEMO_MAX_TASKS;
        svc->queue_count--;
        qurt_cond_signal(&svc->cond_nonfull);

        task = &svc->tasks[slot_idx];
        task->status = RETRIEVAL_ASYNC_DEMO_STATUS_RUNNING;
        qurt_mutex_unlock(&svc->mutex);

        if (retrieval_quantize_query_u8(task->query,
                                        task->dim,
                                        task->scale,
                                        task->zero_point,
                                        query_u8) != 0) {
            qurt_mutex_lock(&svc->mutex);
            task->status = RETRIEVAL_ASYNC_DEMO_STATUS_FAILED;
            qurt_mutex_unlock(&svc->mutex);
            continue;
        }

        if (retrieval_compute_int8_native_distance_batch(
                query_u8,
                task->compressed_batch,
                task->dim,
                task->num_candidates,
                task->scale,
                task->distances) != 0) {
            qurt_mutex_lock(&svc->mutex);
            task->status = RETRIEVAL_ASYNC_DEMO_STATUS_FAILED;
            qurt_mutex_unlock(&svc->mutex);
            continue;
        }

        qurt_mutex_lock(&svc->mutex);
        task->status = RETRIEVAL_ASYNC_DEMO_STATUS_DONE;
        qurt_mutex_unlock(&svc->mutex);
    }

    FARF(HIGH, "retrieval_async_demo worker %d stopped", worker_id);
    qurt_thread_exit(QURT_EOK);
}

static int retrieval_async_demo_service_destroy(void) {
    RetrievalAsyncDemoService *svc = &g_retrieval_async_demo_service;
    int i = 0;

    if (!svc->initialized) {
        return AEE_SUCCESS;
    }

    qurt_mutex_lock(&svc->mutex);
    svc->shutdown = 1;
    qurt_cond_broadcast(&svc->cond_nonempty);
    qurt_cond_broadcast(&svc->cond_nonfull);
    qurt_mutex_unlock(&svc->mutex);

    for (i = 0; i < RETRIEVAL_ASYNC_DEMO_NUM_WORKERS; i++) {
        int status = 0;
        if (svc->worker_stacks[i]) {
            if (svc->workers[i] != 0) {
                qurt_thread_join(svc->workers[i], &status);
            }
            free(svc->worker_stacks[i]);
            svc->worker_stacks[i] = NULL;
        }
    }

    qurt_cond_destroy(&svc->cond_nonempty);
    qurt_cond_destroy(&svc->cond_nonfull);
    qurt_mutex_destroy(&svc->mutex);
    memset(svc, 0, sizeof(*svc));
    return AEE_SUCCESS;
}

static int retrieval_async_demo_service_init(void) {
    RetrievalAsyncDemoService *svc = &g_retrieval_async_demo_service;
    int i = 0;

    if (svc->initialized) {
        return AEE_SUCCESS;
    }

    memset(svc, 0, sizeof(*svc));
    qurt_mutex_init(&svc->mutex);
    qurt_cond_init(&svc->cond_nonempty);
    qurt_cond_init(&svc->cond_nonfull);
    svc->initialized = 1;
    svc->next_task_id = 1;

    for (i = 0; i < RETRIEVAL_ASYNC_DEMO_NUM_WORKERS; i++) {
        qurt_thread_attr_t attr;
        char name[24];
        int ret = 0;

        svc->worker_stacks[i] = malloc(RETRIEVAL_ASYNC_DEMO_STACK_SIZE);
        if (!svc->worker_stacks[i]) {
            retrieval_async_demo_service_destroy();
            return AEE_ENOMEMORY;
        }

        qurt_thread_attr_init(&attr);
        snprintf(name, sizeof(name), "retr_demo_%d", i);
        qurt_thread_attr_set_name(&attr, name);
        qurt_thread_attr_set_stack_addr(&attr, svc->worker_stacks[i]);
        qurt_thread_attr_set_stack_size(&attr, RETRIEVAL_ASYNC_DEMO_STACK_SIZE);
        qurt_thread_attr_set_priority(&attr, QURT_THREAD_ATTR_PRIORITY_DEFAULT / 2);

        ret = qurt_thread_create(&svc->workers[i],
                                 &attr,
                                 retrieval_async_demo_worker,
                                 (void *)(uintptr_t)i);
        if (ret != QURT_EOK) {
            svc->workers[i] = 0;
            retrieval_async_demo_service_destroy();
            return AEE_EFAILED;
        }
    }

    FARF(HIGH, "retrieval_async_demo service initialized with %d workers",
         RETRIEVAL_ASYNC_DEMO_NUM_WORKERS);
    return AEE_SUCCESS;
}

int calculator_retrieval_async_submit(remote_handle64 h,
                                      const float *query,
                                      int queryLen,
                                      const int *candidate_ids,
                                      int candidate_idsLen,
                                      const uint8_t *compressed_batch,
                                      int compressed_batchLen,
                                      int32 dim,
                                      int32 num_candidates,
                                      float scale,
                                      float zero_point,
                                      int32 *task_id) {
    RetrievalAsyncDemoService *svc = &g_retrieval_async_demo_service;
    RetrievalAsyncDemoTask *task = NULL;
    int slot_idx = -1;
    int batch_bytes = 0;

    (void)h;

    if (!query || !candidate_ids || !compressed_batch || !task_id) {
        return AEE_EBADPARM;
    }
    if (dim <= 0 || dim > RETRIEVAL_ASYNC_DEMO_MAX_DIM) {
        return AEE_EBADPARM;
    }
    if (num_candidates <= 0 || num_candidates > RETRIEVAL_ASYNC_DEMO_MAX_BATCH) {
        return AEE_EBADPARM;
    }
    if (queryLen < dim || candidate_idsLen < num_candidates) {
        return AEE_EBADPARM;
    }

    batch_bytes = num_candidates * dim;
    if (compressed_batchLen < batch_bytes) {
        return AEE_EBADPARM;
    }

    if (retrieval_async_demo_service_init() != AEE_SUCCESS) {
        return AEE_EFAILED;
    }

    qurt_mutex_lock(&svc->mutex);
    while (!svc->shutdown &&
           (svc->queue_count == RETRIEVAL_ASYNC_DEMO_MAX_TASKS ||
            retrieval_async_demo_find_free_slot_locked(svc) < 0)) {
        qurt_cond_wait(&svc->cond_nonfull, &svc->mutex);
    }

    if (svc->shutdown) {
        qurt_mutex_unlock(&svc->mutex);
        return AEE_EFAILED;
    }

    slot_idx = retrieval_async_demo_find_free_slot_locked(svc);
    if (slot_idx < 0) {
        qurt_mutex_unlock(&svc->mutex);
        return AEE_EFAILED;
    }

    task = &svc->tasks[slot_idx];
    memset(task, 0, sizeof(*task));
    task->in_use = 1;
    task->task_id = svc->next_task_id++;
    task->status = RETRIEVAL_ASYNC_DEMO_STATUS_PENDING;
    task->num_candidates = (int)num_candidates;
    task->dim = (int)dim;
    task->scale = scale;
    task->zero_point = zero_point;
    memcpy(task->query, query, (size_t)dim * sizeof(float));
    memcpy(task->candidate_ids,
           candidate_ids,
           (size_t)num_candidates * sizeof(int));
    memcpy(task->compressed_batch,
           compressed_batch,
           (size_t)batch_bytes);

    svc->queue[svc->queue_tail] = slot_idx;
    svc->queue_tail = (svc->queue_tail + 1) % RETRIEVAL_ASYNC_DEMO_MAX_TASKS;
    svc->queue_count++;
    *task_id = task->task_id;

    qurt_cond_signal(&svc->cond_nonempty);
    qurt_mutex_unlock(&svc->mutex);
    return AEE_SUCCESS;
}

int calculator_retrieval_async_poll(remote_handle64 h,
                                    int32 task_id,
                                    int32 *status,
                                    int *result_ids,
                                    int result_idsLen,
                                    float *distances,
                                    int distancesLen,
                                    int32 *num_results) {
    RetrievalAsyncDemoService *svc = &g_retrieval_async_demo_service;
    RetrievalAsyncDemoTask *task = NULL;
    int slot_idx = -1;
    int copy_count = 0;
    int i = 0;

    (void)h;

    if (!status || !num_results) {
        return AEE_EBADPARM;
    }
    *status = RETRIEVAL_ASYNC_DEMO_STATUS_NOT_FOUND;
    *num_results = 0;

    if (!svc->initialized) {
        return AEE_SUCCESS;
    }

    qurt_mutex_lock(&svc->mutex);
    task = retrieval_async_demo_find_task_locked(svc, (int)task_id, &slot_idx);
    if (!task) {
        qurt_mutex_unlock(&svc->mutex);
        return AEE_SUCCESS;
    }

    *status = task->status;
    if (task->status == RETRIEVAL_ASYNC_DEMO_STATUS_DONE) {
        copy_count = task->num_candidates;
        if (!result_ids || !distances ||
            result_idsLen < copy_count || distancesLen < copy_count) {
            qurt_mutex_unlock(&svc->mutex);
            return AEE_EBADPARM;
        }

        for (i = 0; i < copy_count; i++) {
            result_ids[i] = task->candidate_ids[i];
            distances[i] = task->distances[i];
        }
        *num_results = copy_count;
        retrieval_async_demo_task_reset(task);
        qurt_cond_signal(&svc->cond_nonfull);
    }
    qurt_mutex_unlock(&svc->mutex);
    return AEE_SUCCESS;
}

int calculator_retrieval_async_reset(remote_handle64 h) {
    (void)h;
    return retrieval_async_demo_service_destroy();
}
