typedef struct SbBacklinkPool SbBacklinkPool;
typedef struct {
    SbBacklinkPool *pool;
    int neighbor, old_degree, rc;
    int *old_ids;
    RoverInsertBuildRuntime runtime;
    RoverInsertBuildStats stats;
} SbBacklinkJob;
typedef struct { SbBacklinkPool *pool; int worker; } SbBacklinkWorker;
struct SbBacklinkPool {
    pthread_mutex_t mutex;
    pthread_cond_t work, drained;
    int mutex_ready, work_ready, drained_ready, created;
    int threads, capacity, dim, stop, pending, count, new_id;
    uint64_t generation;
    pthread_t *tids;
    SbBacklinkWorker *workers;
    SbBacklinkJob *jobs;
    const float *base;
    RoverInsertBuildConfig cfg;
    RetrievalRnnGraph *graph;
    uint64_t vector_bytes;

    int vector_capacity, vector_count, map_capacity;
    int *vector_ids, *vector_slots;
    float *vectors;
    uint64_t old_requests, old_decodes, old_hits;
};

static int sb_backlink_vector_slot(const SbBacklinkPool *pool, int id) {
    unsigned mask = (unsigned)pool->map_capacity - 1;
    unsigned slot = ((uint32_t)id * 2654435761u) & mask;
    while (pool->vector_ids[slot] != -1 && pool->vector_ids[slot] != id)
        slot = (slot + 1) & mask;
    return (int)slot;
}

static int sb_backlink_cached_read(void *ctx, int id, float *out) {
    SbBacklinkJob *job = ctx;
    SbBacklinkPool *pool = job->pool;
    if (id < 0 || id > pool->new_id) return -1;
    int slot = sb_backlink_vector_slot(pool, id);
    if (pool->vector_ids[slot] == id) {
        memcpy(out, pool->vectors + (size_t)pool->vector_slots[slot] * pool->dim,
               (size_t)pool->dim * sizeof(float));
        return 0;
    }

    return -1;
}

static void sb_backlink_compute(SbBacklinkPool *pool, int worker) {
    for (int i = worker; i < pool->count; i += pool->threads) {
        SbBacklinkJob *job = &pool->jobs[i];
        RoverInsertBuildCandidate item = {
            .id = pool->new_id, .search_dist = 0.0f, .exact_dist = FLT_MAX
        };
        job->stats.backlink_rows_considered = 1;
        job->rc = sb_write_pruned_row(pool->base, pool->graph->num_vectors,
            pool->dim, &pool->cfg, &job->runtime, pool->graph, pool->graph,
            job->neighbor, &item, 1, &job->stats);
        if (!job->rc && !sb_row_ids_equal(pool->graph, job->neighbor,
                                         job->old_ids, job->old_degree))
            job->stats.backlink_rows_changed = 1;
    }
}

static void *sb_backlink_worker(void *ctx) {
    SbBacklinkWorker *worker = ctx;
    SbBacklinkPool *pool = worker->pool;
    uint64_t seen = 0;
    pthread_mutex_lock(&pool->mutex);
    for (;;) {
        while (!pool->stop && seen == pool->generation)
            pthread_cond_wait(&pool->work, &pool->mutex);
        if (pool->stop) break;
        seen = pool->generation;
        pthread_mutex_unlock(&pool->mutex);
        sb_backlink_compute(pool, worker->worker);
        pthread_mutex_lock(&pool->mutex);
        if (--pool->pending == 0) pthread_cond_signal(&pool->drained);
    }
    pthread_mutex_unlock(&pool->mutex);
    return NULL;
}

static void sb_backlink_pool_close(SbBacklinkPool *pool) {
    if (!pool) return;
    if (pool->created) {
        pthread_mutex_lock(&pool->mutex);
        pool->stop = 1;
        pthread_cond_broadcast(&pool->work);
        pthread_mutex_unlock(&pool->mutex);
        for (int i = 0; i < pool->created; ++i)
            pthread_join(pool->tids[i], NULL);
    }
    if (pool->drained_ready) pthread_cond_destroy(&pool->drained);
    if (pool->work_ready) pthread_cond_destroy(&pool->work);
    if (pool->mutex_ready) pthread_mutex_destroy(&pool->mutex);
    if (pool->jobs) for (int i = 0; i < pool->capacity; ++i) {
        free(pool->jobs[i].old_ids);
    }
    free(pool->vector_ids); free(pool->vector_slots); free(pool->vectors);
    free(pool->jobs); free(pool->tids); free(pool->workers); free(pool);
}

static int sb_backlink_pool_open(const RoverInsertBuildConfig *cfg,
    const RoverInsertBuildRuntime *runtime, int dim, SbBacklinkPool **out) {
    *out = NULL;
    if (runtime->backlink_threads < 0 || runtime->backlink_threads > 16) return -1;
    if (!runtime->backlink_threads) return 0;
    if (cfg->r <= 0 || cfg->r > 128 || dim <= 0) return -1;
    size_t row_bytes = (size_t)(cfg->r + 2) * (size_t)dim * sizeof(float);
    if (row_bytes > 256u * 1024u || row_bytes * (size_t)cfg->r > 16u * 1024u * 1024u)
        return -1;
    SbBacklinkPool *pool = calloc(1, sizeof(*pool));
    if (!pool) return -1;
    pool->threads = runtime->backlink_threads;
    pool->capacity = cfg->r; pool->dim = dim;
    pool->jobs = calloc((size_t)cfg->r, sizeof(*pool->jobs));
    pool->tids = calloc((size_t)pool->threads, sizeof(*pool->tids));
    pool->workers = calloc((size_t)pool->threads, sizeof(*pool->workers));
    if (!pool->jobs || !pool->tids || !pool->workers) goto fail;
    for (int i = 0; i < cfg->r; ++i) {
        SbBacklinkJob *job = &pool->jobs[i];
        job->pool = pool;
        job->old_ids = malloc((size_t)cfg->r * sizeof(int));
        if (!job->old_ids) goto fail;
    }

    pool->vector_capacity = cfg->r * (cfg->r + 1) + 1;
    pool->map_capacity = 1;
    while (pool->map_capacity < 2 * pool->vector_capacity) pool->map_capacity *= 2;
    if (runtime->read_vector) {
        pool->vector_bytes = (uint64_t)pool->vector_capacity * dim * sizeof(float);
        pool->vectors = malloc((size_t)pool->vector_bytes);
        pool->vector_ids = malloc((size_t)pool->map_capacity * sizeof(int));
        pool->vector_slots = malloc((size_t)pool->map_capacity * sizeof(int));
        if (!pool->vectors || !pool->vector_ids || !pool->vector_slots) goto fail;
    }
    if (pthread_mutex_init(&pool->mutex, NULL)) goto fail;
    pool->mutex_ready = 1;
    if (pthread_cond_init(&pool->work, NULL)) goto fail;
    pool->work_ready = 1;
    if (pthread_cond_init(&pool->drained, NULL)) goto fail;
    pool->drained_ready = 1;
    for (int i = 0; i < pool->threads - 1; ++i) {
        pool->workers[i] = (SbBacklinkWorker){pool, i + 1};
        if (pthread_create(&pool->tids[i], NULL, sb_backlink_worker, &pool->workers[i])) goto fail;
        ++pool->created;
    }
    *out = pool;
    return 0;
fail:
    sb_backlink_pool_close(pool);
    return -1;
}

static int sb_backlink_load_vector(SbBacklinkJob *job,
    const RoverInsertBuildRuntime *runtime, int id) {
    if (!runtime->read_vector) return 0;
    SbBacklinkPool *pool = job->pool;
    if (id < 0 || id > pool->new_id) return -1;
    int slot = sb_backlink_vector_slot(pool, id);
    if (id != pool->new_id) ++pool->old_requests;
    if (pool->vector_ids[slot] == id) {
        if (id != pool->new_id) ++pool->old_hits;
        return 0;
    }
    if (pool->vector_count >= pool->vector_capacity) return -1;
    if (runtime->read_vector(runtime->vector_ctx, id,
            pool->vectors + (size_t)pool->vector_count * pool->dim)) return -1;
    pool->vector_ids[slot] = id;
    pool->vector_slots[slot] = pool->vector_count++;
    if (id != pool->new_id) ++pool->old_decodes;
    return 0;
}


static int sb_backlink_pool_run(SbBacklinkPool *pool, const float *base,
    const RoverInsertBuildConfig *cfg, const RoverInsertBuildRuntime *runtime,
    RetrievalRnnGraph *graph, int new_id, const int *ids, int count,
    RoverInsertBuildStats *stats) {
    if (!pool || count < 0 || count > pool->capacity) return -1;
    for (int i = 0; i < count; ++i) {
        int u = ids[i];
        if (u < 0 || u >= new_id || graph->degrees[u] > pool->capacity) return -1;
        for (int j = 0; j < i; ++j) if (ids[j] == u) return -1;
        if (graph->degrees[u] + 1 < cfg->r || sb_row_contains(graph, u, new_id)) return 1;
    }
    uint64_t begin = sb_now_us();
    pool->base = base; pool->cfg = *cfg; pool->graph = graph;
    pool->new_id = new_id; pool->count = count;
    pool->vector_count = 0;
    pool->old_requests = pool->old_decodes = pool->old_hits = 0;
    if (runtime->read_vector)
        for (int i = 0; i < pool->map_capacity; ++i) pool->vector_ids[i] = -1;
    for (int i = 0; i < count; ++i) {
        SbBacklinkJob *job = &pool->jobs[i];
        job->neighbor = ids[i]; job->old_degree = graph->degrees[ids[i]];
        job->rc = 0;
        memset(&job->stats, 0, sizeof(job->stats));
        job->runtime = *runtime;
        if (runtime->read_vector) {
            job->runtime.read_vector = sb_backlink_cached_read;
            job->runtime.vector_ctx = job;
        }
        if (sb_backlink_load_vector(job, runtime, job->neighbor) ||
            sb_backlink_load_vector(job, runtime, new_id)) return -1;
        for (int j = 0; j < job->old_degree; ++j) {
            int v = graph->neighbors[sb_slot(graph, job->neighbor, j)];
            job->old_ids[j] = v;
            if (v >= 0 && v < graph->num_vectors && v != job->neighbor &&
                sb_backlink_load_vector(job, runtime, v)) return -1;
        }
    }
    uint64_t prepared = sb_now_us();
    pthread_mutex_lock(&pool->mutex);
    pool->pending = pool->threads - 1;
    ++pool->generation;
    pthread_cond_broadcast(&pool->work);
    pthread_mutex_unlock(&pool->mutex);

    sb_backlink_compute(pool, 0);
    pthread_mutex_lock(&pool->mutex);
    while (pool->pending) pthread_cond_wait(&pool->drained, &pool->mutex);
    pthread_mutex_unlock(&pool->mutex);
    uint64_t computed = sb_now_us();
    int rc = 0;
    for (int i = 0; i < count; ++i) {
        sb_stats_accumulate(stats, &pool->jobs[i].stats);
        if (pool->jobs[i].rc) rc = -1;
    }
    if (stats) {
        stats->backlink_prepare_us += prepared - begin;
        stats->backlink_parallel_us += computed - prepared;
        stats->backlink_vector_bytes = pool->vector_bytes;
        stats->backlink_old_requests += pool->old_requests;
        stats->backlink_old_decodes += pool->old_decodes;
        stats->backlink_old_hits += pool->old_hits;
    }
    return rc;
}
