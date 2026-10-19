#include "dsp_demo/retrieval_rnn_descent_core.h"
#include "dsp_demo/retrieval_search_build_core.h"

static void demo_db_buffer_reset(DbSharedBuffer *db_shared);
static void demo_db_buffer_destroy(DbSharedBuffer *db_shared, int domain_id);
static int demo_db_buffer_init(DbSharedBuffer *db_shared, int domain_id);
static void *db_correction_thread_fn(void *arg);

typedef struct DemoRnnDspDistanceCtx {
    remote_handle64 handle;
    const float *base;
    int dim;
    int fallback_on_error;
    pthread_mutex_t mu;
    uint64_t rpc_calls;
    uint64_t rpc_pairs;
    uint64_t rpc_us;
    uint64_t cpu_pairs;
    uint64_t fallback_calls;
    uint64_t fallback_pairs;
    int failed;
    int mu_ready;
} DemoRnnDspDistanceCtx;

typedef struct DemoSearchBuildDspDbCtx {
    remote_handle64 handle;
    int domain_id;
    const RetrievalDemoEncodedDataset *dataset;
    DbSharedBuffer *buffers;
    int *buffer_busy;
    int buffer_count;
    pthread_mutex_t mu;
    pthread_cond_t cond;
    int sync_ready;
    int *graph_stage;
    size_t graph_bytes;
    int graph_uses_rpcmem;
    int graph_fd;
    int graph_mapped;
    int l1_fd;
    int l1_mapped;
    int use_dma;
    int search_loaded;
    uint64_t load_calls;
    uint64_t load_us;
    uint64_t rpc_calls;
    uint64_t rpc_queries;
    uint64_t rpc_us;
    uint64_t buffer_wait_us;
    uint64_t corr_batches;
    uint64_t corr_candidates;
    uint64_t dsp_wait_cpu_done_us;
    uint64_t dsp_invalidate_us;
    uint64_t dsp_wait_poll_count;
} DemoSearchBuildDspDbCtx;

static uint32_t demo_graph_mix_u32(uint32_t h, uint32_t v) {
    h ^= v + 0x9e3779b9u + (h << 6) + (h >> 2);
    return h;
}

static uint32_t demo_search_build_cache_seed(void) {
    uint32_t h = SEARCH_BUILD_SEED;
    h = demo_graph_mix_u32(h, (uint32_t)RETRIEVAL_GRAPH_INDEX_SCHEME);
    h = demo_graph_mix_u32(h, (uint32_t)SEARCH_BUILD_ITERATIONS);
    h = demo_graph_mix_u32(h, (uint32_t)SEARCH_BUILD_R);
    h = demo_graph_mix_u32(h, (uint32_t)SEARCH_BUILD_CANDIDATES);
    h = demo_graph_mix_u32(h, (uint32_t)SEARCH_BUILD_ENTRY_POINTS);
    h = demo_graph_mix_u32(h, (uint32_t)SEARCH_BUILD_EXPAND_COUNT);
    h = demo_graph_mix_u32(h, (uint32_t)SEARCH_BUILD_POOL_SIZE);
    h = demo_graph_mix_u32(h, (uint32_t)SEARCH_BUILD_PROVIDER);
    h = demo_graph_mix_u32(h, (uint32_t)SEARCH_BUILD_PRUNE_MODE);
    h = demo_graph_mix_u32(h, (uint32_t)SEARCH_BUILD_BATCH_ALIGN);
    return h;
}

static int demo_load_fvecs_n_into(const char *path,
                                  int n,
                                  int dim,
                                  float *out) {
    FILE *fp = NULL;
    int i;

    if (!path || !out || n <= 0 || dim <= 0) {
        return -1;
    }

    fp = fopen(path, "rb");
    if (!fp) {
        return -1;
    }

    for (i = 0; i < n; i++) {
        if (demo_read_fvec_row(fp,
                               dim,
                               out + (size_t)i * (size_t)dim) != 0) {
            fclose(fp);
            return -1;
        }
    }

    fclose(fp);
    return 0;
}

static void demo_rnn_cpu_distance_batch(const float *base,
                                        int dim,
                                        const int32_t *ids_a,
                                        const int32_t *ids_b,
                                        int pair_count,
                                        float *out_dists) {
    int i;

    for (i = 0; i < pair_count; i++) {
        out_dists[i] = retrieval_rnn_l2_distance_cpu(base,
                                                     dim,
                                                     ids_a[i],
                                                     ids_b[i]);
    }
}

static int demo_rnn_dsp_distance_batch(void *ctx_ptr,
                                       const int32_t *ids_a,
                                       const int32_t *ids_b,
                                       int pair_count,
                                       float *out_dists) {
    DemoRnnDspDistanceCtx *ctx = (DemoRnnDspDistanceCtx *)ctx_ptr;
    int ret;
    unsigned long long t0;

    if (!ctx || !ctx->base || !ids_a || !ids_b || !out_dists ||
        pair_count <= 0) {
        return -1;
    }

    if (ctx->failed || pair_count < RNN_DSP_L2_MIN_BATCH) {
        demo_rnn_cpu_distance_batch(ctx->base,
                                    ctx->dim,
                                    ids_a,
                                    ids_b,
                                    pair_count,
                                    out_dists);
        ctx->cpu_pairs += (uint64_t)pair_count;
        return 0;
    }

    pthread_mutex_lock(&ctx->mu);
    t0 = now_us();
    ret = calculator_retrieval_rnn_l2_batch(ctx->handle,
                                            ids_a,
                                            pair_count,
                                            ids_b,
                                            pair_count,
                                            pair_count,
                                            out_dists,
                                            pair_count);
    ctx->rpc_us += now_us() - t0;
    if (ret == 0) {
        ctx->rpc_calls++;
        ctx->rpc_pairs += (uint64_t)pair_count;
        pthread_mutex_unlock(&ctx->mu);
        return 0;
    }
    ctx->failed = 1;
    pthread_mutex_unlock(&ctx->mu);

    if (!ctx->fallback_on_error) {
        printf("  ERROR: retrieval_rnn_l2_batch failed 0x%x\n", ret);
        fflush(stdout);
        return -1;
    }

    demo_rnn_cpu_distance_batch(ctx->base,
                                ctx->dim,
                                ids_a,
                                ids_b,
                                pair_count,
                                out_dists);
    ctx->cpu_pairs += (uint64_t)pair_count;
    ctx->fallback_calls++;
    ctx->fallback_pairs += (uint64_t)pair_count;
    return 0;
}

static void demo_rnn_dsp_distance_ctx_destroy(DemoRnnDspDistanceCtx *ctx) {
    if (ctx && ctx->mu_ready) {
        pthread_mutex_destroy(&ctx->mu);
        ctx->mu_ready = 0;
    }
}

static void demo_rnn_l2_provider_cleanup(remote_handle64 handle,
                                         int domain_id,
                                         float *base,
                                         int base_uses_rpcmem,
                                         int fd_base,
                                         int mapped,
                                         int loaded,
                                         size_t base_bytes,
                                         DemoRnnDspDistanceCtx *ctx) {
    if (loaded) {
        calculator_retrieval_rnn_l2_unload_dma(handle);
    }
    if (mapped && fd_base >= 0 && fastrpc_munmap) {
        fastrpc_munmap(domain_id, fd_base, base, base_bytes);
    }
    demo_free_maybe_shared(base, base_uses_rpcmem);
    demo_rnn_dsp_distance_ctx_destroy(ctx);
}

static const char *demo_search_build_provider_name(void) {
    if (SEARCH_BUILD_PROVIDER == 0) {
        return "exact";
    }
    if (SEARCH_BUILD_PROVIDER == 1) {
        return "hlos";
    }
    if (SEARCH_BUILD_PROVIDER == 2) {
        return "dsp_db";
    }
    return "dsp_hmx";
}

static int demo_search_build_dsp_db_ctx_init(
    DemoSearchBuildDspDbCtx *ctx,
    remote_handle64 handle,
    int domain_id,
    const RetrievalDemoEncodedDataset *dataset,
    int num_vectors,
    int graph_degree) {
    int i;
    size_t l1_bytes;

    if (!ctx || !dataset || !dataset->l1_codes ||
        dataset->num_vectors != num_vectors ||
        dataset->dim != DATA_DIM ||
        graph_degree <= 0 ||
        SEARCH_BUILD_THREADS <= 0 ||
        SEARCH_BUILD_THREADS > 4) {
        return -1;
    }

    memset(ctx, 0, sizeof(*ctx));
    ctx->handle = handle;
    ctx->domain_id = domain_id;
    ctx->dataset = dataset;
    ctx->buffer_count = SEARCH_BUILD_THREADS;
    ctx->graph_fd = -1;
    ctx->l1_fd = -1;

    if (pthread_mutex_init(&ctx->mu, NULL) != 0) {
        return -1;
    }
    if (pthread_cond_init(&ctx->cond, NULL) != 0) {
        pthread_mutex_destroy(&ctx->mu);
        return -1;
    }
    ctx->sync_ready = 1;

    ctx->buffers = (DbSharedBuffer *)calloc((size_t)ctx->buffer_count,
                                            sizeof(*ctx->buffers));
    ctx->buffer_busy = (int *)calloc((size_t)ctx->buffer_count,
                                     sizeof(*ctx->buffer_busy));
    ctx->graph_bytes = (size_t)num_vectors *
                       (size_t)graph_degree *
                       sizeof(int);
    ctx->graph_stage = (int *)demo_alloc_shared_buffer(ctx->graph_bytes,
                                                       "search_build_dsp_graph",
                                                       &ctx->graph_uses_rpcmem);
    if (!ctx->buffers || !ctx->buffer_busy || !ctx->graph_stage) {
        return -1;
    }

    for (i = 0; i < ctx->buffer_count; i++) {
        if (demo_db_buffer_init(&ctx->buffers[i], domain_id) != 0) {
            return -1;
        }
    }

    l1_bytes = (size_t)dataset->num_vectors *
               (size_t)dataset->l1_code_bytes;
    if (ctx->graph_uses_rpcmem &&
        dataset->l1_codes_uses_rpcmem &&
        fastrpc_mmap) {
        int mret;

        ctx->graph_fd = rpcmem_to_fd(ctx->graph_stage);
        ctx->l1_fd = demo_buf_to_fd(dataset->l1_codes);
        if (ctx->graph_fd >= 0 && ctx->l1_fd >= 0) {
            mret = fastrpc_mmap(domain_id,
                                ctx->graph_fd,
                                ctx->graph_stage,
                                0,
                                ctx->graph_bytes,
                                FASTRPC_MAP_FD);
            if (mret == 0) {
                ctx->graph_mapped = 1;
                mret = fastrpc_mmap(domain_id,
                                    ctx->l1_fd,
                                    (void *)dataset->l1_codes,
                                    0,
                                    l1_bytes,
                                    FASTRPC_MAP_FD);
                if (mret == 0) {
                    ctx->l1_mapped = 1;
                    ctx->use_dma = 1;
                } else {
                    printf("  WARN: search-build DSP fastrpc_mmap(l1) failed 0x%x, using copy load\n",
                           mret);
                    fflush(stdout);
                }
            } else {
                printf("  WARN: search-build DSP fastrpc_mmap(graph) failed 0x%x, using copy load\n",
                       mret);
                fflush(stdout);
            }
        }
    }

    if (!ctx->use_dma) {
        if (ctx->graph_mapped && ctx->graph_fd >= 0 && fastrpc_munmap) {
            fastrpc_munmap(domain_id,
                           ctx->graph_fd,
                           ctx->graph_stage,
                           ctx->graph_bytes);
        }
        if (ctx->l1_mapped && ctx->l1_fd >= 0 && fastrpc_munmap) {
            fastrpc_munmap(domain_id,
                           ctx->l1_fd,
                           (void *)dataset->l1_codes,
                           l1_bytes);
        }
        ctx->graph_mapped = 0;
        ctx->l1_mapped = 0;
        ctx->graph_fd = -1;
        ctx->l1_fd = -1;
        printf("  search-build DSP index load: copy fallback\n");
    } else {
        printf("  search-build DSP index load: DMA zero-copy staging, workers=%d\n",
               ctx->buffer_count);
    }
    fflush(stdout);
    return 0;
}

static void demo_search_build_dsp_db_ctx_destroy(
    DemoSearchBuildDspDbCtx *ctx) {
    int i;
    size_t l1_bytes = 0;

    if (!ctx) {
        return;
    }
    if (ctx->search_loaded) {
        calculator_retrieval_unload(ctx->handle);
        ctx->search_loaded = 0;
    }
    if (ctx->dataset) {
        l1_bytes = (size_t)ctx->dataset->num_vectors *
                   (size_t)ctx->dataset->l1_code_bytes;
    }
    if (ctx->graph_mapped && ctx->graph_fd >= 0 && fastrpc_munmap) {
        fastrpc_munmap(ctx->domain_id,
                       ctx->graph_fd,
                       ctx->graph_stage,
                       ctx->graph_bytes);
    }
    if (ctx->l1_mapped && ctx->l1_fd >= 0 && ctx->dataset &&
        fastrpc_munmap) {
        fastrpc_munmap(ctx->domain_id,
                       ctx->l1_fd,
                       (void *)ctx->dataset->l1_codes,
                       l1_bytes);
    }
    if (ctx->buffers) {
        for (i = 0; i < ctx->buffer_count; i++) {
            demo_db_buffer_destroy(&ctx->buffers[i], ctx->domain_id);
        }
    }
    free(ctx->buffers);
    free(ctx->buffer_busy);
    demo_free_maybe_shared(ctx->graph_stage, ctx->graph_uses_rpcmem);
    if (ctx->sync_ready) {
        pthread_cond_destroy(&ctx->cond);
        pthread_mutex_destroy(&ctx->mu);
    }
    memset(ctx, 0, sizeof(*ctx));
}

static int demo_search_build_dsp_acquire_buffer(
    DemoSearchBuildDspDbCtx *ctx) {
    int idx = -1;
    unsigned long long t0;

    if (!ctx || !ctx->sync_ready) {
        return -1;
    }
    t0 = now_us();
    pthread_mutex_lock(&ctx->mu);
    while (idx < 0) {
        int i;
        for (i = 0; i < ctx->buffer_count; i++) {
            if (!ctx->buffer_busy[i]) {
                ctx->buffer_busy[i] = 1;
                idx = i;
                break;
            }
        }
        if (idx < 0) {
            pthread_cond_wait(&ctx->cond, &ctx->mu);
        }
    }
    ctx->buffer_wait_us += now_us() - t0;
    pthread_mutex_unlock(&ctx->mu);
    return idx;
}

static void demo_search_build_dsp_release_buffer(
    DemoSearchBuildDspDbCtx *ctx,
    int idx) {
    if (!ctx || !ctx->sync_ready ||
        idx < 0 || idx >= ctx->buffer_count) {
        return;
    }
    pthread_mutex_lock(&ctx->mu);
    ctx->buffer_busy[idx] = 0;
    pthread_cond_signal(&ctx->cond);
    pthread_mutex_unlock(&ctx->mu);
}

static int demo_search_build_dsp_prepare_iteration(
    void *ctx_ptr,
    const float *base,
    int num_vectors,
    int dim,
    const RetrievalSearchBuildConfig *cfg,
    const RetrievalRnnGraph *graph_snapshot,
    int iteration) {
    DemoSearchBuildDspDbCtx *ctx = (DemoSearchBuildDspDbCtx *)ctx_ptr;
    int ret;
    unsigned long long t0;

    (void)base;
    if (!ctx || !ctx->dataset || !cfg || !graph_snapshot ||
        !graph_snapshot->neighbors ||
        num_vectors != graph_snapshot->num_vectors ||
        dim != graph_snapshot->dim ||
        cfg->r != graph_snapshot->max_degree ||
        graph_snapshot->max_degree <= 0) {
        return -1;
    }

    memcpy(ctx->graph_stage, graph_snapshot->neighbors, ctx->graph_bytes);
    __sync_synchronize();

    t0 = now_us();
    if (ctx->use_dma) {

        ret = calculator_retrieval_load_dma(ctx->handle,
                                            ctx->graph_fd,
                                            &ctx->l1_fd, 1,
                                            num_vectors,
                                            num_vectors,
                                            dim,
                                            graph_snapshot->max_degree,
                                            DATA_GRAPH_ENTRY_SEARCH_BUILD,
                                            ctx->dataset->l1_qparams.scale,
                                            ctx->dataset->l1_qparams.zero_point);
    } else {
        ret = calculator_retrieval_load(ctx->handle,
                                        ctx->graph_stage,
                                        num_vectors * graph_snapshot->max_degree,
                                        ctx->dataset->l1_codes,
                                        num_vectors *
                                            ctx->dataset->l1_code_bytes,
                                        num_vectors,
                                        dim,
                                        graph_snapshot->max_degree,
                                        DATA_GRAPH_ENTRY_SEARCH_BUILD,
                                        ctx->dataset->l1_qparams.scale,
                                        ctx->dataset->l1_qparams.zero_point);
    }
    ctx->load_us += now_us() - t0;
    if (ret != 0) {
        printf("  ERROR: search-build DSP load iter=%d failed 0x%x\n",
               iteration,
               ret);
        fflush(stdout);
        return -1;
    }
    ctx->load_calls++;
    ctx->search_loaded = 1;
    return 0;
}

static int demo_search_build_copy_rpc_results(
    const int *ids,
    const float *dists,
    int count,
    int candidate_cap,
    RetrievalSearchBuildCandidate *out_candidates,
    int *out_count) {
    int i;

    if (!ids || !dists || !out_candidates || !out_count ||
        count < 0 || count > candidate_cap) {
        return -1;
    }
    for (i = 0; i < count; i++) {
        out_candidates[i].id = ids[i];
        out_candidates[i].search_dist = dists[i];
        out_candidates[i].exact_dist = dists[i];
    }
    *out_count = count;
    return 0;
}

static int demo_search_build_dsp_db_pair_provider(
    void *ctx_ptr,
    const float *base,
    int num_vectors,
    int dim,
    const RetrievalSearchBuildConfig *cfg,
    const RetrievalRnnGraph *graph_snapshot,
    int query_id_a,
    int query_id_b,
    int candidate_cap,
    RetrievalSearchBuildCandidate *out_candidates_a,
    int *out_count_a,
    RetrievalSearchBuildCandidate *out_candidates_b,
    int *out_count_b) {
    DemoSearchBuildDspDbCtx *ctx = (DemoSearchBuildDspDbCtx *)ctx_ptr;
    DbSharedBuffer *db_shared;
    DbCorrectionCtx corr_ctx;
    pthread_t corr_thread;
    int corr_started = 0;
    int buffer_idx = -1;
    int *ids_a = NULL;
    int *ids_b = NULL;
    float *dists_a = NULL;
    float *dists_b = NULL;
    int32_t actual_count_a = 0;
    int32_t actual_count_b = 0;
    int ret = -1;
    unsigned long long t0;
    unsigned long long rpc_us = 0;

    (void)graph_snapshot;
    if (!ctx || !base || !cfg ||
        query_id_a < 0 || query_id_a >= num_vectors ||
        query_id_b < 0 || query_id_b >= num_vectors ||
        candidate_cap <= 0 ||
        !out_candidates_a || !out_count_a ||
        !out_candidates_b || !out_count_b ||
        !ctx->search_loaded) {
        return -1;
    }

    ids_a = (int *)malloc((size_t)candidate_cap * sizeof(*ids_a));
    ids_b = (int *)malloc((size_t)candidate_cap * sizeof(*ids_b));
    dists_a = (float *)malloc((size_t)candidate_cap * sizeof(*dists_a));
    dists_b = (float *)malloc((size_t)candidate_cap * sizeof(*dists_b));
    if (!ids_a || !ids_b || !dists_a || !dists_b) {
        goto done;
    }

    buffer_idx = demo_search_build_dsp_acquire_buffer(ctx);
    if (buffer_idx < 0) {
        goto done;
    }
    db_shared = &ctx->buffers[buffer_idx];
    demo_db_buffer_reset(db_shared);

    memset(&corr_ctx, 0, sizeof(corr_ctx));
    corr_ctx.buf = db_shared->buf;
    corr_ctx.query_a = base + (size_t)query_id_a * (size_t)dim;
    corr_ctx.query_b = base + (size_t)query_id_b * (size_t)dim;
    corr_ctx.dataset = ctx->dataset;
    if (pthread_create(&corr_thread, NULL,
                       db_correction_thread_fn, &corr_ctx) != 0) {
        goto done;
    }
    corr_started = 1;

    t0 = now_us();
    ret = calculator_retrieval_search_db(
        ctx->handle,
        corr_ctx.query_a, dim,
        corr_ctx.query_b, dim,
        db_shared->fd,
        cfg->entry_points,
        cfg->expand_count,
        cfg->pool_size,
        candidate_cap,
        ids_a, candidate_cap,
        dists_a, candidate_cap,
        &actual_count_a,
        ids_b, candidate_cap,
        dists_b, candidate_cap,
        &actual_count_b);
    rpc_us = now_us() - t0;
    corr_ctx.stop = 1;
    pthread_join(corr_thread, NULL);
    corr_started = 0;
    pthread_mutex_lock(&ctx->mu);
    ctx->rpc_us += rpc_us;
    ctx->corr_batches += corr_ctx.corr_batches;
    ctx->corr_candidates += corr_ctx.corr_candidates;
    if (db_shared->stats) {
        ctx->dsp_wait_cpu_done_us +=
            (uint64_t)db_shared->stats->dsp_wait_cpu_done_us;
        ctx->dsp_invalidate_us +=
            (uint64_t)db_shared->stats->dsp_invalidate_us;
        ctx->dsp_wait_poll_count +=
            (uint64_t)db_shared->stats->dsp_wait_poll_count;
    }
    pthread_mutex_unlock(&ctx->mu);
    if (ret != 0) {
        printf("  ERROR: search-build retrieval_search_db failed 0x%x\n", ret);
        fflush(stdout);
        ret = -1;
        goto done;
    }

    if (demo_search_build_copy_rpc_results(ids_a,
                                           dists_a,
                                           actual_count_a,
                                           candidate_cap,
                                           out_candidates_a,
                                           out_count_a) != 0 ||
        demo_search_build_copy_rpc_results(ids_b,
                                           dists_b,
                                           actual_count_b,
                                           candidate_cap,
                                           out_candidates_b,
                                           out_count_b) != 0) {
        ret = -1;
        goto done;
    }
    pthread_mutex_lock(&ctx->mu);
    ctx->rpc_calls++;
    ctx->rpc_queries += (query_id_a == query_id_b) ? 1u : 2u;
    pthread_mutex_unlock(&ctx->mu);
    ret = 0;

done:
    if (corr_started) {
        corr_ctx.stop = 1;
        pthread_join(corr_thread, NULL);
    }
    if (buffer_idx >= 0) {
        demo_search_build_dsp_release_buffer(ctx, buffer_idx);
    }
    free(ids_a);
    free(ids_b);
    free(dists_a);
    free(dists_b);
    return ret;
}

static int demo_resolve_graph_index_legacy(remote_handle64 handle,
                                    int domain_id,
                                    int32_t **out_graph,
                                    int *out_uses_rpcmem,
                                    int32_t *out_entry_point) {
    int32_t *graph = NULL;
    int uses_rpcmem = 0;
    int cpu_owned_graph = 0;
    size_t graph_bytes = (size_t)DATA_N_BASE *
                         (size_t)DATA_GRAPH_K *
                         sizeof(int32_t);
    unsigned long long t0;

    if (!out_graph || !out_uses_rpcmem || !out_entry_point) {
        return -1;
    }
    (void)handle;
    (void)domain_id;

    *out_graph = NULL;
    *out_uses_rpcmem = 0;
    *out_entry_point = DATA_GRAPH_ENTRY_POINT;


    cpu_owned_graph = (strcmp(RETRIEVAL_DB_MODE, "sqmd") == 0);
    if (cpu_owned_graph) {
        graph = (int32_t *)malloc(graph_bytes);
        if (graph) {
            printf("  graph: %llu bytes via CPU malloc (SQMD CPU-owned)\n",
                   (unsigned long long)graph_bytes);
        }
    } else {
        graph = (int32_t *)demo_alloc_shared_buffer(graph_bytes,
                                                    "graph",
                                                    &uses_rpcmem);
    }
    if (!graph) {
        printf("  ERROR: alloc failed\n");
        fflush(stdout);
        return -1;
    }

    t0 = now_us();

#if (RETRIEVAL_GRAPH_INDEX_SCHEME == 1)
    {
        char path[4096];
        char load_error[256];
        RetrievalNsgLoadStats load_stats;

        snprintf(path, sizeof(path), "%s", DATA_GRAPH_PATH);
        if (retrieval_nsg_load_fixed(path,
                                     DATA_N_BASE,
                                     DATA_GRAPH_K,
                                     graph,
                                     &load_stats,
                                     load_error,
                                     sizeof(load_error)) != 0) {
            printf("  ERROR: NSG load failed: %s\n", load_error);
            fflush(stdout);
            demo_free_maybe_shared(graph, uses_rpcmem);
            return -1;
        }
        printf("  NSG width=%u runtime_width=%d entry=%d "
               "truncated_nodes=%llu truncated_edges=%llu\n",
               load_stats.file_width,
               DATA_GRAPH_K,
               load_stats.entry_point,
               (unsigned long long)load_stats.truncated_nodes,
               (unsigned long long)load_stats.truncated_edges);
        if (load_stats.file_width != (uint32_t)DATA_GRAPH_K) {
            printf("  ERROR: NSG width %u does not match DATA_GRAPH_K=%d\n",
                   load_stats.file_width, DATA_GRAPH_K);
            fflush(stdout);
            demo_free_maybe_shared(graph, uses_rpcmem);
            return -1;
        }
        if (load_stats.truncated_nodes != 0) {
            printf("  ERROR: formal NSG run refuses loader-side truncation\n");
            fflush(stdout);
            demo_free_maybe_shared(graph, uses_rpcmem);
            return -1;
        }
        *out_entry_point = load_stats.entry_point;
        printf("  done. time=%llu ms (%s)\n",
               (now_us() - t0) / 1000ULL,
               uses_rpcmem ? "rpcmem" : "malloc fallback");
        fflush(stdout);
    }
#elif (RETRIEVAL_GRAPH_INDEX_SCHEME == 2) || \
      (RETRIEVAL_GRAPH_INDEX_SCHEME == 3)
    {
        RetrievalRnnBuildConfig cfg;
        RetrievalRnnBuildConfig cache_cfg;
        RetrievalRnnBuildRuntime runtime;
        RetrievalRnnBuildStats stats;
#if (RETRIEVAL_GRAPH_INDEX_SCHEME == 3)
        RetrievalSearchBuildConfig search_cfg;
        RetrievalSearchBuildRuntime search_runtime;
        RetrievalSearchBuildStats search_stats;
        DemoSearchBuildDspDbCtx search_db_ctx;
        int search_db_ctx_ready = 0;
#endif
        RetrievalRnnGraph rnn_graph;
        float *base = NULL;
        int base_uses_rpcmem = 0;
        int fd_base = -1;
        int base_mapped = 0;
        int rnn_l2_loaded = 0;
        int use_dsp_provider = 0;
        int have_graph = 0;
        size_t base_bytes = (size_t)DATA_N_BASE *
                            (size_t)DATA_DIM *
                            sizeof(float);
        DemoRnnDspDistanceCtx dsp_ctx;
        char cache_path[256];
        char base_path[256];
        uint32_t cache_version = RNN_BUILD_CACHE_VERSION;

        memset(&rnn_graph, 0, sizeof(rnn_graph));
        memset(&runtime, 0, sizeof(runtime));
        memset(&dsp_ctx, 0, sizeof(dsp_ctx));
        retrieval_rnn_build_stats_reset(&stats);
#if (RETRIEVAL_GRAPH_INDEX_SCHEME == 3)
        memset(&search_cfg, 0, sizeof(search_cfg));
        memset(&search_runtime, 0, sizeof(search_runtime));
        memset(&search_db_ctx, 0, sizeof(search_db_ctx));
        retrieval_search_build_stats_reset(&search_stats);
#endif

        cfg.s = RNN_BUILD_S;
        cfg.r = RNN_BUILD_R;
        cfg.t1 = RNN_BUILD_T1;
        cfg.t2 = RNN_BUILD_T2;
        cfg.seed = RNN_BUILD_SEED;
        cache_cfg = cfg;
#if (RETRIEVAL_GRAPH_INDEX_SCHEME == 3)
        cache_cfg.s = SEARCH_BUILD_CANDIDATES;
        cache_cfg.r = SEARCH_BUILD_R;
        cache_cfg.t1 = SEARCH_BUILD_ITERATIONS;
        cache_cfg.t2 = SEARCH_BUILD_POOL_SIZE;
        cache_cfg.seed = demo_search_build_cache_seed();
        cache_version = SEARCH_BUILD_CACHE_VERSION;
#endif

        snprintf(cache_path, sizeof(cache_path), "%s", DATA_GRAPH_PATH);
        snprintf(base_path, sizeof(base_path), "%s", DATA_BASE_PATH);

        if (GRAPH_BUILD_MODE != 2) {
            unsigned long long t_cache = now_us();
            if (retrieval_rnn_graph_load_cache(cache_path,
                                               DATA_LAYOUT_NAME,
                                               DATA_N_BASE,
                                               DATA_DIM,
                                               &cache_cfg,
	                                               cache_version,
	                                               &rnn_graph) == 0) {
	                have_graph = 1;
#if (RETRIEVAL_GRAPH_INDEX_SCHEME == 3)
	                printf("  search-build cache hit: load=%llu ms degree_cap=%d\n",
	                       (now_us() - t_cache) / 1000ULL,
	                       rnn_graph.max_degree);
#else
	                printf("  RNN cache hit: load=%llu ms degree_cap=%d\n",
	                       (now_us() - t_cache) / 1000ULL,
	                       rnn_graph.max_degree);
#endif
	                fflush(stdout);
	            }
	        }

        if (!have_graph) {
#if (RETRIEVAL_GRAPH_INDEX_SCHEME == 3)
            if (GRAPH_BUILD_MODE == 0) {
                printf("  ERROR: search-build cache missing/stale and GRAPH_BUILD_MODE=0\n");
#else
            if (GRAPH_BUILD_MODE == 0) {
                printf("  ERROR: RNN cache missing/stale and GRAPH_BUILD_MODE=0\n");
#endif
                fflush(stdout);
                demo_free_maybe_shared(graph, uses_rpcmem);
                return -1;
            }

#if (RETRIEVAL_GRAPH_INDEX_SCHEME == 3)
            printf("  search-build cache %s: iter=%d R=%d cand=%d entry=%d expand=%d pool=%d seed=%u threads=%d provider=%s prune=%s\n",
                   (GRAPH_BUILD_MODE == 2) ? "force rebuild" : "miss/stale",
                   SEARCH_BUILD_ITERATIONS,
                   SEARCH_BUILD_R,
                   SEARCH_BUILD_CANDIDATES,
                   SEARCH_BUILD_ENTRY_POINTS,
                   SEARCH_BUILD_EXPAND_COUNT,
                   SEARCH_BUILD_POOL_SIZE,
                   (unsigned)SEARCH_BUILD_SEED,
                   SEARCH_BUILD_THREADS,
                   (SEARCH_BUILD_PROVIDER == 0) ? "exact" :
                   (SEARCH_BUILD_PROVIDER == 1) ? "hlos" :
                   (SEARCH_BUILD_PROVIDER == 2) ? "dsp_db" : "dsp_hmx",
                   (SEARCH_BUILD_PRUNE_MODE == 0) ? "topr" : "robust");
#else
            printf("  RNN cache %s: build S=%d R=%d T1=%d T2=%d seed=%u threads=%d provider=%s\n",
                   (GRAPH_BUILD_MODE == 2) ? "force rebuild" : "miss/stale",
                   cfg.s, cfg.r, cfg.t1, cfg.t2,
                   (unsigned)cfg.seed, RNN_BUILD_THREADS,
                   (RNN_BUILD_DISTANCE_PROVIDER == 1) ? "dsp" : "cpu");
#endif
            fflush(stdout);

            {
                unsigned long long t_base = now_us();
                if ((RETRIEVAL_GRAPH_INDEX_SCHEME == 2) &&
                    (RNN_BUILD_DISTANCE_PROVIDER == 1)) {
                    base = (float *)demo_alloc_shared_buffer(base_bytes,
                                                             "rnn_base_float",
                                                             &base_uses_rpcmem);
                    if (!base) {
                        printf("  ERROR: alloc RNN base shared buffer failed\n");
                        fflush(stdout);
                        demo_free_maybe_shared(graph, uses_rpcmem);
                        return -1;
                    }
                    if (demo_load_fvecs_n_into(base_path,
                                               DATA_N_BASE,
                                               DATA_DIM,
                                               base) != 0) {
                        printf("  ERROR: load base fvecs for RNN build failed\n");
                        fflush(stdout);
                        demo_rnn_l2_provider_cleanup(handle,
                                                      domain_id,
                                                      base,
                                                      base_uses_rpcmem,
                                                      fd_base,
                                                      base_mapped,
                                                      rnn_l2_loaded,
                                                      base_bytes,
                                                      &dsp_ctx);
                        demo_free_maybe_shared(graph, uses_rpcmem);
                        return -1;
                    }
	                } else if (load_fvecs_n(base_path, DATA_N_BASE, DATA_DIM,
	                                        &base) != 0) {
	                    printf("  ERROR: load base fvecs for graph build failed\n");
	                    fflush(stdout);
	                    demo_free_maybe_shared(graph, uses_rpcmem);
	                    return -1;
	                }
	                printf("  graph build base loaded: %llu ms (%s)\n",
	                       (now_us() - t_base) / 1000ULL,
	                       base_uses_rpcmem ? "rpcmem" : "malloc");
                fflush(stdout);
            }

            runtime.threads = RNN_BUILD_THREADS;
            runtime.batch_size = RNN_DSP_L2_BATCH_SIZE;
            runtime.min_batch = RNN_DSP_L2_MIN_BATCH;
            runtime.progress_every_t2 = RNN_BUILD_PROGRESS_EVERY_T2;

            if ((RETRIEVAL_GRAPH_INDEX_SCHEME == 2) &&
                (RNN_BUILD_DISTANCE_PROVIDER == 1)) {
                int setup_ok = 0;
                if (!base_uses_rpcmem || !fastrpc_mmap) {
                    printf("  WARN: RNN DSP L2 provider unavailable (rpcmem=%d fastrpc_mmap=%p)\n",
                           base_uses_rpcmem,
                           (void *)fastrpc_mmap);
                    fflush(stdout);
                } else {
                    int mret;
                    fd_base = rpcmem_to_fd(base);
                    if (fd_base < 0) {
                        printf("  WARN: rpcmem_to_fd(rnn_base_float) failed fd=%d\n",
                               fd_base);
                        fflush(stdout);
                    } else {
                        mret = fastrpc_mmap(domain_id,
                                            fd_base,
                                            base,
                                            0,
                                            base_bytes,
                                            FASTRPC_MAP_FD);
                        if (mret == 0) {
                            base_mapped = 1;
                            mret = calculator_retrieval_rnn_l2_load_dma(
                                handle,
                                fd_base,
                                DATA_N_BASE,
                                DATA_DIM);
                            if (mret == 0) {
                                rnn_l2_loaded = 1;
                                setup_ok = 1;
                            } else {
                                printf("  WARN: retrieval_rnn_l2_load_dma failed 0x%x\n",
                                       mret);
                                fflush(stdout);
                            }
                        } else {
                            printf("  WARN: fastrpc_mmap(rnn_base_float) failed 0x%x\n",
                                   mret);
                            fflush(stdout);
                        }
                    }
                }

                if (setup_ok) {
                    if (pthread_mutex_init(&dsp_ctx.mu, NULL) != 0) {
                        printf("  WARN: RNN DSP L2 ctx mutex init failed\n");
                        fflush(stdout);
                    } else {
                        dsp_ctx.mu_ready = 1;
                        dsp_ctx.handle = handle;
                        dsp_ctx.base = base;
                        dsp_ctx.dim = DATA_DIM;
                        dsp_ctx.fallback_on_error =
                            RNN_DSP_L2_FALLBACK_ON_ERROR ? 1 : 0;
                        runtime.distance_batch = demo_rnn_dsp_distance_batch;
                        runtime.distance_ctx = &dsp_ctx;
                        use_dsp_provider = 1;
                        printf("  RNN DSP L2 load: fd=%d batch=%d min=%d fallback=%d\n",
                               fd_base,
                               RNN_DSP_L2_BATCH_SIZE,
                               RNN_DSP_L2_MIN_BATCH,
                               RNN_DSP_L2_FALLBACK_ON_ERROR);
                        fflush(stdout);
                    }
                }

                if (!use_dsp_provider) {
                    if (!RNN_DSP_L2_FALLBACK_ON_ERROR) {
                        printf("  ERROR: RNN DSP L2 provider setup failed and fallback disabled\n");
                        fflush(stdout);
                        demo_rnn_l2_provider_cleanup(handle,
                                                      domain_id,
                                                      base,
                                                      base_uses_rpcmem,
                                                      fd_base,
                                                      base_mapped,
                                                      rnn_l2_loaded,
                                                      base_bytes,
                                                      &dsp_ctx);
                        demo_free_maybe_shared(graph, uses_rpcmem);
                        return -1;
                    }
                    printf("  RNN DSP L2 fallback: CPU distance provider\n");
                    fflush(stdout);
                }
            }

#if (RETRIEVAL_GRAPH_INDEX_SCHEME == 3)
            search_cfg.r = SEARCH_BUILD_R;
            search_cfg.iterations = SEARCH_BUILD_ITERATIONS;
            search_cfg.candidate_count = SEARCH_BUILD_CANDIDATES;
            search_cfg.entry_points = SEARCH_BUILD_ENTRY_POINTS;
            search_cfg.expand_count = SEARCH_BUILD_EXPAND_COUNT;
            search_cfg.pool_size = SEARCH_BUILD_POOL_SIZE;
            search_cfg.seed = SEARCH_BUILD_SEED;
            search_cfg.prune_mode = SEARCH_BUILD_PRUNE_MODE;
            search_cfg.batch_align = SEARCH_BUILD_BATCH_ALIGN;
            search_runtime.threads = SEARCH_BUILD_THREADS;
            search_runtime.progress_every_iter = RNN_BUILD_PROGRESS_EVERY_T2;
            if (SEARCH_BUILD_PROVIDER == 0) {
                search_runtime.search = retrieval_search_build_exact_provider;
            } else if (SEARCH_BUILD_PROVIDER == 1) {
                search_runtime.search = retrieval_search_build_graph_provider;
            } else if (SEARCH_BUILD_PROVIDER == 2) {
                if (!g_dataset.l1_codes ||
                    g_dataset.num_vectors != DATA_N_BASE ||
                    g_dataset.dim != DATA_DIM) {
                    printf("  ERROR: search-build DSP provider requires encoded dataset before graph resolve\n");
                    fflush(stdout);
                    demo_rnn_l2_provider_cleanup(handle,
                                                  domain_id,
                                                  base,
                                                  base_uses_rpcmem,
                                                  fd_base,
                                                  base_mapped,
                                                  rnn_l2_loaded,
                                                  base_bytes,
                                                  &dsp_ctx);
                    demo_free_maybe_shared(graph, uses_rpcmem);
                    return -1;
                }
                if (demo_search_build_dsp_db_ctx_init(&search_db_ctx,
                                                      handle,
                                                      domain_id,
                                                      &g_dataset,
                                                      DATA_N_BASE,
                                                      SEARCH_BUILD_R) != 0) {
                    printf("  ERROR: search-build DSP DB provider init failed\n");
                    fflush(stdout);
                    demo_search_build_dsp_db_ctx_destroy(&search_db_ctx);
                    demo_rnn_l2_provider_cleanup(handle,
                                                  domain_id,
                                                  base,
                                                  base_uses_rpcmem,
                                                  fd_base,
                                                  base_mapped,
                                                  rnn_l2_loaded,
                                                  base_bytes,
                                                  &dsp_ctx);
                    demo_free_maybe_shared(graph, uses_rpcmem);
                    return -1;
                }
                search_db_ctx_ready = 1;
                search_runtime.search_pair =
                    demo_search_build_dsp_db_pair_provider;
                search_runtime.search_ctx = &search_db_ctx;
                search_runtime.prepare_iteration =
                    demo_search_build_dsp_prepare_iteration;
                search_runtime.iteration_ctx = &search_db_ctx;
            } else {
                printf("  ERROR: search-build provider=%d is planned for HMX follow-up; use provider=0 exact, 1 hlos, or 2 dsp_db\n",
                       SEARCH_BUILD_PROVIDER);
                fflush(stdout);
                demo_rnn_l2_provider_cleanup(handle,
                                              domain_id,
                                              base,
                                              base_uses_rpcmem,
                                              fd_base,
                                              base_mapped,
                                              rnn_l2_loaded,
                                              base_bytes,
                                              &dsp_ctx);
                demo_free_maybe_shared(graph, uses_rpcmem);
                return -1;
            }
            if (retrieval_search_build_graph_ex(base,
                                                DATA_N_BASE,
                                                DATA_DIM,
                                                &search_cfg,
                                                &search_runtime,
                                                &rnn_graph,
                                                &search_stats) != 0) {
#else
            if (retrieval_rnn_build_graph_ex(base,
                                             DATA_N_BASE,
                                             DATA_DIM,
                                             &cfg,
                                             &runtime,
                                             &rnn_graph,
                                             &stats) != 0) {
#endif
#if (RETRIEVAL_GRAPH_INDEX_SCHEME == 3)
		                printf("  ERROR: search-build failed\n");
#else
		                printf("  ERROR: RNN build failed\n");
#endif
		                fflush(stdout);
#if (RETRIEVAL_GRAPH_INDEX_SCHEME == 3)
                if (search_db_ctx_ready) {
                    demo_search_build_dsp_db_ctx_destroy(&search_db_ctx);
                    search_db_ctx_ready = 0;
                }
#endif
	                demo_rnn_l2_provider_cleanup(handle,
	                                              domain_id,
	                                              base,
                                              base_uses_rpcmem,
                                              fd_base,
                                              base_mapped,
                                              rnn_l2_loaded,
                                              base_bytes,
                                              &dsp_ctx);
                demo_free_maybe_shared(graph, uses_rpcmem);
                return -1;
            }
#if (RETRIEVAL_GRAPH_INDEX_SCHEME == 3)
            demo_rnn_l2_provider_cleanup(handle,
                                          domain_id,
                                          base,
                                          base_uses_rpcmem,
                                          fd_base,
                                          base_mapped,
                                          rnn_l2_loaded,
	                                          base_bytes,
	                                          &dsp_ctx);
            if (search_db_ctx_ready) {
                printf("  search-build DSP DB: loads=%llu load=%llu ms rpc=%llu queries=%llu rpc_time=%llu ms wait_buf=%llu ms corr_batches=%llu corr_candidates=%llu dsp_wait_cpu=%llu ms dsp_invalidate=%llu ms dsp_polls=%llu\n",
                       (unsigned long long)search_db_ctx.load_calls,
                       (unsigned long long)search_db_ctx.load_us / 1000ULL,
                       (unsigned long long)search_db_ctx.rpc_calls,
                       (unsigned long long)search_db_ctx.rpc_queries,
                       (unsigned long long)search_db_ctx.rpc_us / 1000ULL,
                       (unsigned long long)search_db_ctx.buffer_wait_us / 1000ULL,
                       (unsigned long long)search_db_ctx.corr_batches,
                       (unsigned long long)search_db_ctx.corr_candidates,
                       (unsigned long long)search_db_ctx.dsp_wait_cpu_done_us / 1000ULL,
                       (unsigned long long)search_db_ctx.dsp_invalidate_us / 1000ULL,
                       (unsigned long long)search_db_ctx.dsp_wait_poll_count);
                fflush(stdout);
                demo_search_build_dsp_db_ctx_destroy(&search_db_ctx);
                search_db_ctx_ready = 0;
            }
            base = NULL;
            have_graph = 1;

            printf("  search-build stats: provider=%s threads=%d iter=%d init=%llu ms search=%llu ms prune=%llu ms refine=%llu ms queries=%llu candidates=%llu kept=%llu exact_pairs=%llu fill=%llu avg_degree=%.2f min_degree=%d max_degree=%d isolated=%d\n",
                   demo_search_build_provider_name(),
                   search_stats.threads_used,
                   search_stats.iterations_done,
                   (unsigned long long)search_stats.init_us / 1000ULL,
                   (unsigned long long)search_stats.search_us / 1000ULL,
                   (unsigned long long)search_stats.prune_us / 1000ULL,
                   (unsigned long long)search_stats.refine_us / 1000ULL,
                   (unsigned long long)search_stats.searched_queries,
                   (unsigned long long)search_stats.candidates_seen,
                   (unsigned long long)search_stats.candidates_kept,
                   (unsigned long long)search_stats.exact_distance_pairs,
                   (unsigned long long)search_stats.fill_edges,
                   (double)search_stats.avg_degree_x100 / 100.0,
                   search_stats.min_degree,
                   search_stats.max_degree,
                   search_stats.isolated_nodes);
            fflush(stdout);
#else
            if (use_dsp_provider) {
                stats.distance_pairs_dsp = dsp_ctx.rpc_pairs;
                stats.distance_pairs_cpu = dsp_ctx.cpu_pairs;
                stats.distance_rpc_us = dsp_ctx.rpc_us;
                stats.distance_fallbacks = dsp_ctx.fallback_calls;
            }
            demo_rnn_l2_provider_cleanup(handle,
                                          domain_id,
                                          base,
                                          base_uses_rpcmem,
                                          fd_base,
                                          base_mapped,
                                          rnn_l2_loaded,
                                          base_bytes,
                                          &dsp_ctx);
            base = NULL;
            have_graph = 1;

            printf("  RNN build stats: provider=%s threads=%d init=%llu ms update=%llu ms reverse=%llu ms pairs=%llu batches=%llu dsp_pairs=%llu cpu_pairs=%llu rpc=%llu ms fallback=%llu avg_degree=%.2f max_degree=%d isolated=%d\n",
                   use_dsp_provider ? "dsp" : "cpu",
                   stats.threads_used,
                   (unsigned long long)stats.init_us / 1000ULL,
                   (unsigned long long)stats.update_us / 1000ULL,
                   (unsigned long long)stats.reverse_us / 1000ULL,
                   (unsigned long long)stats.distance_pairs_requested,
                   (unsigned long long)stats.distance_batches,
                   (unsigned long long)stats.distance_pairs_dsp,
                   (unsigned long long)stats.distance_pairs_cpu,
                   (unsigned long long)stats.distance_rpc_us / 1000ULL,
                   (unsigned long long)stats.distance_fallbacks,
                   (double)stats.avg_degree_x100 / 100.0,
                   stats.max_degree,
                   stats.isolated_nodes);
            fflush(stdout);
#endif

            if (retrieval_rnn_graph_save_cache(cache_path,
                                               DATA_LAYOUT_NAME,
                                               &rnn_graph,
                                               &cache_cfg,
                                               cache_version) != 0) {
#if (RETRIEVAL_GRAPH_INDEX_SCHEME == 3)
	                printf("  ERROR: save search-build cache failed: %s\n",
	                       cache_path);
#else
	                printf("  ERROR: save RNN cache failed: %s\n", cache_path);
#endif
	                fflush(stdout);
                retrieval_rnn_graph_free(&rnn_graph);
                demo_free_maybe_shared(graph, uses_rpcmem);
                return -1;
            }
#if (RETRIEVAL_GRAPH_INDEX_SCHEME == 3)
	            printf("  search-build cache saved: %s\n", cache_path);
#else
	            printf("  RNN cache saved: %s\n", cache_path);
#endif
	            fflush(stdout);
	        }

	        {
	            unsigned long long t_export = now_us();
#if (RETRIEVAL_GRAPH_INDEX_SCHEME == 3)
	            int export_threads = SEARCH_BUILD_THREADS;
#else
	            int export_threads = RNN_BUILD_THREADS;
#endif
	            if (retrieval_rnn_export_topk_threads(&rnn_graph,
	                                                  DATA_GRAPH_K,
	                                                  export_threads,
	                                                  graph) != 0) {
#if (RETRIEVAL_GRAPH_INDEX_SCHEME == 3)
	                printf("  ERROR: export search-build top-K graph failed\n");
#else
	                printf("  ERROR: export RNN top-K graph failed\n");
#endif
	                fflush(stdout);
	                retrieval_rnn_graph_free(&rnn_graph);
	                demo_free_maybe_shared(graph, uses_rpcmem);
	                return -1;
	            }
#if (RETRIEVAL_GRAPH_INDEX_SCHEME == 3)
	            printf("  search-build export topK=%d: %llu ms\n",
	                   DATA_GRAPH_K,
	                   (now_us() - t_export) / 1000ULL);
#else
	            printf("  RNN export topK=%d: %llu ms\n",
	                   DATA_GRAPH_K,
	                   (now_us() - t_export) / 1000ULL);
#endif
	            fflush(stdout);
	        }

        retrieval_rnn_graph_free(&rnn_graph);
        printf("  done. time=%llu ms (%s) entry=%d\n",
               (now_us() - t0) / 1000ULL,
               uses_rpcmem ? "rpcmem" : "malloc fallback",
               *out_entry_point);
        fflush(stdout);
    }
#else
    {
        char path[4096];
        snprintf(path, sizeof(path), "%s", DATA_GRAPH_PATH);
        if (load_graph_nndescent(path, DATA_N_BASE, DATA_GRAPH_K, graph) != 0) {
            printf("  ERROR\n");
            fflush(stdout);
            demo_free_maybe_shared(graph, uses_rpcmem);
            return -1;
        }
        printf("  done. time=%llu ms (%s)\n",
               (now_us() - t0) / 1000ULL,
               uses_rpcmem ? "rpcmem" : "malloc fallback");
        fflush(stdout);
    }
#endif

    *out_graph = graph;
    *out_uses_rpcmem = uses_rpcmem;
    return 0;
}

static int demo_resolve_graph_index(remote_handle64 handle,int domain_id,
    int32_t **out,int *shared,int32_t *entry) {
    (void)handle;(void)domain_id;
    if(rover_load_residual_map())return -1;
    size_t count=(size_t)DATA_N_BASE*DATA_GRAPH_K;
    *shared=0;
    int32_t *graph=!strcmp(RETRIEVAL_DB_MODE,"db_pair")?
        demo_alloc_shared_buffer(count*sizeof(*graph),"graph",shared):malloc(count*sizeof(*graph));
    if (!graph)return -1;
    int rc=-1;*entry=rover_config.graph_entry;
    if (!strcmp(rover_config.graph_format,"nsg")) {
        RetrievalNsgLoadStats stats;char error[256];
        rc=retrieval_nsg_load_fixed(DATA_GRAPH_PATH,DATA_N_BASE,DATA_GRAPH_K,graph,&stats,error,sizeof(error));
        if (!rc)*entry=stats.entry_point;
    } else if (!strcmp(rover_config.graph_format,"nndescent")) {
        rc=load_graph_nndescent(DATA_GRAPH_PATH,DATA_N_BASE,DATA_GRAPH_K,graph);
    } else {
        FILE *f=fopen(DATA_GRAPH_PATH,"rb");
        if(f) {
            int32_t h[4];int header_ok=1;
            if (!strcmp(rover_config.graph_format,"rover"))
                header_ok=fread(h,sizeof(h),1,f)==1 && h[0]==0x35314252 && h[1]==DATA_N_BASE && h[2]==DATA_DIM && h[3]==DATA_GRAPH_K;
            else if(strcmp(rover_config.graph_format,"dense"))header_ok=0;
            if(header_ok && fread(graph,sizeof(*graph),count,f)==count && fgetc(f)==EOF && !ferror(f))rc=0;
            fclose(f);
        }
    }
    if(!rc)for(size_t i=0;i<count;i++)if(graph[i]<-1 || graph[i]>=DATA_N_BASE){rc=-1;break;}
    if(rc){demo_free_maybe_shared(graph,*shared);*shared=0;return -1;}
    *out=graph;return 0;
}
