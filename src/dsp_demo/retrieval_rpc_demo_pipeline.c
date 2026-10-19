static int sqmd_final_decode(void *provider,int id,float *vector);
static int rover_open_sq_refine(ResidualBlockCache **cache,RetrievalFinalRefineContext **out);


static unsigned long long now_us(void);

static int pipeline_all_free_locked(const RetrievalPipeline *pipe) {
    int i;

    for (i = 0; i < pipe->slot_count; i++) {
        if (pipe->state[i] != RETRIEVAL_SLOT_FREE) {
            return 0;
        }
    }
    return 1;
}

static int pipeline_init(RetrievalPipeline *pipe,
                         int slot_count,
                         int rerank_count,
                         DemoResult *results,
                         const RetrievalDemoConfig *configs) {
    int i;

    if (!pipe || slot_count <= 0 || rerank_count <= 0 || !results || !configs) {
        return -1;
    }

    memset(pipe, 0, sizeof(*pipe));
    pipe->slot_count = slot_count;
    pipe->rerank_count = rerank_count;
    pipe->results = results;
    pipe->configs = configs;

    if (pthread_mutex_init(&pipe->mu, NULL) != 0) {
        return -1;
    }
    if (pthread_cond_init(&pipe->cond_ready, NULL) != 0) {
        pthread_mutex_destroy(&pipe->mu);
        return -1;
    }
    if (pthread_cond_init(&pipe->cond_free, NULL) != 0) {
        pthread_cond_destroy(&pipe->cond_ready);
        pthread_mutex_destroy(&pipe->mu);
        return -1;
    }

    pipe->slots = (RetrievalSlot *)calloc((size_t)slot_count, sizeof(RetrievalSlot));
    pipe->state = (int *)calloc((size_t)slot_count, sizeof(int));
    if (!pipe->slots || !pipe->state) {
        free(pipe->slots);
        free(pipe->state);
        pthread_cond_destroy(&pipe->cond_free);
        pthread_cond_destroy(&pipe->cond_ready);
        pthread_mutex_destroy(&pipe->mu);
        return -1;
    }

    for (i = 0; i < slot_count; i++) {
        pipe->slots[i].candidate_ids =
            (int32_t *)malloc((size_t)rerank_count * sizeof(int32_t));
        pipe->slots[i].candidate_dists =
            (float *)malloc((size_t)rerank_count * sizeof(float));
        pipe->state[i] = RETRIEVAL_SLOT_FREE;
        if (!pipe->slots[i].candidate_ids || !pipe->slots[i].candidate_dists) {
            int j;

            for (j = 0; j <= i; j++) {
                free(pipe->slots[j].candidate_ids);
                free(pipe->slots[j].candidate_dists);
                pipe->slots[j].candidate_ids = NULL;
                pipe->slots[j].candidate_dists = NULL;
            }
            free(pipe->slots);
            free(pipe->state);
            pipe->slots = NULL;
            pipe->state = NULL;
            pthread_cond_destroy(&pipe->cond_free);
            pthread_cond_destroy(&pipe->cond_ready);
            pthread_mutex_destroy(&pipe->mu);
            return -1;
        }
    }

    return 0;
}

static void pipeline_destroy(RetrievalPipeline *pipe) {
    int i;

    if (!pipe) {
        return;
    }

    for (i = 0; i < pipe->slot_count; i++) {
        retrieval_final_refine_close(pipe->slots[i].refinement);
        free(pipe->slots[i].candidate_ids);
        free(pipe->slots[i].candidate_dists);
        pipe->slots[i].candidate_ids = NULL;
        pipe->slots[i].candidate_dists = NULL;
    }
    free(pipe->slots);
    free(pipe->state);
    pipe->slots = NULL;
    pipe->state = NULL;
    pthread_cond_destroy(&pipe->cond_free);
    pthread_cond_destroy(&pipe->cond_ready);
    pthread_mutex_destroy(&pipe->mu);
    memset(pipe, 0, sizeof(*pipe));
}

static int pipeline_acquire_slot(RetrievalPipeline *pipe) {
    int s;

    pthread_mutex_lock(&pipe->mu);
    while (pipe->ret == 0) {
        for (s = 0; s < pipe->slot_count; s++) {
            if (pipe->state[s] == RETRIEVAL_SLOT_FREE) {
                pipe->state[s] = RETRIEVAL_SLOT_FILLING;
                pthread_mutex_unlock(&pipe->mu);
                return s;
            }
        }
        pthread_cond_wait(&pipe->cond_free, &pipe->mu);
    }
    pthread_mutex_unlock(&pipe->mu);
    return -1;
}

static void pipeline_abort_slot(RetrievalPipeline *pipe, int slot_idx) {
    pthread_mutex_lock(&pipe->mu);
    if (slot_idx >= 0 && slot_idx < pipe->slot_count &&
        pipe->state[slot_idx] == RETRIEVAL_SLOT_FILLING) {
        pipe->state[slot_idx] = RETRIEVAL_SLOT_FREE;
        pthread_cond_signal(&pipe->cond_free);
    }
    pthread_mutex_unlock(&pipe->mu);
}

static void pipeline_mark_ready(RetrievalPipeline *pipe, int slot_idx) {
    pthread_mutex_lock(&pipe->mu);
    pipe->state[slot_idx] = RETRIEVAL_SLOT_READY;
    pthread_cond_signal(&pipe->cond_ready);
    pthread_mutex_unlock(&pipe->mu);
}

static int pipeline_drain(RetrievalPipeline *pipe) {
    pthread_mutex_lock(&pipe->mu);
    while (pipe->ret == 0 && !pipeline_all_free_locked(pipe)) {
        pthread_cond_wait(&pipe->cond_free, &pipe->mu);
    }
    pthread_mutex_unlock(&pipe->mu);
    return pipe->ret;
}

static void pipeline_shutdown(RetrievalPipeline *pipe) {
    pthread_mutex_lock(&pipe->mu);
    pipe->shutdown = 1;
    pthread_cond_broadcast(&pipe->cond_ready);
    pthread_cond_broadcast(&pipe->cond_free);
    pthread_mutex_unlock(&pipe->mu);
}


static void *rerank_thread_fn(void *arg) {
    RetrievalPipeline *pipe = (RetrievalPipeline *)arg;

    while (1) {
        RetrievalSlot *slot = NULL;
        DemoResult *res = NULL;
        const RetrievalDemoConfig *config = NULL;
        int result_ids[SEARCH_K_MAX];
        int rc;
        int slot_idx = -1;
        unsigned long long t0;

        pthread_mutex_lock(&pipe->mu);
        while (!pipe->shutdown && pipe->ret == 0) {
            int i;

            for (i = 0; i < pipe->slot_count; i++) {
                if (pipe->state[i] == RETRIEVAL_SLOT_READY) {
                    pipe->state[i] = RETRIEVAL_SLOT_PROCESSING;
                    slot_idx = i;
                    slot = &pipe->slots[i];
                    break;
                }
            }
            if (slot) {
                break;
            }
            pthread_cond_wait(&pipe->cond_ready, &pipe->mu);
        }

        if (!slot) {
            pthread_mutex_unlock(&pipe->mu);
            break;
        }

        res = &pipe->results[slot->config_idx];
        config = &pipe->configs[slot->config_idx];
        pthread_mutex_unlock(&pipe->mu);

        t0 = now_us();
        if (FINAL_REFINE_MODE) {
            if (!slot->refinement && rover_open_sq_refine(&slot->res_cache,&slot->refinement))rc=-1;
            else {
                int count=slot->actual_count;
                if(count>FINAL_REFINE_CANDIDATES)count=FINAL_REFINE_CANDIDATES;
                rc=retrieval_final_refine(slot->refinement,slot->query,slot->candidate_ids,
                    slot->candidate_dists,count,config->top_k,NULL);
                if(rc>0)memcpy(result_ids,slot->candidate_ids,(size_t)rc*sizeof(*result_ids));
            }
        } else if (!rover_config.correction) {
            rc=slot->actual_count<config->top_k?slot->actual_count:config->top_k;
            memcpy(result_ids,slot->candidate_ids,(size_t)rc*sizeof(*result_ids));
        } else {
        rc = demo_rerank_candidates(slot->query,
                                    config,
                                    slot->candidate_ids,
                                    slot->candidate_dists,
                                    slot->actual_count,
                                    result_ids,
                                    slot->res_cache);
        }
        if (rc < 0) {
            pthread_mutex_lock(&pipe->mu);
            pipe->ret = -1;
            pipe->state[slot_idx] = RETRIEVAL_SLOT_FREE;
            pthread_cond_broadcast(&pipe->cond_ready);
            pthread_cond_broadcast(&pipe->cond_free);
            pthread_mutex_unlock(&pipe->mu);
            break;
        }


        if (slot->query_idx<rover_config.nq && rover_write_result(slot->query_idx,result_ids,slot->candidate_dists,rc)) {
            pthread_mutex_lock(&pipe->mu); pipe->ret=-1; pthread_mutex_unlock(&pipe->mu);
        }
        if(slot->query_idx<rover_config.nq)demo_accumulate_query_recall(res, result_ids, rc, slot->gt_row);
        res->phase2_us_sum += now_us() - t0;

        residual_cache_flush_stats(slot->res_cache, &g_residual_stats,
                                   RESIDUAL_CACHE_ROLE_RERANK);

        pthread_mutex_lock(&pipe->mu);
        pipe->state[slot_idx] = RETRIEVAL_SLOT_FREE;
        pthread_cond_signal(&pipe->cond_free);
        pthread_mutex_unlock(&pipe->mu);
    }

    return NULL;
}

static void demo_db_buffer_reset(DbSharedBuffer *db_shared) {
    if (!db_shared || !db_shared->layout) {
        return;
    }

    memset(db_shared->layout, 0, sizeof(*db_shared->layout));
    DB_WRITE_BARRIER();
}

static void demo_db_buffer_destroy(DbSharedBuffer *db_shared, int domain_id) {
    if (!db_shared) {
        return;
    }

    if (db_shared->mapped && db_shared->fd >= 0 && fastrpc_munmap) {
        fastrpc_munmap(domain_id, db_shared->fd,
                       db_shared->layout, sizeof(DbSharedLayout));
        db_shared->mapped = 0;
    }
    db_shared->fd = -1;

    if (db_shared->layout) {
        demo_free_maybe_shared(db_shared->layout, db_shared->uses_rpcmem);
        db_shared->layout = NULL;
        db_shared->buf = NULL;
        db_shared->stats = NULL;
    }
    db_shared->uses_rpcmem = 0;
}

static int demo_db_buffer_init(DbSharedBuffer *db_shared, int domain_id) {
    const size_t buf_bytes = sizeof(DbSharedLayout);
    int fd;
    int mret;

    if (!db_shared) {
        return -1;
    }

    memset(db_shared, 0, sizeof(*db_shared));
    db_shared->fd = -1;
    db_shared->layout = (DbSharedLayout *)rpcmem_alloc(
        RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, (int)buf_bytes);
    db_shared->uses_rpcmem = (db_shared->layout != NULL);
    if (!db_shared->layout) {
        printf("  ERROR: rpcmem_alloc(%zu) failed for expansion buffer\n", buf_bytes);
        return -1;
    }
    db_shared->buf = db_shared->layout->buf;
    db_shared->stats = &db_shared->layout->stats;

    memset(db_shared->layout, 0, buf_bytes);
    fd = rpcmem_to_fd(db_shared->layout);
    if (fd < 0 || !fastrpc_mmap) {
        printf("  ERROR: rpcmem_to_fd failed (fd=%d, fastrpc_mmap=%p)\n",
               fd, (void *)fastrpc_mmap);
        demo_db_buffer_destroy(db_shared, domain_id);
        return -1;
    }

    mret = fastrpc_mmap(domain_id, fd, db_shared->layout, 0, buf_bytes, FASTRPC_MAP_FD);
    if (mret != 0) {
        printf("  ERROR: fastrpc_mmap(expansion_buf) failed 0x%x\n", mret);
        demo_db_buffer_destroy(db_shared, domain_id);
        return -1;
    }

    db_shared->fd = fd;
    db_shared->mapped = 1;

    return 0;
}

static void demo_sqmd_buffer_reset(SqmdSharedBuffer *sqmd_shared,
                                   int lane_count) {
    if (!sqmd_shared || !sqmd_shared->ctrl) {
        return;
    }

    sqmd_shared_control_init(sqmd_shared->ctrl, lane_count);
    DB_WRITE_BARRIER();
    sqmd_shared->ctrl->ctrl_state = SQMD_CTRL_ABORT;
    DB_WRITE_BARRIER();
}

static void demo_sqmd_buffer_destroy(SqmdSharedBuffer *sqmd_shared,
                                     int domain_id) {
    if (!sqmd_shared) {
        return;
    }

    if (sqmd_shared->mapped && sqmd_shared->fd >= 0 && fastrpc_munmap) {
        fastrpc_munmap(domain_id, sqmd_shared->fd,
                       sqmd_shared->ctrl, sizeof(SqmdSharedControl));
        sqmd_shared->mapped = 0;
    }
    sqmd_shared->fd = -1;

    if (sqmd_shared->ctrl) {
        demo_free_maybe_shared(sqmd_shared->ctrl, sqmd_shared->uses_rpcmem);
        sqmd_shared->ctrl = NULL;
    }
    sqmd_shared->uses_rpcmem = 0;
}

static int demo_sqmd_buffer_init(SqmdSharedBuffer *sqmd_shared,
                                 int domain_id,
                                 int lane_count) {
    const size_t buf_bytes = sizeof(SqmdSharedControl);
    int fd;
    int mret;

    if (!sqmd_shared) {
        return -1;
    }

    memset(sqmd_shared, 0, sizeof(*sqmd_shared));
    sqmd_shared->fd = -1;
    sqmd_shared->ctrl = (SqmdSharedControl *)rpcmem_alloc(
        RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, (int)buf_bytes);
    sqmd_shared->uses_rpcmem = (sqmd_shared->ctrl != NULL);
    if (!sqmd_shared->ctrl) {
        printf("  ERROR: rpcmem_alloc(%zu) failed for SQMD control\n",
               buf_bytes);
        return -1;
    }

    demo_sqmd_buffer_reset(sqmd_shared, lane_count);
    fd = rpcmem_to_fd(sqmd_shared->ctrl);
    if (fd < 0 || !fastrpc_mmap) {
        printf("  ERROR: rpcmem_to_fd failed for SQMD (fd=%d, fastrpc_mmap=%p)\n",
               fd, (void *)fastrpc_mmap);
        demo_sqmd_buffer_destroy(sqmd_shared, domain_id);
        return -1;
    }

    mret = fastrpc_mmap(domain_id,
                        fd,
                        sqmd_shared->ctrl,
                        0,
                        buf_bytes,
                        FASTRPC_MAP_FD);
    if (mret != 0) {
        printf("  ERROR: fastrpc_mmap(SQMD control) failed 0x%x\n", mret);
        demo_sqmd_buffer_destroy(sqmd_shared, domain_id);
        return -1;
    }

    sqmd_shared->fd = fd;
    sqmd_shared->mapped = 1;
    return 0;
}

static void demo_db_buffers_destroy_all(int domain_id) {
    int i;

    for (i = 0; i < DEMO_RPC_MAX_CONCURRENT; i++) {
        demo_db_buffer_destroy(&g_db_shared[i], domain_id);
    }
}

static void *rpc_search_worker_fn(void *arg) {
    RetrievalRpcWorkerArgs *worker = (RetrievalRpcWorkerArgs *)arg;
    RetrievalPipeline *pipe = worker->pipe;
    RetrievalSlot *slot = &pipe->slots[worker->slot_idx];
    unsigned long long t0;
    unsigned long long t1;

    t0 = now_us();
    worker->ret = calculator_retrieval_search(
        worker->handle,
        slot->query, DATA_DIM,
        worker->config->entry_points,
        worker->config->expand_count,
        worker->config->pool_size,
        worker->num_out,
        slot->candidate_ids, worker->num_out,
        slot->candidate_dists, worker->num_out,
        &slot->actual_count);
    t1 = now_us();

    if (worker->ret != 0) {
        pthread_mutex_lock(&pipe->mu);
        if (pipe->ret == 0) {
            pipe->ret = worker->ret;
        }
        pthread_mutex_unlock(&pipe->mu);
        pipeline_abort_slot(pipe, worker->slot_idx);
        return NULL;
    }


    pthread_mutex_lock(&pipe->mu);
    pipe->results[slot->config_idx].phase1_us_sum += t1 - t0;
    pthread_mutex_unlock(&pipe->mu);
    pipeline_mark_ready(pipe, worker->slot_idx);
    return NULL;
}


static void *db_correction_thread_fn(void *arg) {
    DbCorrectionCtx *ctx = (DbCorrectionCtx *)arg;
    ExpansionBuffer *buf = ctx->buf;

    while (!ctx->stop) {
        int i;
        int any_work = 0;

        for (i = 0; i < 2; i++) {
            if (buf[i].state != DB_STATE_DSP_DONE)
                continue;
            DB_READ_BARRIER();

            {
                const float *query = (i == 0) ? ctx->query_a : ctx->query_b;
                ResidualBlockCache *cache = (i == 0) ? ctx->res_cache_a
                                                     : ctx->res_cache_b;
                int j;

                for (j = 0; j < buf[i].count; j++) {
                    buf[i].corrected_dists[j] = !rover_config.correction ? buf[i].coarse_dists[j] :
                        retrieval_demo_exact_distance_residual_cached(
                            query,
                            ctx->dataset,
                            buf[i].neighbor_ids[j],
                            cache);
                }
                if (buf[i].count > 0) {
                    ctx->corr_batches++;
                    ctx->corr_candidates += (uint64_t)buf[i].count;
                }
            }

            DB_WRITE_BARRIER();
            buf[i].state = DB_STATE_CPU_DONE;
            any_work = 1;
        }

        if (!any_work) {
            sched_yield();
        }
    }

    return NULL;
}


static void *rpc_search_db_worker_fn(void *arg) {
    RetrievalRpcDbWorkerArgs *worker = (RetrievalRpcDbWorkerArgs *)arg;
    RetrievalPipeline *pipe = worker->pipe;
    RetrievalSlot *slot_a = &pipe->slots[worker->slot_idx_a];
    RetrievalSlot *slot_b = &pipe->slots[worker->slot_idx_b];
    DbSharedBuffer *db_shared = worker->db_shared;
    DbCorrectionCtx corr_ctx;
    pthread_t corr_thread;
    int corr_started = 0;
    unsigned long long t0, t1;

    if (!db_shared || !db_shared->buf || db_shared->fd < 0) {
        worker->ret = -1;
        pipeline_abort_slot(pipe, worker->slot_idx_a);
        pipeline_abort_slot(pipe, worker->slot_idx_b);
        return NULL;
    }
    demo_db_buffer_reset(db_shared);

    memset(&corr_ctx, 0, sizeof(corr_ctx));
    corr_ctx.buf         = db_shared->buf;
    corr_ctx.query_a     = slot_a->query;
    corr_ctx.query_b     = slot_b->query;
    corr_ctx.dataset     = &g_dataset;
    corr_ctx.res_cache_a = slot_a->res_cache;
    corr_ctx.res_cache_b = slot_b->res_cache;
    if (pthread_create(&corr_thread, NULL,
                       db_correction_thread_fn, &corr_ctx) != 0) {
        worker->ret = -1;
        pipeline_abort_slot(pipe, worker->slot_idx_a);
        pipeline_abort_slot(pipe, worker->slot_idx_b);
        return NULL;
    }
    corr_started = 1;

    t0 = now_us();
    worker->ret = calculator_retrieval_search_db(
        worker->handle,
        slot_a->query, DATA_DIM,
        slot_b->query, DATA_DIM,
        db_shared->fd,
        worker->config->entry_points,
        worker->config->expand_count,
        worker->config->pool_size,
        worker->num_out,
        slot_a->candidate_ids, worker->num_out,
        slot_a->candidate_dists, worker->num_out,
        &slot_a->actual_count,
        slot_b->candidate_ids, worker->num_out,
        slot_b->candidate_dists, worker->num_out,
        &slot_b->actual_count);
    t1 = now_us();
    corr_ctx.stop = 1;
    if (corr_started)
        pthread_join(corr_thread, NULL);

    residual_cache_flush_stats(slot_a->res_cache, &g_residual_stats,
                               RESIDUAL_CACHE_ROLE_CORRECTION);
    residual_cache_flush_stats(slot_b->res_cache, &g_residual_stats,
                               RESIDUAL_CACHE_ROLE_CORRECTION);

    if (worker->ret != 0) {
        pthread_mutex_lock(&pipe->mu);
        if (pipe->ret == 0)
            pipe->ret = worker->ret;
        pthread_mutex_unlock(&pipe->mu);
        pipeline_abort_slot(pipe, worker->slot_idx_a);
        pipeline_abort_slot(pipe, worker->slot_idx_b);
        return NULL;
    }

    pthread_mutex_lock(&pipe->mu);
    pipe->results[slot_a->config_idx].phase1_us_sum += t1 - t0;
    if (db_shared->stats) {
        DemoResult *res = &pipe->results[slot_a->config_idx];
        res->cpu_wait_us += db_shared->stats->dsp_wait_cpu_done_us;
        res->dsp_invalidate_us += db_shared->stats->dsp_invalidate_us;
        res->dsp_wait_poll_count += db_shared->stats->dsp_wait_poll_count;
    }
    pthread_mutex_unlock(&pipe->mu);

    pipeline_mark_ready(pipe, worker->slot_idx_a);
    pipeline_mark_ready(pipe, worker->slot_idx_b);
    return NULL;
}

static void *rpc_search_sqmd_worker_fn(void *arg) {
    RetrievalRpcSqmdWorkerArgs *worker = (RetrievalRpcSqmdWorkerArgs *)arg;

    if (!worker || !worker->sqmd_shared || !worker->sqmd_shared->ctrl ||
        worker->sqmd_shared->fd < 0 || !worker->query) {
        if (worker) {
            worker->ret = -1;
            worker->done = 1;
        }
        return NULL;
    }

    worker->ret = calculator_retrieval_expand_sqmd(
        worker->handle,
        worker->query, DATA_DIM,
        DATA_DIM,
        worker->sqmd_shared->fd,
        worker->lane_count,
        worker->work_items_cap);
    if (worker->ret != AEE_SUCCESS) {
        printf("  ERROR: SQMD RPC returned 0x%x"
               " (fd=%d lanes=%d cap=%d ctrl_state=%d)\n",
               worker->ret,
               worker->sqmd_shared->fd,
               worker->lane_count,
               worker->work_items_cap,
               worker->sqmd_shared->ctrl->ctrl_state);
        fflush(stdout);
    }
    DB_WRITE_BARRIER();
    worker->done = 1;
    return NULL;
}

static int sqmd_wait_for_rpc_ready(SqmdSharedControl *ctrl,
                                   const RetrievalRpcSqmdWorkerArgs *worker) {
    unsigned long long start_us;

    if (!ctrl)
        return -1;

    start_us = now_us();
    while (now_us() - start_us < 5000000ULL) {
        DB_READ_BARRIER();
        if (ctrl->ctrl_state == SQMD_CTRL_RUNNING) {
            ctrl->startup_slot.acknowledged = 1;
            DB_WRITE_BARRIER();
            return 0;
        }
        if (ctrl->ctrl_state == SQMD_CTRL_DONE) {
            printf("  ERROR: SQMD wait ready failed: ctrl already DONE\n");
            fflush(stdout);
            return -1;
        }
        if (worker && worker->done) {
            printf("  ERROR: SQMD wait ready failed: RPC exited ret=0x%x ctrl_state=%d\n",
                   worker->ret, ctrl->ctrl_state);
            fflush(stdout);
            return -1;
        }
        sched_yield();
    }

    printf("  ERROR: SQMD wait ready failed: timeout ctrl_state=%d\n",
           ctrl->ctrl_state);
    fflush(stdout);
    return -1;
}

static void sqmd_cpu_wait_yield(void *ctx) {
    sqmdCpuSearchState *st = (sqmdCpuSearchState *)ctx;

    if (st)
        st->lane_wait_us++;
    sched_yield();
}

static uint64_t sqmd_host_now_us(void *ctx) {
    (void)ctx;
    return (uint64_t)now_us();
}

static const char *sqmd_abort_stage_name(int stage) {
    switch (stage) {
    case SQMD_ABORT_NONE:
        return "none";
    case SQMD_ABORT_DISPATCH:
        return "dispatch";
    case SQMD_ABORT_COLLECT:
        return "collect";
    case SQMD_ABORT_CTRL:
        return "ctrl";
    case SQMD_ABORT_APPLY_HEADER:
        return "apply_header";
    case SQMD_ABORT_APPLY_RESULT:
        return "apply_result";
    case SQMD_ABORT_DSP_PROCESS:
        return "dsp_process";
    case SQMD_ABORT_TURN_WAIT_TIMEOUT:
        return "turn_wait_timeout";
    default:
        return "unknown";
    }
}

static void sqmd_print_abort_detail(const char *label,
                                    int stage,
                                    int rc,
                                    int turn,
                                    int lane,
                                    int buffer,
                                    int item,
                                    int state,
                                    int buf_turn,
                                    int work_count,
                                    int result_count,
                                    int neighbor_id,
                                    int source_id) {
    printf("  ERROR: SQMD %s abort detail stage=%s(%d) rc=%d"
           " turn=%d lane=%d buf=%d item=%d state=%d buf_turn=%d"
           " work=%d result=%d neighbor=%d source=%d\n",
           label,
           sqmd_abort_stage_name(stage),
           stage,
           rc,
           turn,
           lane,
           buffer,
           item,
           state,
           buf_turn,
           work_count,
           result_count,
           neighbor_id,
           source_id);
}

static void sqmd_print_abort_buffers(const SqmdSharedControl *ctrl,
                                     int abort_turn) {
    int lane_count;
    int lane;

    if (!ctrl)
        return;

    lane_count = ctrl->lane_count;
    if (lane_count <= 0 || lane_count > rover_config.lanes)
        lane_count = rover_config.lanes;

    for (lane = 0; lane < lane_count; lane++) {
        int buf_idx;

        for (buf_idx = 0; buf_idx < SQMD_BUFFERS_PER_LANE; buf_idx++) {
            const SqmdLaneBuffer *buf = &ctrl->lanes[lane].buf[buf_idx];
            int state = buf->state;
            int turn_id = buf->turn_id;
            int work_count = buf->work_count;
            int result_count = buf->result_count;
            int first_work_neighbor = -1;
            int first_work_source = -1;
            int first_result_neighbor = -1;
            int first_result_source = -1;

            if (state == SQMD_BUF_EMPTY &&
                turn_id != abort_turn &&
                turn_id != abort_turn - 1 &&
                turn_id != abort_turn + 1) {
                continue;
            }
            if (work_count > 0 && work_count <= SQMD_MAX_WORK_ITEMS) {
                first_work_neighbor = buf->work[0].neighbor_id;
                first_work_source = buf->work[0].source_id;
            }
            if (result_count > 0 && result_count <= SQMD_MAX_WORK_ITEMS) {
                first_result_neighbor = buf->result[0].neighbor_id;
                first_result_source = buf->result[0].source_id;
            }

            printf("    SQMD buf lane=%d buf=%d state=%d turn=%d"
                   " work=%d result=%d first_work=%d/%d"
                   " first_result=%d/%d\n",
                   lane,
                   buf_idx,
                   state,
                   turn_id,
                   work_count,
                   result_count,
                   first_work_neighbor,
                   first_work_source,
                   first_result_neighbor,
                   first_result_source);
        }
    }
}

static void demo_sqmd_accumulate_stats(DemoResult *res,
                                       const SqmdSharedStats *stats) {
    if (!res || !stats)
        return;

    DB_READ_BARRIER();
    res->sqmd_work_items += stats->sqmd_work_items;
    res->sqmd_dedup_skips += stats->sqmd_dedup_skips;
    res->sqmd_empty_turns += stats->sqmd_empty_turns;
    res->sqmd_dsp_compute_us += stats->sqmd_dsp_compute_us;
    res->sqmd_cpu_correct_us += stats->sqmd_cpu_correct_us;
    res->sqmd_cpu_wait_dsp_us += stats->sqmd_cpu_wait_dsp_us;
    res->sqmd_dsp_wait_cpu_us += stats->sqmd_dsp_wait_cpu_us;
    res->sqmd_dsp_invalidate_us += stats->sqmd_dsp_invalidate_us;
    res->sqmd_lane_wait_us += stats->sqmd_lane_wait_us;
    res->sqmd_dsp_done_batches += stats->sqmd_dsp_done_batches;
    res->sqmd_dsp_wait_poll_count += stats->sqmd_dsp_wait_poll_count;
    res->sqmd_abort_count += stats->sqmd_abort_count;
    res->sqmd_source_jobs_created += stats->sqmd_source_jobs_created;
    res->sqmd_source_jobs_retired += stats->sqmd_source_jobs_retired;
    res->sqmd_speculative_sources += stats->sqmd_speculative_sources;
    res->sqmd_source_window_starvation +=
        stats->sqmd_source_window_starvation;
    if (res->sqmd_inflight_sources_highwater <
        stats->sqmd_inflight_sources_highwater) {
        res->sqmd_inflight_sources_highwater =
            stats->sqmd_inflight_sources_highwater;
    }
}

static int sqmd_parse_max_inflight_sources(int *max_inflight_sources) {
    const char *value;

    if (!max_inflight_sources)
        return -1;

    value = getenv("SQMD_MAX_INFLIGHT_SOURCES");
    if (!value || value[0] == '\0' || strcmp(value, "1") == 0) {
        *max_inflight_sources = 1;
        return 0;
    }
    if (strcmp(value, "2") == 0) {
        *max_inflight_sources = 2;
        return 0;
    }

    printf("  ERROR: SQMD_MAX_INFLIGHT_SOURCES must be 1 or 2 (got '%s')\n",
           value);
    fflush(stdout);
    return -1;
}

static int sqmd_parse_split_distance(int *split_enabled) {
    const char *value;

    if (!split_enabled)
        return -1;
    value = getenv("SQMD_SPLIT_DISTANCE");
    if (!value || value[0] == '\0') {
        *split_enabled = SQMD_SPLIT_DISTANCE_DEFAULT;
        return 0;
    }
    if (strcmp(value, "0") == 0 || strcmp(value, "off") == 0) {
        *split_enabled = 0;
        return 0;
    }
    if (strcmp(value, "1") == 0 || strcmp(value, "on") == 0) {
        *split_enabled = 1;
        return 0;
    }
    printf("  ERROR: SQMD_SPLIT_DISTANCE must be 0/off or 1/on (got '%s')\n",
           value);
    fflush(stdout);
    return -1;
}

static int sqmd_query_buffers_quiescent(const SqmdSharedControl *ctrl) {
    int lane;
    int buffer;

    if (!ctrl || ctrl->lane_count <= 0 ||
        ctrl->lane_count > rover_config.lanes)
        return 0;

    for (lane = 0; lane < ctrl->lane_count; lane++) {
        for (buffer = 0; buffer < SQMD_BUFFERS_PER_LANE; buffer++) {
            if (ctrl->lanes[lane].buf[buffer].state != SQMD_BUF_EMPTY)
                return 0;
        }
    }
    return 1;
}

typedef struct SqmdHostDistanceCtx {
    const RetrievalDemoEncodedDataset *dataset;
    const uint8_t *query_l1;
} SqmdHostDistanceCtx;

static int sqmd_host_distance_fn(const SqmdWorkItem *work,
                                 float *coarse_dist,
                                 void *ctx) {
    SqmdHostDistanceCtx *dist_ctx = (SqmdHostDistanceCtx *)ctx;

    if (!work || !coarse_dist || !dist_ctx || !dist_ctx->dataset ||
        !dist_ctx->query_l1)
        return -1;
    if (work->neighbor_id < 0 ||
        work->neighbor_id >= dist_ctx->dataset->num_vectors)
        return -1;

    *coarse_dist = demo_compute_l1_distance(
        dist_ctx->dataset,
        dist_ctx->query_l1,
        demo_l1_code_ptr(dist_ctx->dataset, work->neighbor_id));
    return 0;
}

typedef struct SqmdHostCorrectionCtx {
    const float *query;
    const RetrievalDemoEncodedDataset *dataset;
    ResidualBlockCache *res_cache;
} SqmdHostCorrectionCtx;

typedef struct SqmdSplitQueryCtx {
    const RetrievalDemoEncodedDataset *dataset;
    ResidualBlockCache *res_cache;
    const float *bias;
    uint8_t query_u8[RETRIEVAL_ASYNC_DEMO_MAX_DIM];
    RetrievalQueryU8Stats query_stats;
} SqmdSplitQueryCtx;

static int sqmd_split_query_init(
        SqmdSplitQueryCtx *ctx,
        const float *query,
        const RetrievalDemoEncodedDataset *dataset,
        ResidualBlockCache *res_cache,
        const float *bias) {
    if (!ctx || !query || !dataset || !bias || dataset->l1_bits != 4 ||
        (dataset->residual_bits != 4 && dataset->residual_bits != 8) ||
        dataset->dim <= 0 ||
        dataset->dim > RETRIEVAL_ASYNC_DEMO_MAX_DIM ||
        !dataset->residual_qparams)
        return -1;
    memset(ctx, 0, sizeof(*ctx));
    ctx->dataset = dataset;
    ctx->res_cache = res_cache;
    ctx->bias = bias;
    return retrieval_quantize_query_l1_aligned_u8(
        query, dataset->dim, dataset->l1_qparams.scale,
        dataset->l1_qparams.zero_point, ctx->query_u8,
        &ctx->query_stats);
}

static float sqmd_split_residual_delta_fn(int neighbor_id,
                                          int source_id,
                                          void *opaque) {
    SqmdSplitQueryCtx *ctx = (SqmdSplitQueryCtx *)opaque;
    const RetrievalDemoEncodedDataset *dataset;
    const QCResidualQuantParams *qp;
    const uint8_t *residual;
    double dot;
    double delta;

    (void)source_id;
    if (!ctx || !ctx->dataset || !ctx->bias || neighbor_id < 0 ||
        neighbor_id >= ctx->dataset->num_vectors)
        return NAN;
    dataset = ctx->dataset;
    qp = &dataset->residual_qparams[neighbor_id];
    if (ctx->res_cache) {
        residual = residual_cache_get(ctx->res_cache, neighbor_id);
    } else if (dataset->residual_codes) {
        residual = dataset->residual_codes +
            (size_t)neighbor_id * (size_t)dataset->residual_code_bytes;
    } else {
        residual = NULL;
    }
    if (!residual)
        return NAN;

    if (dataset->residual_bits == 8) {
        dot = retrieval_compute_u8_u8_residual_dot(
            ctx->query_u8, ctx->query_stats.sum_q, residual, dataset->dim,
            dataset->l1_qparams.scale, dataset->l1_qparams.zero_point,
            qp->scale, qp->zero_point);
    } else {
        dot = retrieval_compute_u8_u4_residual_dot(
            ctx->query_u8, ctx->query_stats.sum_q, residual, dataset->dim,
            dataset->l1_qparams.scale, dataset->l1_qparams.zero_point,
            qp->scale, qp->zero_point);
    }
    delta = (double)ctx->bias[neighbor_id] - 2.0 * dot;
    return isfinite(delta) ? (float)delta : NAN;
}

static float sqmd_split_distance_for_id(SqmdSplitQueryCtx *ctx, int id) {
    float coarse;
    float delta;

    if (!ctx || id < 0 || id >= ctx->dataset->num_vectors)
        return FLT_MAX;
    coarse = retrieval_compute_u8_u4_l1_distance(
        ctx->query_u8, demo_l1_code_ptr(ctx->dataset, id),
        ctx->dataset->dim, ctx->dataset->l1_qparams.scale);
    delta = sqmd_split_residual_delta_fn(id, -1, ctx);
    if (!isfinite(coarse) || !isfinite(delta))
        return FLT_MAX;
    coarse += delta;
    if (coarse < 0.0f && coarse > -1.0e-4f)
        coarse = 0.0f;
    return coarse;
}

static float sqmd_host_correction_fn(int neighbor_id,
                                     int source_id,
                                     float coarse_dist,
                                     void *ctx) {
    SqmdHostCorrectionCtx *corr_ctx = (SqmdHostCorrectionCtx *)ctx;

    (void)source_id;
    if (!rover_config.correction) return coarse_dist;
    (void)coarse_dist;
    if (!corr_ctx || !corr_ctx->query || !corr_ctx->dataset)
        return coarse_dist;
    return retrieval_demo_exact_distance_residual_cached(corr_ctx->query,
                                                         corr_ctx->dataset,
                                                         neighbor_id,
                                                         corr_ctx->res_cache);
}

static int sqmd_make_seed(const RetrievalDemoEncodedDataset *dataset,
                          const float *query,
                          SqmdSplitQueryCtx *split_ctx,
                          int entry_points,
                          int graph_entry,
                          unsigned int seed,
                          int *seed_ids,
                          float *seed_dists,
                          int seed_cap) {
    int count;
    int index;

    if (!dataset || !query || !seed_ids || !seed_dists || entry_points <= 0)
        return -1;

    count = retrieval_sqmd_make_seed_ids(dataset->num_vectors,
                                         entry_points,
                                         graph_entry,
                                         seed,
                                         seed_ids,
                                         seed_cap);
    if (count <= 0)
        return count;

    uint8_t coarse_query[RETRIEVAL_ASYNC_DEMO_MAX_DIM];
    if(!rover_config.correction && demo_quantize_query_l1(query,dataset,coarse_query))return -1;
    for (index = 0; index < count; index++) {
        if(!rover_config.correction) {
            seed_dists[index]=demo_compute_l1_distance(dataset,coarse_query,demo_l1_code_ptr(dataset,seed_ids[index]));
            continue;
        }
        seed_dists[index] = split_ctx
            ? sqmd_split_distance_for_id(split_ctx, seed_ids[index])
            : retrieval_demo_exact_distance_residual_cached(query,
                                                            dataset,
                                                            seed_ids[index],
                                                            NULL);
        if (!isfinite(seed_dists[index]) || seed_dists[index] < 0.0f)
            return -1;
    }

    return count;
}

static int sqmd_search_one_query_fake_dsp(const RetrievalDemoEncodedDataset *dataset,
                                          const int32_t *graph,
                                          int graph_degree,
                                          int graph_entry,
                                          const float *query,
                                          const RetrievalDemoConfig *config,
                                          ResidualBlockCache *res_cache,
                                          int *out_ids,
                                          float *out_dists,
                                          int out_cap) {
    int *seed_ids = NULL;
    float *seed_dists = NULL;
    uint8_t query_l1[RETRIEVAL_ASYNC_DEMO_MAX_DIM];
    SqmdHostDistanceCtx dist_ctx;
    SqmdHostCorrectionCtx corr_ctx;
    int seed_count;
    int result_count;

    if (!dataset || !graph || !query || !config || !out_ids || !out_dists ||
        config->entry_points <= 0 || out_cap <= 0)
        return -1;
    if (demo_quantize_query_l1(query, dataset, query_l1) != 0)
        return -1;

    seed_ids = (int *)malloc((size_t)config->entry_points * sizeof(*seed_ids));
    seed_dists = (float *)malloc((size_t)config->entry_points *
                                 sizeof(*seed_dists));
    if (!seed_ids || !seed_dists) {
        free(seed_ids);
        free(seed_dists);
        return -1;
    }

    seed_count = sqmd_make_seed(dataset,
                                query,
                                NULL,
                                config->entry_points,
                                graph_entry,
                                ENTRY_POINT_SEED,
                                seed_ids,
                                seed_dists,
                                config->entry_points);
    if (seed_count <= 0) {
        free(seed_ids);
        free(seed_dists);
        return -1;
    }

    dist_ctx.dataset = dataset;
    dist_ctx.query_l1 = query_l1;
    corr_ctx.query = query;
    corr_ctx.dataset = dataset;
    corr_ctx.res_cache = res_cache;

    result_count = sqmd_cpu_state_run_fake_dsp(graph,
                                               graph_degree,
                                               dataset->num_vectors,
                                               config->pool_size,
                                               config->entry_points,
                                               config->expand_count,
                                               seed_ids,
                                               seed_dists,
                                               seed_count,
                                               rover_config.lanes,
                                               sqmd_host_distance_fn,
                                               &dist_ctx,
                                               sqmd_host_correction_fn,
                                               &corr_ctx,
                                               out_ids,
                                               out_dists,
                                               out_cap);
    free(seed_ids);
    free(seed_dists);
    return result_count;
}

static int sqmd_search_one_query_rpc(const RetrievalDemoEncodedDataset *dataset,
                                     const int32_t *graph,
                                     int graph_degree,
                                     int graph_entry,
                                     const float *query,
                                     const RetrievalDemoConfig *config,
                                     ResidualBlockCache *res_cache,
                                     remote_handle64 handle,
                                     SqmdSharedBuffer *sqmd_shared,
                                     int max_inflight_sources,
                                     int *out_ids,
                                     float *out_dists,
                                     int out_cap) {
    sqmdCpuSearchState st;
    SqmdHostCorrectionCtx corr_ctx;
    RetrievalRpcSqmdWorkerArgs rpc_arg;
    pthread_t rpc_thread;
    int rpc_started = 0;
    int rpc_ready = 0;
    int *seed_ids = NULL;
    float *seed_dists = NULL;
    int seed_count;
    int result_count = -1;

    memset(&st, 0, sizeof(st));
    memset(&rpc_arg, 0, sizeof(rpc_arg));

    if (!dataset || !graph || !query || !config || !sqmd_shared ||
        !sqmd_shared->ctrl || sqmd_shared->fd < 0 ||
        !out_ids || !out_dists || config->entry_points <= 0 ||
        out_cap <= 0)
        return -1;

    seed_ids = (int *)malloc((size_t)config->entry_points * sizeof(*seed_ids));
    seed_dists = (float *)malloc((size_t)config->entry_points *
                                 sizeof(*seed_dists));
    if (!seed_ids || !seed_dists)
        goto cleanup;

    seed_count = sqmd_make_seed(dataset,
                                query,
                                NULL,
                                config->entry_points,
                                graph_entry,
                                ENTRY_POINT_SEED,
                                seed_ids,
                                seed_dists,
                                config->entry_points);
    if (seed_count <= 0)
        goto cleanup;

    demo_sqmd_buffer_reset(sqmd_shared, rover_config.lanes);

    if (sqmd_cpu_state_init(&st,
                            dataset->num_vectors,
                            config->pool_size,
                            config->entry_points,
                            config->expand_count,
                            query,
                            dataset,
                            res_cache) != SQMD_STATUS_OK)
        goto cleanup;
    if (sqmd_cpu_state_seed(&st, seed_ids, seed_dists, seed_count) !=
        SQMD_STATUS_OK)
        goto cleanup;

    rpc_arg.handle = handle;
    rpc_arg.query = query;
    rpc_arg.sqmd_shared = sqmd_shared;
    rpc_arg.lane_count = rover_config.lanes;
    rpc_arg.work_items_cap = SQMD_MAX_WORK_ITEMS;
    if (pthread_create(&rpc_thread,
                       NULL,
                       rpc_search_sqmd_worker_fn,
                       &rpc_arg) != 0)
        goto cleanup;
    rpc_started = 1;

    if (sqmd_wait_for_rpc_ready(sqmd_shared->ctrl, &rpc_arg) != 0)
        goto cleanup;
    rpc_ready = 1;

    corr_ctx.query = query;
    corr_ctx.dataset = dataset;
    corr_ctx.res_cache = res_cache;
    result_count = sqmd_cpu_state_run_source_pipeline(
        &st,
        sqmd_shared->ctrl,
        max_inflight_sources,
        graph,
        graph_degree,
        dataset->num_vectors,
        sqmd_host_correction_fn,
        &corr_ctx,
        sqmd_cpu_wait_yield,
        &st,
        sqmd_host_now_us,
        NULL,
        out_ids,
        out_dists,
        out_cap);
    if (result_count < 0) {
        const SqmdSharedStats *stats = &sqmd_shared->ctrl->stats;

        printf("  ERROR: SQMD shared-control loop failed"
               " ctrl_state=%d turn=%d stats_abort=%llu"
               " work=%llu dedup=%llu empty=%llu\n",
               sqmd_shared->ctrl->ctrl_state,
               sqmd_shared->ctrl->turn_id,
               (unsigned long long)sqmd_shared->ctrl->stats.sqmd_abort_count,
               (unsigned long long)st.work_items,
               (unsigned long long)st.dedup_skips,
               (unsigned long long)st.empty_turns);
        sqmd_print_abort_detail("CPU",
                                st.abort_stage,
                                st.abort_rc,
                                st.abort_turn,
                                st.abort_lane,
                                st.abort_buffer,
                                st.abort_item,
                                st.abort_state,
                                st.abort_buf_turn,
                                st.abort_work_count,
                                st.abort_result_count,
                                st.abort_neighbor_id,
                                st.abort_source_id);
        sqmd_print_abort_detail("shared",
                                stats->sqmd_abort_stage,
                                stats->sqmd_abort_rc,
                                stats->sqmd_abort_turn,
                                stats->sqmd_abort_lane,
                                stats->sqmd_abort_buffer,
                                stats->sqmd_abort_item,
                                stats->sqmd_abort_state,
                                stats->sqmd_abort_buf_turn,
                                stats->sqmd_abort_work_count,
                                stats->sqmd_abort_result_count,
                                stats->sqmd_abort_neighbor_id,
                                stats->sqmd_abort_source_id);
        sqmd_print_abort_buffers(sqmd_shared->ctrl, st.abort_turn);
        fflush(stdout);
    }

cleanup:
    if (result_count < 0 && sqmd_shared && sqmd_shared->ctrl) {
        if (rpc_started && !rpc_ready && !rpc_arg.done) {
            sqmd_shared->ctrl->startup_slot.cancel = 1;
            DB_WRITE_BARRIER();
        }
        DB_WRITE_BARRIER();
        sqmd_shared->ctrl->ctrl_state = SQMD_CTRL_ABORT;
        DB_WRITE_BARRIER();
    }
    if (rpc_started) {
        pthread_join(rpc_thread, NULL);
        if (result_count >= 0 && rpc_arg.ret != 0)
            result_count = -1;
    }
    sqmd_cpu_state_cleanup(&st);
    free(seed_ids);
    free(seed_dists);
    return result_count;
}


static void *rpc_search_sqmd_session_worker_fn(void *arg) {
    RetrievalRpcSqmdSessionArgs *s = (RetrievalRpcSqmdSessionArgs *)arg;

    if (!s || !s->sqmd_shared || !s->sqmd_shared->ctrl ||
        s->sqmd_shared->fd < 0) {
        if (s) {
            s->ret = -1;
            s->done = 1;
        }
        return NULL;
    }

    s->ret = calculator_retrieval_expand_sqmd_run(
        s->handle,
        s->sqmd_shared->fd,
        s->lane_count,
        s->work_items_cap);
    if (s->ret != AEE_SUCCESS) {
        printf("  ERROR: SQMD session RPC returned 0x%x"
               " (fd=%d lanes=%d cap=%d session_state=%d)\n",
               s->ret,
               s->sqmd_shared->fd,
               s->lane_count,
               s->work_items_cap,
               s->sqmd_shared->ctrl->session_state);
        fflush(stdout);
    }
    DB_WRITE_BARRIER();
    s->done = 1;
    return NULL;
}

static int sqmd_wait_for_session_ready(SqmdSharedControl *ctrl,
                                       const RetrievalRpcSqmdSessionArgs *s) {
    unsigned long long start_us;

    if (!ctrl)
        return -1;

    start_us = now_us();
    while (now_us() - start_us < 5000000ULL) {
        DB_READ_BARRIER();
        if (ctrl->session_state == SQMD_SESSION_RUNNING) {
            ctrl->startup_slot.acknowledged = 1;
            DB_WRITE_BARRIER();
            return 0;
        }
        if (s && s->done) {
            printf("  ERROR: SQMD session ready failed: RPC exited ret=0x%x"
                   " session_state=%d\n",
                   s->ret, ctrl->session_state);
            fflush(stdout);
            return -1;
        }
        sched_yield();
    }

    printf("  ERROR: SQMD session ready failed: timeout session_state=%d\n",
           ctrl->session_state);
    fflush(stdout);
    return -1;
}

static int sqmd_session_open(remote_handle64 handle,
                             SqmdSharedBuffer *sqmd_shared,
                             pthread_t *thread,
                             RetrievalRpcSqmdSessionArgs *args) {
    memset(args, 0, sizeof(*args));
    if (!sqmd_shared || !sqmd_shared->ctrl || sqmd_shared->fd < 0 || !thread)
        return -1;

    args->handle = handle;
    args->sqmd_shared = sqmd_shared;
    args->lane_count = rover_config.lanes;
    args->work_items_cap = SQMD_MAX_WORK_ITEMS;
    args->query_gen = 0;
    sqmd_shared->ctrl->startup_slot.cancel = 0;
    sqmd_shared->ctrl->startup_slot.acknowledged = 0;
    DB_WRITE_BARRIER();

    if (pthread_create(thread, NULL,
                       rpc_search_sqmd_session_worker_fn, args) != 0) {
        printf("  ERROR: SQMD session thread create failed\n");
        fflush(stdout);
        return -1;
    }
    args->started = 1;

    if (sqmd_wait_for_session_ready(sqmd_shared->ctrl, args) != 0) {

        sqmd_shared->ctrl->startup_slot.cancel = 1;
        DB_WRITE_BARRIER();
        sqmd_shared->ctrl->session_state = SQMD_SESSION_SHUTDOWN;
        DB_WRITE_BARRIER();
        pthread_join(*thread, NULL);
        args->started = 0;
        return -1;
    }
    return 0;
}

static int sqmd_session_close(SqmdSharedBuffer *sqmd_shared,
                              pthread_t thread,
                              RetrievalRpcSqmdSessionArgs *args) {
    int ret = 0;

    if (sqmd_shared && sqmd_shared->ctrl &&
        sqmd_shared->ctrl->session_state != SQMD_SESSION_ABORT) {
        DB_WRITE_BARRIER();
        sqmd_shared->ctrl->session_state = SQMD_SESSION_SHUTDOWN;
        DB_WRITE_BARRIER();
    }
    if (args->started) {
        pthread_join(thread, NULL);
        args->started = 0;
        if (args->ret != 0)
            ret = -1;
    }
    return ret;
}


static int sqmd_search_one_query_session_with_reader(const RetrievalDemoEncodedDataset *dataset,
                                         const int32_t *graph,
                                         int graph_degree,
                                         int graph_entry,
                                         const float *query,
                                         const RetrievalDemoConfig *config,
                                         ResidualBlockCache *res_cache,
                                         SqmdSharedBuffer *sqmd_shared,
                                         RetrievalRpcSqmdSessionArgs *session,
                                         int max_inflight_sources,
                                         int split_enabled,
                                         DemoResult *metrics,
                                         int *out_ids,
                                         float *out_dists,
                                         int out_cap, const RetrievalGraphReader *graph_reader) {
    sqmdCpuSearchState st;
    SqmdHostCorrectionCtx corr_ctx;
    SqmdSplitQueryCtx split_ctx;
    SqmdSharedControl *ctrl;
    int *seed_ids = NULL;
    float *seed_dists = NULL;
    int seed_count;
    int result_count = -1;
    int qd;

    memset(&st, 0, sizeof(st));
    memset(&split_ctx, 0, sizeof(split_ctx));

    if (!dataset || !graph || !query || !config || !sqmd_shared ||
        !sqmd_shared->ctrl || sqmd_shared->fd < 0 || !session ||
        !out_ids || !out_dists || config->entry_points <= 0 || out_cap <= 0 ||
        dataset->dim <= 0 || dataset->dim > RETRIEVAL_ASYNC_DEMO_MAX_DIM)
        return -1;
    if (config->pool_size > DSP_SEARCH_MAX_POOL) return -1;
    ctrl = sqmd_shared->ctrl;

    if (split_enabled &&
        sqmd_split_query_init(&split_ctx, query, dataset, res_cache,
                              g_sqmd_correction_bias) != 0)
        return -1;

    if (session->done) {
        printf("  ERROR: SQMD session RPC already exited ret=0x%x\n",
               session->ret);
        fflush(stdout);
        return -1;
    }
    DB_READ_BARRIER();
    if (ctrl->session_state != SQMD_SESSION_RUNNING ||
        ctrl->ctrl_state != SQMD_CTRL_DONE ||
        !sqmd_query_buffers_quiescent(ctrl)) {
        printf("  ERROR: SQMD session query started before quiescence"
               " session_state=%d ctrl_state=%d turn=%d\n",
               ctrl->session_state,
               ctrl->ctrl_state,
               ctrl->turn_id);
        fflush(stdout);
        return -1;
    }

    seed_ids = (int *)malloc((size_t)config->entry_points * sizeof(*seed_ids));
    seed_dists = (float *)malloc((size_t)config->entry_points *
                                 sizeof(*seed_dists));
    if (!seed_ids || !seed_dists)
        goto cleanup;

    seed_count = sqmd_make_seed(dataset, query,
                               split_enabled ? &split_ctx : NULL,
                               config->entry_points,
                               graph_entry, ENTRY_POINT_SEED,
                               seed_ids, seed_dists, config->entry_points);
    if (seed_count <= 0)
        goto cleanup;

    if (sqmd_cpu_state_init(&st, dataset->num_vectors, config->pool_size,
                            config->entry_points, config->expand_count,
                            query, dataset, res_cache) != SQMD_STATUS_OK)
        goto cleanup;
    if (graph_reader) st.graph_reader = *graph_reader;
    if (sqmd_cpu_state_seed(&st, seed_ids, seed_dists, seed_count) !=
        SQMD_STATUS_OK)
        goto cleanup;

    if (session->query_gen == INT_MAX) {
        printf("  ERROR: SQMD session query generation exhausted\n");
        fflush(stdout);
        DB_WRITE_BARRIER();
        ctrl->session_state = SQMD_SESSION_ABORT;
        DB_WRITE_BARRIER();
        goto cleanup;
    }


    qd = dataset->dim;
    memcpy((void *)ctrl->query_slot.query, query, (size_t)qd * sizeof(float));
    ctrl->query_slot.query_dim = qd;
    ctrl->query_slot.contract_id = split_enabled
        ? SQMD_QUERY_CONTRACT_L1_ALIGNED_U8 : SQMD_QUERY_CONTRACT_LEGACY;
    if (split_enabled) {
        memcpy((void *)ctrl->query_slot.query_u8, split_ctx.query_u8,
               (size_t)qd);
    }
    DB_WRITE_BARRIER();
    session->query_gen++;
    ctrl->query_slot.query_gen = session->query_gen;
    DB_WRITE_BARRIER();

    DB_WRITE_BARRIER();
    ctrl->ctrl_state = SQMD_CTRL_RUNNING;
    DB_WRITE_BARRIER();

    corr_ctx.query = query;
    corr_ctx.dataset = dataset;
    corr_ctx.res_cache = res_cache;
    if (split_enabled) {
        result_count = sqmd_cpu_state_run_source_pipeline_split(
            &st, ctrl, max_inflight_sources, graph, graph_degree,
            dataset->num_vectors, sqmd_split_residual_delta_fn, &split_ctx,
            sqmd_cpu_wait_yield, &st, sqmd_host_now_us, NULL,
            out_ids, out_dists, out_cap);
        if (result_count >= 0 && metrics) {
            metrics->split_queries++;
            metrics->split_candidates += st.exact_corrected;
            metrics->query_u8_clipped_low +=
                split_ctx.query_stats.clipped_low;
            metrics->query_u8_clipped_high +=
                split_ctx.query_stats.clipped_high;
            metrics->query_u8_error_sq +=
                split_ctx.query_stats.reconstruction_error_sq;
            metrics->query_u8_norm_sq += split_ctx.query_stats.query_norm_sq;
        }
    } else {
        result_count = sqmd_cpu_state_run_source_pipeline(
            &st, ctrl, max_inflight_sources, graph, graph_degree,
            dataset->num_vectors, sqmd_host_correction_fn, &corr_ctx,
            sqmd_cpu_wait_yield, &st, sqmd_host_now_us, NULL,
            out_ids, out_dists, out_cap);
    }
    if (result_count < 0) {
        const SqmdSharedStats *stats = &ctrl->stats;


        DB_WRITE_BARRIER();
        ctrl->session_state = SQMD_SESSION_ABORT;
        DB_WRITE_BARRIER();
        printf("  ERROR: SQMD session loop failed"
               " ctrl_state=%d session_state=%d turn=%d stats_abort=%llu\n",
               ctrl->ctrl_state, ctrl->session_state, ctrl->turn_id,
               (unsigned long long)stats->sqmd_abort_count);
        sqmd_print_abort_detail("CPU",
                                st.abort_stage, st.abort_rc, st.abort_turn,
                                st.abort_lane, st.abort_buffer, st.abort_item,
                                st.abort_state, st.abort_buf_turn,
                                st.abort_work_count, st.abort_result_count,
                                st.abort_neighbor_id, st.abort_source_id);
        sqmd_print_abort_buffers(ctrl, st.abort_turn);
        fflush(stdout);
    }

cleanup:
    if (result_count >= 0 && (st.pool_capacity != config->pool_size ||
        st.pool_size < config->top_k)) result_count=-1;
    sqmd_cpu_state_cleanup(&st);
    free(seed_ids);
    free(seed_dists);
    return result_count;
}

static int sqmd_search_one_query_session(const RetrievalDemoEncodedDataset *dataset,
                                         const int32_t *graph,
                                         int graph_degree,
                                         int graph_entry,
                                         const float *query,
                                         const RetrievalDemoConfig *config,
                                         ResidualBlockCache *res_cache,
                                         SqmdSharedBuffer *sqmd_shared,
                                         RetrievalRpcSqmdSessionArgs *session,
                                         int max_inflight_sources,
                                         int split_enabled,
                                         DemoResult *metrics,
                                         int *out_ids,
                                         float *out_dists,
                                         int out_cap) {
    return sqmd_search_one_query_session_with_reader(dataset,graph,graph_degree,graph_entry,
        query,config,res_cache,sqmd_shared,session,max_inflight_sources,split_enabled,
        metrics,out_ids,out_dists,out_cap,NULL);
}


static size_t demo_l1_chunk_bytes(int c) {
    int vpc   = g_dataset.l1_chunk_vectors;
    int vecs  = (c == g_dataset.l1_chunk_count - 1)
                    ? (g_dataset.num_vectors - c * vpc) : vpc;
    return (size_t)vecs * (size_t)g_dataset.l1_code_bytes;
}


static int demo_map_l1_chunks(int domain_id, int *fds_out) {
    int c;
    g_dma_l1_mapped_count = 0;
    for (c = 0; c < g_dataset.l1_chunk_count; c++) {
        size_t bytes = demo_l1_chunk_bytes(c);
        int fd = demo_buf_to_fd(g_dataset.l1_chunks[c]);
        int mret;
        if (fd < 0) {
            printf("  WARN: L1 chunk %d has no fd\n", c);
            goto fail;
        }
        mret = fastrpc_mmap(domain_id, fd, g_dataset.l1_chunks[c],
                            0, bytes, FASTRPC_MAP_FD);
        if (mret != 0) {
            printf("  WARN: fastrpc_mmap(l1 chunk %d/%d, %llu bytes) failed 0x%x\n",
                   c, g_dataset.l1_chunk_count, (unsigned long long)bytes, mret);
            fflush(stdout);
            goto fail;
        }
        fds_out[c]         = fd;
        g_dma_l1_fds[c]    = fd;
        g_dma_l1_mapped_count = c + 1;
    }
    g_dma_l1_mapped = 1;
    return g_dataset.l1_chunk_count;
fail:
    {
        int k;
        for (k = 0; k < g_dma_l1_mapped_count; k++) {
            if (fastrpc_munmap) {
                fastrpc_munmap(domain_id, g_dma_l1_fds[k],
                               g_dataset.l1_chunks[k], demo_l1_chunk_bytes(k));
            }
            g_dma_l1_fds[k] = -1;
        }
        g_dma_l1_mapped_count = 0;
        g_dma_l1_mapped = 0;
    }
    return -1;
}

static void demo_dma_unmap(void) {
    size_t graph_bytes = (size_t)DATA_N_BASE * (size_t)DATA_GRAPH_K * sizeof(int32_t);
    int c;

    if (g_dma_graph_mapped && g_dma_graph_fd >= 0 && fastrpc_munmap) {
        fastrpc_munmap(g_dma_domain_id, g_dma_graph_fd, g_graph, graph_bytes);
        g_dma_graph_mapped = 0;
    }
    if (g_dma_l1_mapped && fastrpc_munmap) {
        for (c = 0; c < g_dma_l1_mapped_count; c++) {
            fastrpc_munmap(g_dma_domain_id, g_dma_l1_fds[c],
                           g_dataset.l1_chunks[c], demo_l1_chunk_bytes(c));
            g_dma_l1_fds[c] = -1;
        }
        g_dma_l1_mapped = 0;
        g_dma_l1_mapped_count = 0;
    }
    g_dma_graph_fd    = -1;
    g_dma_domain_id   = -1;
}

int run_retrieval_search_rpc_demo(remote_handle64 handle, int domain_id) {
    const int entry_vals[] = RETRIEVAL_DEMO_ENTRY_SWEEP_VALUES;
    const int pool_vals[]  = RETRIEVAL_DEMO_POOL_SWEEP_VALUES;
    const int n_entry   = (int)(sizeof(entry_vals) / sizeof(entry_vals[0]));
    const int n_pool    = (int)(sizeof(pool_vals)  / sizeof(pool_vals[0]));
    const int n_configs = n_entry * n_pool;
    RetrievalDemoConfig configs[40];
    DemoResult results[40];
    int num_queries   = RETRIEVAL_RPC_DEMO_QUERY_LIMIT;
    int rerank_count = FINAL_REFINE_MODE ? FINAL_REFINE_CANDIDATES : SEARCH_TOP_K * SEARCH_RERANK_FACTOR;
    int batch_width   = DEMO_RPC_MAX_CONCURRENT;
    int config_idx    = 0;
    int i;
    int ret = 0;
    int use_dma = 0;
    unsigned long long t_load;

    if (num_queries > DATA_N_QUERIES) num_queries = DATA_N_QUERIES;
    if (num_queries <= 0)             num_queries = DATA_N_QUERIES;

    for (i = 0; i < n_entry; i++) {
        int p;
        for (p = 0; p < n_pool; p++) {
            configs[config_idx].entry_points  = entry_vals[i];
            configs[config_idx].expand_count  = rover_config.expand_count;
            configs[config_idx].pool_size     = pool_vals[p];
            configs[config_idx].top_k         = SEARCH_TOP_K;
            configs[config_idx].rerank_factor = SEARCH_RERANK_FACTOR;
            config_idx++;
        }
    }

    printf("\n=== Retrieval Demo: DSP beam search + CPU L2 rerank ===\n");
    printf("N=%d DIM=%d GRAPH_K=%d QUERIES=%d K=%d EXPAND=%d RERANK×%d\n",
           DATA_N_BASE, DATA_DIM, DATA_GRAPH_K,
           num_queries, SEARCH_K_MAX, SEARCH_EXPAND_COUNT, SEARCH_RERANK_FACTOR);
    printf("DATA_LAYOUT=%s DATA_DIR=%s\n", DATA_LAYOUT_NAME, DATA_DIR);
    printf("GRAPH_SCHEME=%d(%s) GRAPH_FILE=%s GRAPH_ENTRY=%d\n",
           RETRIEVAL_GRAPH_INDEX_SCHEME,
           DATA_GRAPH_INDEX_SCHEME_NAME, DATA_GRAPH_FILENAME,
           DATA_GRAPH_ENTRY_POINT);
    printf("%d entry × %d pool = %d configs\n\n", n_entry, n_pool, n_configs);


    printf("[1/4] Encoding base vectors...\n");
    {
        char path[4096];
        snprintf(path, sizeof(path), "%s", DATA_BASE_PATH);
        t_load = now_us();


#if RESIDUAL_REORDER_SCHEME == 0
        {
            const char *ufs_path = residual_resolve_ufs_path();
            const char *qp_path  = RESIDUAL_QPARAMS_CACHE_PATH;
#if RETRIEVAL_DEMO_ENCODE_SOURCE == 1
            printf("  loading pre-encoded L1+residual (encode skipped)\n");
            ret = retrieval_demo_load_l1_and_qparams_from_disk(
                L1_CODES_CACHE_PATH,
                qp_path,
                DATA_N_BASE, DATA_DIM, RETRIEVAL_DEMO_RESIDUAL_BITS,
                &g_dataset);
if (RETRIEVAL_DEMO_USE_DISK == 0) {
            if (ret == 0) {
                ret = retrieval_demo_load_residual_codes_from_block_file(
                    RESIDUAL_UFS_PATH, &g_dataset);
            }
}
            if (ret != 0) {
                printf("  ERROR: pre-encoded files missing/corrupt "
                       "(expected %s + %s + %s)\n",
                       L1_CODES_CACHE_PATH, RESIDUAL_UFS_PATH, qp_path);
                goto cleanup;
            }
#else
if (RETRIEVAL_DEMO_USE_DISK == 0) {
            ret = retrieval_demo_encode_dataset_from_fvecs_streaming(
                path, DATA_N_BASE, DATA_DIM, &g_dataset);
            if (ret != 0) { printf("  ERROR (in-memory encode)\n"); goto cleanup; }
} else {
            ret = retrieval_demo_encode_dataset_streaming_to_disk(
                path, DATA_N_BASE, DATA_DIM,
                ufs_path ? ufs_path : RESIDUAL_UFS_PATH,
                qp_path,
                NULL,
                &g_dataset);
            if (ret != 0) { printf("  ERROR (streaming encode)\n"); goto cleanup; }
}
#endif
        }
#else
        ret = retrieval_demo_encode_dataset_from_fvecs_streaming(
            path, DATA_N_BASE, DATA_DIM, &g_dataset);
        if (ret != 0) { printf("  ERROR\n"); goto cleanup; }
#endif
        printf("  L1 int%d scale=%.6f zero=%.4f bytes/vec=%d time=%llu ms\n",
               g_dataset.l1_bits,
               g_dataset.l1_qparams.scale, g_dataset.l1_qparams.zero_point,
               g_dataset.l1_code_bytes,
               (now_us() - t_load) / 1000ULL);

        if (demo_promote_l1_codes_to_shared(&g_dataset) != 0) {
            printf("  ERROR: cannot allocate shared l1_codes buffer\n");
            ret = -1;
            goto cleanup;
        }
        printf("  l1_codes storage: %s\n",
               g_dataset.l1_codes_uses_rpcmem ? "rpcmem" : "malloc fallback");
    }

    printf("[2/4] Loading %s graph (%s, n=%d, k=%d)...\n",
           DATA_GRAPH_INDEX_SCHEME_NAME, DATA_GRAPH_FILENAME,
           DATA_N_BASE, DATA_GRAPH_K);
    if (demo_resolve_graph_index(handle,
                                 domain_id,
                                 &g_graph,
                                 &g_graph_uses_rpcmem,
                                 &g_nsg_entry_point) != 0) {
        ret = -1;
        goto cleanup;
    }


#if RESIDUAL_REORDER_SCHEME > 0
    {
        unsigned long long t_reorder = now_us();
        int vpb = RESIDUAL_BLOCK_SIZE / g_dataset.residual_code_bytes;
        if (reorder_compute(g_graph, DATA_N_BASE, DATA_GRAPH_K, vpb,
                            RESIDUAL_REORDER_SCHEME,
                            &g_reorder_map, &g_reverse_map) != 0) {
            printf("  ERROR: reorder computation failed\n");
            ret = -1;
            goto cleanup;
        }
        {
            int total_edges = 0;
            int same_block = 0;
            int ei, ej;
            for (ei = 0; ei < DATA_N_BASE; ei++) {
                int block_i = g_reorder_map[ei] / vpb;
                const int32_t *nb = g_graph + (size_t)ei * DATA_GRAPH_K;
                for (ej = 0; ej < DATA_GRAPH_K; ej++) {
                    int v = nb[ej];
                    if (v < 0 || v >= DATA_N_BASE || v == ei) continue;
                    total_edges++;
                    if (g_reorder_map[v] / vpb == block_i)
                        same_block++;
                }
            }
            printf("  Reorder scheme=%d time=%llu ms\n",
                   RESIDUAL_REORDER_SCHEME,
                   (now_us() - t_reorder) / 1000ULL);
            printf("  Edge co-location: %d/%d (%.1f%%) edges in same block\n",
                   same_block, total_edges,
                   100.0 * (double)same_block / (double)total_edges);
        }
        fflush(stdout);
    }
#endif


    {
        const char *ufs_path = residual_resolve_ufs_path();
#if RESIDUAL_REORDER_SCHEME == 0
        if (ufs_path) {
            int rfd = open(ufs_path, O_RDONLY);
            if (rfd >= 0) {
                g_residual_fd = rfd;
                g_residual_vpb = RESIDUAL_BLOCK_SIZE / g_dataset.residual_code_bytes;
                g_residual_total_blocks = (g_dataset.num_vectors + g_residual_vpb - 1)
                                          / g_residual_vpb;
                printf("  Residual codes -> UFS (%s), %d blocks, vpb=%d\n",
                       ufs_path, g_residual_total_blocks, g_residual_vpb);
            } else {
                g_residual_fd = -1;
                printf("  WARN: open(%s) failed, using passthrough\n", ufs_path);
            }
        } else {
            g_residual_fd = -1;
        }
#else
        if (ufs_path &&
            residual_write_to_file(&g_dataset, ufs_path, g_reverse_map) == 0) {
            int rfd = open(ufs_path, O_RDONLY);
            if (rfd >= 0) {
                g_residual_fd = rfd;
                g_residual_vpb = RESIDUAL_BLOCK_SIZE / g_dataset.residual_code_bytes;
                g_residual_total_blocks = (g_dataset.num_vectors + g_residual_vpb - 1)
                                          / g_residual_vpb;
                free(g_dataset.residual_codes);
                g_dataset.residual_codes = NULL;
                printf("  Residual codes -> UFS (%s), %d blocks, vpb=%d\n",
                       ufs_path, g_residual_total_blocks, g_residual_vpb);
            } else {
                g_residual_fd = -1;
                printf("  WARN: open(%s) failed, using passthrough\n", ufs_path);
                unlink(ufs_path);
            }
        } else {
            g_residual_fd = -1;
if (RETRIEVAL_DEMO_USE_DISK != 0) {
            if (!ufs_path)
                printf("  WARN: residual UFS path disabled, using memory fallback\n");
            if (ufs_path) printf("  WARN: residual write failed, using passthrough\n");
}
        }
#endif
        if (g_residual_fd < 0) {
            if (!g_dataset.residual_codes) {
                printf("  ERROR: residual codes have no UFS or memory backing\n");
                ret = -1;
                goto cleanup;
            }
            printf("  Residual codes -> memory (%llu bytes)\n",
                   (unsigned long long)g_dataset.num_vectors *
                   (unsigned long long)g_dataset.residual_code_bytes);
            g_residual_vpb = RESIDUAL_BLOCK_SIZE / g_dataset.residual_code_bytes;
            g_residual_total_blocks = (g_dataset.num_vectors + g_residual_vpb - 1)
                                      / g_residual_vpb;
        }
        memset(&g_residual_stats, 0, sizeof(g_residual_stats));
        pthread_mutex_init(&g_residual_stats.mu, NULL);
        fflush(stdout);
    }

    printf("[3/4] Loading queries and ground truth...\n");
    {
        char qp[4096], gtp[4096];
        snprintf(qp,  sizeof(qp),  "%s", DATA_QUERY_PATH);
        snprintf(gtp, sizeof(gtp), "%s", DATA_GT_PATH);
        ret  = load_fvecs_n(qp,  num_queries, DATA_DIM,   &g_queries);
        ret |= load_ivecs_n(gtp, num_queries, DEMO_GT_K,  &g_gt);
        if (ret != 0) { printf("  ERROR\n"); goto cleanup; }
        printf("  done. %d queries, GT@%d\n", num_queries, DEMO_GT_K);
    }

    printf("[4/4a] Loading index to DSP (graph=%d MB, l1_codes=%d MB)...\n",
           (int)((size_t)DATA_N_BASE * DATA_GRAPH_K * 4 / 1024 / 1024),
           (int)((size_t)DATA_N_BASE * (size_t)g_dataset.l1_code_bytes /
                 1024 / 1024));
    {
        t_load = now_us();


        if (g_graph_uses_rpcmem && g_dataset.l1_codes_uses_rpcmem && fastrpc_mmap) {
            int fd_graph    = rpcmem_to_fd(g_graph);
            int l1_fds[MAX_L1_CHUNKS];

            if (fd_graph >= 0) {
                size_t graph_bytes = (size_t)DATA_N_BASE * (size_t)DATA_GRAPH_K * sizeof(int32_t);
                int mret;

                mret = fastrpc_mmap(domain_id, fd_graph, g_graph,
                                    0, graph_bytes, FASTRPC_MAP_FD);
                if (mret == 0) {
                    g_dma_graph_fd     = fd_graph;
                    g_dma_graph_mapped = 1;
                    g_dma_domain_id    = domain_id;
                } else {
                    printf("  WARN: fastrpc_mmap(graph) failed 0x%x, fallback to copy\n", mret);
                }

                if (g_dma_graph_mapped &&
                    demo_map_l1_chunks(domain_id, l1_fds) < 0) {
                    fastrpc_munmap(domain_id, fd_graph, g_graph, graph_bytes);
                    g_dma_graph_mapped = 0;
                    g_dma_graph_fd     = -1;
                    g_dma_domain_id    = -1;
                }

                if (g_dma_graph_mapped && g_dma_l1_mapped) {
                    ret = calculator_retrieval_load_dma(
                        handle,
                        fd_graph, l1_fds, g_dataset.l1_chunk_count,
                        g_dataset.l1_chunk_vectors,
                        DATA_N_BASE, DATA_DIM, DATA_GRAPH_K,
                        g_nsg_entry_point,
                        g_dataset.l1_qparams.scale,
                        g_dataset.l1_qparams.zero_point);
                    if (ret == 0) {
                        use_dma = 1;
                    } else {
                        printf("  WARN: retrieval_load_dma failed 0x%x, fallback to copy\n", ret);
                        demo_dma_unmap();
                        ret = 0;
                    }
                }
            }
        }

        if (!use_dma) {
            size_t l1_total = (size_t)DATA_N_BASE * (size_t)g_dataset.l1_code_bytes;
            if (l1_total > (size_t)INT_MAX) {
                printf("  ERROR: DMA load unavailable and L1 codes (%llu bytes) "
                       "exceed the by-value RPC limit; refusing (would OOM).\n",
                       (unsigned long long)l1_total);
                ret = -1;
                goto cleanup;
            }
            ret = calculator_retrieval_load(
                handle,
                g_graph,            DATA_N_BASE * DATA_GRAPH_K,
                g_dataset.l1_codes, DATA_N_BASE * g_dataset.l1_code_bytes,
                DATA_N_BASE, DATA_DIM, DATA_GRAPH_K,
                g_nsg_entry_point,
                g_dataset.l1_qparams.scale,
                g_dataset.l1_qparams.zero_point);
        }

        if (ret != 0) {
            printf("  ERROR: retrieval_load%s failed 0x%x\n",
                   use_dma ? "_dma" : "", ret);
            goto cleanup;
        }
        printf("  done. time=%llu ms (%s)\n",
               (now_us() - t_load) / 1000ULL,
               use_dma ? "DMA zero-copy" : "copy");
    }

    printf("[4/4b] Running %d configs × %d queries (2D concurrency=%d)...\n\n",
           n_configs,
           num_queries,
           batch_width);

    for (config_idx = 0; config_idx < n_configs; config_idx++) {
        const RetrievalDemoConfig *config = &configs[config_idx];
        DemoResult *res = &results[config_idx];
        RetrievalPipeline pipe;
        pthread_t rerank_thread;
        pthread_t rpc_threads[DEMO_RPC_MAX_CONCURRENT];
        RetrievalRpcWorkerArgs rpc_args[DEMO_RPC_MAX_CONCURRENT];
        int pipe_started = 0;
        int num_out = FINAL_REFINE_MODE ? FINAL_REFINE_CANDIDATES : config->top_k * config->rerank_factor;
        int query_idx;
        unsigned long long t_cfg;

        memset(res, 0, sizeof(*res));
        if (pipeline_init(&pipe, batch_width, rerank_count, results, configs) != 0) {
            printf("  ERROR: pipeline init failed\n");
            ret = -1;
            goto cleanup;
        }
        {
            int s;
            for (s = 0; s < pipe.slot_count; s++) {
                if (residual_init_slot_cache(&pipe.slots[s], &g_dataset) != 0) {
                    printf("  ERROR: slot cache init failed\n");
                    {
                        int t;
                        for (t = 0; t < s; t++) {
                            residual_cache_destroy(pipe.slots[t].res_cache);
                            pipe.slots[t].res_cache = NULL;
                        }
                    }
                    pipeline_destroy(&pipe); ret = -1; goto cleanup;
                }
            }
        }
        if (pthread_create(&rerank_thread, NULL, rerank_thread_fn, &pipe) != 0) {
            printf("  ERROR: rerank thread create failed\n");
            {
                int s;
                for (s = 0; s < pipe.slot_count; s++) {
                    residual_cache_destroy(pipe.slots[s].res_cache);
                    pipe.slots[s].res_cache = NULL;
                }
            }
            pipeline_destroy(&pipe);
            ret = -1;
            goto cleanup;
        }
        pipe_started = 1;
        t_cfg = now_us();

        for (query_idx = 0; query_idx < num_queries; query_idx += batch_width) {
            int batch_count = num_queries - query_idx;
            int b;

            if (batch_count > batch_width) {
                batch_count = batch_width;
            }

            memset(rpc_args, 0, sizeof(rpc_args));
            for (b = 0; b < batch_count; b++) {
                int slot_idx;
                RetrievalSlot *slot;
                int qidx = query_idx + b;

                slot_idx = pipeline_acquire_slot(&pipe);
                if (slot_idx < 0) {
                    printf("  ERROR: pipeline acquire failed\n");
                    ret = -1;
                    break;
                }

                slot = &pipe.slots[slot_idx];
                slot->query = g_queries + (size_t)qidx * DATA_DIM;
                slot->gt_row = g_gt + (size_t)qidx * DEMO_GT_K;
                slot->query_idx = qidx;
                slot->config_idx = config_idx;

                rpc_args[b].pipe = &pipe;
                rpc_args[b].handle = handle;
                rpc_args[b].config = config;
                rpc_args[b].num_out = num_out;
                rpc_args[b].slot_idx = slot_idx;
                rpc_args[b].started = 0;
                if (pthread_create(&rpc_threads[b], NULL, rpc_search_worker_fn, &rpc_args[b]) != 0) {
                    printf("  ERROR: rpc worker create failed\n");
                    pipeline_abort_slot(&pipe, slot_idx);
                    ret = -1;
                    break;
                }
                rpc_args[b].started = 1;
            }

            for (b = 0; b < batch_count; b++) {
                if (rpc_args[b].started) {
                    pthread_join(rpc_threads[b], NULL);
                    if (ret == 0 && rpc_args[b].ret != 0) {
                        ret = rpc_args[b].ret;
                    }
                }
            }
            if (ret != 0) {
                printf("    FAILED near q%d: 0x%x\n", query_idx, ret);
                break;
            }
        }

        if (ret == 0) {
            ret = pipeline_drain(&pipe);
            if (ret != 0) {
                printf("  ERROR: pipeline drain failed\n");
            }
        }
        res->total_us_sum = now_us() - t_cfg;
        pipeline_shutdown(&pipe);
        if (pipe_started) {
            pthread_join(rerank_thread, NULL);
        }
        if (ret == 0 && pipe.ret != 0) {
            ret = pipe.ret;
        }
        {
            int s;
            for (s = 0; s < pipe.slot_count; s++) {
                residual_cache_destroy(pipe.slots[s].res_cache);
                pipe.slots[s].res_cache = NULL;
            }
        }
        pipeline_destroy(&pipe);
        if (ret != 0) {
            goto cleanup;
        }

        {
            printf("  [%2d/%d] e=%3d p=%4d  R@1=%5.2f R@10=%5.2f R@100=%5.2f"
                   "  wall=%4llu us (dsp_wait=%4llu rr=%4llu conc=%d)\n",
                   config_idx + 1, n_configs,
                   config->entry_points, config->pool_size,
                   100.0*(double)res->recall1_hits   / (double)num_queries,
                   100.0*(double)res->recall10_hits  / ((double)num_queries*10.0),
                   100.0*(double)res->recall100_hits / ((double)num_queries*100.0),
                   res->total_us_sum  / (unsigned long long)num_queries,
                   res->phase1_us_sum / (unsigned long long)num_queries,
                   res->phase2_us_sum / (unsigned long long)num_queries,
                   batch_width);
            fflush(stdout);
        }
    }

    printf("\n  %-5s %-6s %9s %9s %10s %10s %9s %9s\n",
           "Entry","Pool","R@1(%)","R@10(%)","R@100(%)",
           "wall_us","dsp_us","rerank_us");
    printf("  %-5s %-6s %9s %9s %10s %10s %9s %9s\n",
           "-----","------","-------","--------","---------",
           "-------","------","--------");
    for (config_idx = 0; config_idx < n_configs; config_idx++) {
        DemoResult *res = &results[config_idx];
        printf("  %-5d %-6d %8.2f%% %8.2f%% %9.2f%% %10llu %9llu %9llu\n",
               configs[config_idx].entry_points,
               configs[config_idx].pool_size,
               100.0*(double)res->recall1_hits   / (double)num_queries,
               100.0*(double)res->recall10_hits  / ((double)num_queries*10.0),
               100.0*(double)res->recall100_hits / ((double)num_queries*100.0),
               res->total_us_sum  / (unsigned long long)num_queries,
               res->phase1_us_sum / (unsigned long long)num_queries,
               res->phase2_us_sum / (unsigned long long)num_queries);
    }

cleanup:
    if (g_residual_fd >= 0) {
        close(g_residual_fd);
        g_residual_fd = -1;
    }
    pthread_mutex_destroy(&g_residual_stats.mu);
    {
        int r;
        printf("\nResidual cache stats:\n");
        for (r = 0; r < RESIDUAL_CACHE_ROLE__COUNT; r++) {
            uint64_t h = g_residual_stats.hits[r];
            uint64_t m = g_residual_stats.misses[r];
            uint64_t e = g_residual_stats.errors[r];
            if (h + m + e > 0) {
                printf("  role=%d: hits=%llu misses=%llu errors=%llu hit_rate=%.2f%%\n",
                       r, (unsigned long long)h, (unsigned long long)m,
                       (unsigned long long)e,
                       (h + m > 0) ? 100.0 * (double)h / (double)(h + m) : 0.0);
            }
        }
        if (g_residual_stats.io_error)
            printf("  ** I/O ERROR detected -- results may be invalid **\n");
        fflush(stdout);
    }

    if (use_dma) {
        calculator_retrieval_unload_dma(handle);
        demo_dma_unmap();
    } else {
        calculator_retrieval_unload(handle);
    }
    demo_free_loaded_data();
    return ret;
}

int run_retrieval_search_rpc_demo_db(remote_handle64 handle, int domain_id) {
    const int entry_vals[] = RETRIEVAL_DEMO_ENTRY_SWEEP_VALUES;
    const int pool_vals[]  = RETRIEVAL_DEMO_POOL_SWEEP_VALUES;
    const int n_entry   = (int)(sizeof(entry_vals) / sizeof(entry_vals[0]));
    const int n_pool    = (int)(sizeof(pool_vals)  / sizeof(pool_vals[0]));
    const int n_configs = n_entry * n_pool;
    const int db_pair_batch_width = DEMO_DB_PAIR_MAX_CONCURRENT;
    RetrievalDemoConfig configs[40];
    DemoResult results[40];
    int num_queries   = RETRIEVAL_RPC_DEMO_QUERY_LIMIT;
    int rerank_count = FINAL_REFINE_MODE ? FINAL_REFINE_CANDIDATES : SEARCH_TOP_K * SEARCH_RERANK_FACTOR;
    int config_idx    = 0;
    int i;
    int ret = 0;
    int use_dma = 0;
    unsigned long long t_load;

    if (num_queries > DATA_N_QUERIES) num_queries = DATA_N_QUERIES;
    if (num_queries <= 0)             num_queries = DATA_N_QUERIES;


    for (i = 0; i < n_entry; i++) {
        int p;
        for (p = 0; p < n_pool; p++) {
            configs[config_idx].entry_points  = entry_vals[i];
            configs[config_idx].expand_count  = rover_config.expand_count;
            configs[config_idx].pool_size     = pool_vals[p];
            configs[config_idx].top_k         = SEARCH_TOP_K;
            configs[config_idx].rerank_factor = SEARCH_RERANK_FACTOR;
            config_idx++;
        }
    }

    printf("\n=== Retrieval Demo: Double-Buffer beam search (P3-E) ===\n");
    printf("N=%d DIM=%d GRAPH_K=%d QUERIES=%d K=%d EXPAND=%d RERANK*%d\n",
           DATA_N_BASE, DATA_DIM, DATA_GRAPH_K,
           num_queries, SEARCH_K_MAX, SEARCH_EXPAND_COUNT, SEARCH_RERANK_FACTOR);
    printf("DATA_LAYOUT=%s DATA_DIR=%s\n", DATA_LAYOUT_NAME, DATA_DIR);
    printf("GRAPH_SCHEME=%d(%s) GRAPH_FILE=%s GRAPH_ENTRY=%d\n",
           RETRIEVAL_GRAPH_INDEX_SCHEME,
           DATA_GRAPH_INDEX_SCHEME_NAME, DATA_GRAPH_FILENAME,
           DATA_GRAPH_ENTRY_POINT);
    printf("%d entry x %d pool = %d configs\n\n", n_entry, n_pool, n_configs);
    fflush(stdout);


    printf("[1/4] Encoding base vectors...\n");
    fflush(stdout);
    {
        char path[4096];
        snprintf(path, sizeof(path), "%s", DATA_BASE_PATH);
        t_load = now_us();


#if RESIDUAL_REORDER_SCHEME == 0
        {
            const char *ufs_path = residual_resolve_ufs_path();
            const char *qp_path  = RESIDUAL_QPARAMS_CACHE_PATH;
#if RETRIEVAL_DEMO_ENCODE_SOURCE == 1
            printf("  loading pre-encoded L1+residual (encode skipped)\n");
            ret = retrieval_demo_load_l1_and_qparams_from_disk(
                L1_CODES_CACHE_PATH,
                qp_path,
                DATA_N_BASE, DATA_DIM, RETRIEVAL_DEMO_RESIDUAL_BITS,
                &g_dataset);
if (RETRIEVAL_DEMO_USE_DISK == 0) {
            if (ret == 0) {
                ret = retrieval_demo_load_residual_codes_from_block_file(
                    RESIDUAL_UFS_PATH, &g_dataset);
            }
}
            if (ret != 0) {
                printf("  ERROR: pre-encoded files missing/corrupt "
                       "(expected %s + %s + %s)\n",
                       L1_CODES_CACHE_PATH, RESIDUAL_UFS_PATH, qp_path);
                fflush(stdout);
                goto cleanup;
            }
#else
if (RETRIEVAL_DEMO_USE_DISK == 0) {
            ret = retrieval_demo_encode_dataset_from_fvecs_streaming(
                path, DATA_N_BASE, DATA_DIM, &g_dataset);
            if (ret != 0) { printf("  ERROR (in-memory encode)\n"); fflush(stdout); goto cleanup; }
} else {
            ret = retrieval_demo_encode_dataset_streaming_to_disk(
                path, DATA_N_BASE, DATA_DIM,
                ufs_path ? ufs_path : RESIDUAL_UFS_PATH,
                qp_path,
                NULL,
                &g_dataset);
            if (ret != 0) { printf("  ERROR (streaming encode)\n"); fflush(stdout); goto cleanup; }
}
#endif
        }
#else
        ret = retrieval_demo_encode_dataset_from_fvecs_streaming(
            path, DATA_N_BASE, DATA_DIM, &g_dataset);
        if (ret != 0) { printf("  ERROR\n"); fflush(stdout); goto cleanup; }
#endif
        printf("  L1 int%d scale=%.6f zero=%.4f bytes/vec=%d time=%llu ms\n",
               g_dataset.l1_bits,
               g_dataset.l1_qparams.scale, g_dataset.l1_qparams.zero_point,
               g_dataset.l1_code_bytes,
               (now_us() - t_load) / 1000ULL);
        fflush(stdout);
        if (demo_promote_l1_codes_to_shared(&g_dataset) != 0) {
            printf("  ERROR: cannot allocate shared l1_codes buffer\n");
            fflush(stdout);
            ret = -1; goto cleanup;
        }
    }

    printf("[2/4] Loading %s graph (%s, n=%d, k=%d)...\n",
           DATA_GRAPH_INDEX_SCHEME_NAME, DATA_GRAPH_FILENAME,
           DATA_N_BASE, DATA_GRAPH_K);
    fflush(stdout);
    if (demo_resolve_graph_index(handle,
                                 domain_id,
                                 &g_graph,
                                 &g_graph_uses_rpcmem,
                                 &g_nsg_entry_point) != 0) {
        ret = -1; goto cleanup;
    }


#if RESIDUAL_REORDER_SCHEME > 0
    {
        unsigned long long t_reorder = now_us();
        int vpb = RESIDUAL_BLOCK_SIZE / g_dataset.residual_code_bytes;
        if (reorder_compute(g_graph, DATA_N_BASE, DATA_GRAPH_K, vpb,
                            RESIDUAL_REORDER_SCHEME,
                            &g_reorder_map, &g_reverse_map) != 0) {
            printf("  ERROR: reorder computation failed\n");
            fflush(stdout);
            ret = -1;
            goto cleanup;
        }
        {
            int total_edges = 0;
            int same_block = 0;
            int ei, ej;
            for (ei = 0; ei < DATA_N_BASE; ei++) {
                int block_i = g_reorder_map[ei] / vpb;
                const int32_t *nb = g_graph + (size_t)ei * DATA_GRAPH_K;
                for (ej = 0; ej < DATA_GRAPH_K; ej++) {
                    int v = nb[ej];
                    if (v < 0 || v >= DATA_N_BASE || v == ei) continue;
                    total_edges++;
                    if (g_reorder_map[v] / vpb == block_i)
                        same_block++;
                }
            }
            printf("  Reorder scheme=%d time=%llu ms\n",
                   RESIDUAL_REORDER_SCHEME,
                   (now_us() - t_reorder) / 1000ULL);
            printf("  Edge co-location: %d/%d (%.1f%%) edges in same block\n",
                   same_block, total_edges,
                   100.0 * (double)same_block / (double)total_edges);
        }
        fflush(stdout);
    }
#endif


    {
        const char *ufs_path = residual_resolve_ufs_path();
#if RESIDUAL_REORDER_SCHEME == 0
        if (ufs_path) {
            int rfd = open(ufs_path, O_RDONLY);
            if (rfd >= 0) {
                g_residual_fd = rfd;
                g_residual_vpb = RESIDUAL_BLOCK_SIZE / g_dataset.residual_code_bytes;
                g_residual_total_blocks = (g_dataset.num_vectors + g_residual_vpb - 1)
                                          / g_residual_vpb;
                printf("  Residual codes -> UFS (%s), %d blocks, vpb=%d\n",
                       ufs_path, g_residual_total_blocks, g_residual_vpb);
            } else {
                g_residual_fd = -1;
                printf("  WARN: open(%s) failed, using passthrough\n", ufs_path);
            }
        } else {
            g_residual_fd = -1;
        }
#else
        if (ufs_path &&
            residual_write_to_file(&g_dataset, ufs_path, g_reverse_map) == 0) {
            int rfd = open(ufs_path, O_RDONLY);
            if (rfd >= 0) {
                g_residual_fd = rfd;
                g_residual_vpb = RESIDUAL_BLOCK_SIZE / g_dataset.residual_code_bytes;
                g_residual_total_blocks = (g_dataset.num_vectors + g_residual_vpb - 1)
                                          / g_residual_vpb;
                free(g_dataset.residual_codes);
                g_dataset.residual_codes = NULL;
                printf("  Residual codes -> UFS (%s), %d blocks, vpb=%d\n",
                       ufs_path, g_residual_total_blocks, g_residual_vpb);
            } else {
                g_residual_fd = -1;
                printf("  WARN: open(%s) failed, using passthrough\n", ufs_path);
                unlink(ufs_path);
            }
        } else {
            g_residual_fd = -1;
if (RETRIEVAL_DEMO_USE_DISK != 0) {
            if (!ufs_path)
                printf("  WARN: residual UFS path disabled, using memory fallback\n");
            if (ufs_path) printf("  WARN: residual write failed, using passthrough\n");
}
        }
#endif
        if (g_residual_fd < 0) {
            if (!g_dataset.residual_codes) {
                printf("  ERROR: residual codes have no UFS or memory backing\n");
                ret = -1;
                goto cleanup;
            }
            printf("  Residual codes -> memory (%llu bytes)\n",
                   (unsigned long long)g_dataset.num_vectors *
                   (unsigned long long)g_dataset.residual_code_bytes);
            g_residual_vpb = RESIDUAL_BLOCK_SIZE / g_dataset.residual_code_bytes;
            g_residual_total_blocks = (g_dataset.num_vectors + g_residual_vpb - 1)
                                      / g_residual_vpb;
        }
        memset(&g_residual_stats, 0, sizeof(g_residual_stats));
        pthread_mutex_init(&g_residual_stats.mu, NULL);
        fflush(stdout);
    }

    printf("[3/4] Loading queries and ground truth...\n");
    fflush(stdout);
    {
        char qp[4096], gtp[4096];
        snprintf(qp,  sizeof(qp),  "%s", DATA_QUERY_PATH);
        snprintf(gtp, sizeof(gtp), "%s", DATA_GT_PATH);
        ret  = load_fvecs_n(qp,  num_queries, DATA_DIM,   &g_queries);
        ret |= load_ivecs_n(gtp, num_queries, DEMO_GT_K,  &g_gt);
        if (ret != 0) { printf("  ERROR\n"); fflush(stdout); goto cleanup; }
        printf("  done. %d queries\n", num_queries);
        fflush(stdout);
    }

    printf("[4/4a] Loading index to DSP (DMA)...\n");
    fflush(stdout);
    {
        t_load = now_us();

        if (g_graph_uses_rpcmem && g_dataset.l1_codes_uses_rpcmem && fastrpc_mmap) {
            int fd_graph    = rpcmem_to_fd(g_graph);
            int l1_fds[MAX_L1_CHUNKS];

            if (fd_graph >= 0) {
                size_t graph_bytes = (size_t)DATA_N_BASE * (size_t)DATA_GRAPH_K * sizeof(int32_t);
                int mret;

                mret = fastrpc_mmap(domain_id, fd_graph, g_graph, 0, graph_bytes, FASTRPC_MAP_FD);
                if (mret == 0) {
                    g_dma_graph_fd = fd_graph; g_dma_graph_mapped = 1; g_dma_domain_id = domain_id;
                }

                if (g_dma_graph_mapped &&
                    demo_map_l1_chunks(domain_id, l1_fds) < 0) {
                    fastrpc_munmap(domain_id, fd_graph, g_graph, graph_bytes);
                    g_dma_graph_mapped = 0; g_dma_graph_fd = -1; g_dma_domain_id = -1;
                }

                if (g_dma_graph_mapped && g_dma_l1_mapped) {
                    if (!use_dma) {
                        ret = calculator_retrieval_load_dma(
                            handle, fd_graph,
                            l1_fds, g_dataset.l1_chunk_count,
                            g_dataset.l1_chunk_vectors,
                            DATA_N_BASE, DATA_DIM, DATA_GRAPH_K,
                            g_nsg_entry_point,
                            g_dataset.l1_qparams.scale, g_dataset.l1_qparams.zero_point);
                    }
                    if (ret == 0) {
                        use_dma = 1;
                    } else {
                        demo_dma_unmap(); ret = 0;
                    }
                }
            }
        }

        if (!use_dma) {
            printf("  ERROR: double-buffer requires DMA mode\n");
            fflush(stdout);
            ret = -1; goto cleanup;
        }

        printf("  done. time=%llu ms (DMA)\n", (now_us() - t_load) / 1000ULL);
        fflush(stdout);
    }

    printf("[4/4b] Setting up %d double-buffer expansion buffers...\n",
           db_pair_batch_width);
    fflush(stdout);
    {
        size_t buf_bytes = sizeof(DbSharedLayout);
        for (i = 0; i < db_pair_batch_width; i++) {
            if (demo_db_buffer_init(&g_db_shared[i], domain_id) != 0) {
                printf("  ERROR: expansion buffer[%d] init failed\n", i);
                fflush(stdout);
                ret = -1;
                goto cleanup;
            }
        }
        printf("  done. %d workers x %zu bytes\n", db_pair_batch_width, buf_bytes);
        fflush(stdout);
    }

    printf("[4/4c] Running %d configs x %d queries (DB pair-batch=%d / query-conc=%d)...\n\n",
           n_configs, num_queries, db_pair_batch_width, db_pair_batch_width * 2);
    fflush(stdout);

    for (config_idx = 0; config_idx < n_configs; config_idx++) {
        const RetrievalDemoConfig *config = &configs[config_idx];
        DemoResult *res = &results[config_idx];
        RetrievalPipeline pipe;
        pthread_t rerank_thread;
        pthread_t db_threads[DEMO_RPC_MAX_CONCURRENT];
        RetrievalRpcDbWorkerArgs db_args[DEMO_RPC_MAX_CONCURRENT];
        int pipe_started = 0;
        int num_out = FINAL_REFINE_MODE ? FINAL_REFINE_CANDIDATES : config->top_k * config->rerank_factor;
        int query_idx;
        unsigned long long t_cfg;

        memset(res, 0, sizeof(*res));
        if (pipeline_init(&pipe, db_pair_batch_width * 2,
                          rerank_count, results, configs) != 0) {
            ret = -1; goto cleanup;
        }
        {
            int s;
            for (s = 0; s < pipe.slot_count; s++) {
                if (residual_init_slot_cache(&pipe.slots[s], &g_dataset) != 0) {
                    printf("  ERROR: slot cache init failed\n");
                    {
                        int t;
                        for (t = 0; t < s; t++) {
                            residual_cache_destroy(pipe.slots[t].res_cache);
                            pipe.slots[t].res_cache = NULL;
                        }
                    }
                    pipeline_destroy(&pipe); ret = -1; goto cleanup;
                }
            }
        }
        if (pthread_create(&rerank_thread, NULL, rerank_thread_fn, &pipe) != 0) {
            {
                int s;
                for (s = 0; s < pipe.slot_count; s++) {
                    residual_cache_destroy(pipe.slots[s].res_cache);
                    pipe.slots[s].res_cache = NULL;
                }
            }
            pipeline_destroy(&pipe); ret = -1; goto cleanup;
        }
        pipe_started = 1;
        t_cfg = now_us();

        for (query_idx = 0; query_idx < num_queries; query_idx += db_pair_batch_width * 2) {
            int remaining_pairs = (num_queries - query_idx + 1) / 2;
            int batch_pairs = remaining_pairs;
            int b;

            if (batch_pairs > db_pair_batch_width) {
                batch_pairs = db_pair_batch_width;
            }

            memset(db_args, 0, sizeof(db_args));
            for (b = 0; b < batch_pairs; b++) {
                int qidx = query_idx + (b * 2);
                int s_a = pipeline_acquire_slot(&pipe);
                int s_b;

                if (s_a < 0) { ret = -1; break; }
                s_b = pipeline_acquire_slot(&pipe);
                if (s_b < 0) {
                    pipeline_abort_slot(&pipe, s_a);
                    ret = -1;
                    break;
                }

                pipe.slots[s_a].query      = g_queries + (size_t)qidx * DATA_DIM;
                pipe.slots[s_a].gt_row     = g_gt + (size_t)qidx * DEMO_GT_K;
                pipe.slots[s_a].query_idx  = qidx;
                pipe.slots[s_a].config_idx = config_idx;

                pipe.slots[s_b].query      = g_queries + (size_t)(qidx+1<num_queries?qidx+1:qidx) * DATA_DIM;
                pipe.slots[s_b].gt_row     = g_gt + (size_t)(qidx+1<num_queries?qidx+1:qidx) * DEMO_GT_K;
                pipe.slots[s_b].query_idx  = qidx + 1;
                pipe.slots[s_b].config_idx = config_idx;

                db_args[b].pipe       = &pipe;
                db_args[b].handle     = handle;
                db_args[b].config     = config;
                db_args[b].db_shared  = &g_db_shared[b];
                db_args[b].num_out    = num_out;
                db_args[b].slot_idx_a = s_a;
                db_args[b].slot_idx_b = s_b;

                if (pthread_create(&db_threads[b], NULL,
                                   rpc_search_db_worker_fn, &db_args[b]) != 0) {
                    pipeline_abort_slot(&pipe, s_a);
                    pipeline_abort_slot(&pipe, s_b);
                    ret = -1;
                    break;
                }
                db_args[b].started = 1;
            }

            for (b = 0; b < batch_pairs; b++) {
                if (db_args[b].started) {
                    pthread_join(db_threads[b], NULL);
                    if (ret == 0 && db_args[b].ret != 0) {
                        ret = db_args[b].ret;
                    }
                }
            }
            if (ret != 0) { break; }
        }

        if (ret == 0) ret = pipeline_drain(&pipe);
        res->total_us_sum = now_us() - t_cfg;
        pipeline_shutdown(&pipe);
        if (pipe_started) pthread_join(rerank_thread, NULL);
        if (ret == 0 && pipe.ret != 0) ret = pipe.ret;
        {
            int s;
            for (s = 0; s < pipe.slot_count; s++) {
                residual_cache_destroy(pipe.slots[s].res_cache);
                pipe.slots[s].res_cache = NULL;
            }
        }
        pipeline_destroy(&pipe);
        if (ret != 0) goto cleanup;

        {
            printf("  [%2d/%d] e=%3d p=%4d  R@1=%5.2f R@10=%5.2f R@100=%5.2f"
                   "  wall=%4llu us (dsp=%4llu rr=%4llu wait=%llu polls=%llu conc=%d)\n",
                   config_idx + 1, n_configs,
                   config->entry_points, config->pool_size,
                   100.0*(double)res->recall1_hits   / (double)num_queries,
                   100.0*(double)res->recall10_hits  / ((double)num_queries*10.0),
                   100.0*(double)res->recall100_hits / ((double)num_queries*100.0),
                   res->total_us_sum  / (unsigned long long)num_queries,
                   res->phase1_us_sum / (unsigned long long)num_queries,
                   res->phase2_us_sum / (unsigned long long)num_queries,
                   res->cpu_wait_us / (unsigned long long)num_queries,
                   res->dsp_wait_poll_count / (unsigned long long)num_queries,
                   db_pair_batch_width * 2);
        }
        fflush(stdout);
    }

    printf("\n  %-5s %-6s %9s %9s %10s %10s %9s %9s %10s %8s %8s\n",
           "Entry","Pool","R@1(%)","R@10(%)","R@100(%)",
           "wall_us","dsp_us","rerank_us",
           "cpu_wait","polls","dsp_inv");
    printf("  %-5s %-6s %9s %9s %10s %10s %9s %9s %10s %8s %8s\n",
           "-----","------","-------","--------","---------",
           "-------","------","--------",
           "---------","-----","-------");
    for (config_idx = 0; config_idx < n_configs; config_idx++) {
        DemoResult *res = &results[config_idx];
        printf("  %-5d %-6d %8.2f%% %8.2f%% %9.2f%% %10llu %9llu %9llu %10llu %8llu %8llu\n",
               configs[config_idx].entry_points,
               configs[config_idx].pool_size,
               100.0*(double)res->recall1_hits   / (double)num_queries,
               100.0*(double)res->recall10_hits  / ((double)num_queries*10.0),
               100.0*(double)res->recall100_hits / ((double)num_queries*100.0),
               res->total_us_sum  / (unsigned long long)num_queries,
               res->phase1_us_sum / (unsigned long long)num_queries,
               res->phase2_us_sum / (unsigned long long)num_queries,
               res->cpu_wait_us / (unsigned long long)num_queries,
               res->dsp_wait_poll_count /
                   (unsigned long long)num_queries,
               res->dsp_invalidate_us /
                   (unsigned long long)num_queries);
    }

cleanup:
    demo_db_buffers_destroy_all(domain_id);

    if (g_residual_fd >= 0) {
        close(g_residual_fd);
        g_residual_fd = -1;
    }
    pthread_mutex_destroy(&g_residual_stats.mu);
    {
        int r;
        printf("\nResidual cache stats:\n");
        for (r = 0; r < RESIDUAL_CACHE_ROLE__COUNT; r++) {
            uint64_t h = g_residual_stats.hits[r];
            uint64_t m = g_residual_stats.misses[r];
            uint64_t e = g_residual_stats.errors[r];
            if (h + m + e > 0) {
                printf("  role=%d: hits=%llu misses=%llu errors=%llu hit_rate=%.2f%%\n",
                       r, (unsigned long long)h, (unsigned long long)m,
                       (unsigned long long)e,
                       (h + m > 0) ? 100.0 * (double)h / (double)(h + m) : 0.0);
            }
        }
        if (g_residual_stats.io_error)
            printf("  ** I/O ERROR detected -- results may be invalid **\n");
        fflush(stdout);
    }

    if (use_dma) {
        calculator_retrieval_unload_dma(handle);
        demo_dma_unmap();
    } else {
        calculator_retrieval_unload(handle);
    }
    demo_free_loaded_data();
    return ret;
}

#if 1

static int sqmd_final_decode(void *provider, int id, float *vector) {
    ResidualBlockCache **cache = (ResidualBlockCache **)provider;
    const uint8_t *r;
    const QCResidualQuantParams *qp;
    if (id < 0 || id >= g_dataset.num_vectors || !cache || !*cache)
        return RETRIEVAL_REFINE_INVALID;
    r = residual_cache_get(*cache, id);
    if (!r) return RETRIEVAL_REFINE_IO;
    qp = &g_dataset.residual_qparams[id];
    return retrieval_final_refine_decode_sq(demo_l1_code_ptr(&g_dataset,id),
        g_dataset.l1_bits,g_dataset.l1_qparams.scale,g_dataset.l1_qparams.zero_point,
        r,g_dataset.residual_bits,qp->scale,qp->zero_point,g_dataset.dim,vector);
}
static int refine_time_compare(const void *a, const void *b) {
    uint64_t x=*(const uint64_t *)a,y=*(const uint64_t *)b;
    return (x>y)-(x<y);
}
#endif

int run_retrieval_search_rpc_demo_sqmd(remote_handle64 handle, int domain_id) {
    const int entry_vals[] = RETRIEVAL_DEMO_ENTRY_SWEEP_VALUES;
    const int pool_vals[]  = RETRIEVAL_DEMO_POOL_SWEEP_VALUES;
    const int n_entry   = (int)(sizeof(entry_vals) / sizeof(entry_vals[0]));
    const int n_pool    = (int)(sizeof(pool_vals)  / sizeof(pool_vals[0]));
    const int n_configs = n_entry * n_pool;
    RetrievalDemoConfig configs[40];
    DemoResult results[40];
    int num_queries = RETRIEVAL_RPC_DEMO_QUERY_LIMIT;
    int rerank_count = FINAL_REFINE_MODE ? FINAL_REFINE_CANDIDATES : SEARCH_TOP_K * SEARCH_RERANK_FACTOR;
    int config_idx = 0;
    int i;
    int ret = 0;
    int use_dma = 0;
    int residual_stats_initialized = 0;
    int max_inflight_sources = 1;
    int split_enabled = 0;
    unsigned long long split_bias_load_us = 0;
    unsigned long long t_load;
#if 1
    RetrievalFinalRefineContext *refine_ctx = NULL;
    ResidualBlockCache *refine_base_cache = NULL;
    uint64_t *query_times = NULL;
    FILE *refine_trace = NULL;
    if (FINAL_REFINE_MODE) rerank_count = FINAL_REFINE_TRACE ? DSP_SEARCH_MAX_POOL : FINAL_REFINE_CANDIDATES;
#endif

    if (num_queries > DATA_N_QUERIES) num_queries = DATA_N_QUERIES;
    if (num_queries <= 0)             num_queries = DATA_N_QUERIES;
    if (sqmd_parse_max_inflight_sources(&max_inflight_sources) != 0)
        return -1;
    if (sqmd_parse_split_distance(&split_enabled) != 0)
        return -1;
#if !SQMD_USE_SESSION
    if (split_enabled) {
        printf("  ERROR: SQMD split distance requires SQMD_USE_SESSION=1\n");
        return -1;
    }
#endif

    for (i = 0; i < n_entry; i++) {
        int p;
        for (p = 0; p < n_pool; p++) {
            configs[config_idx].entry_points  = entry_vals[i];
            configs[config_idx].expand_count  = rover_config.expand_count;
            configs[config_idx].pool_size     = pool_vals[p];
            configs[config_idx].top_k         = SEARCH_TOP_K;
            configs[config_idx].rerank_factor = SEARCH_RERANK_FACTOR;
            config_idx++;
        }
    }

    printf("\n=== Retrieval Demo: SQMD CPU-owner single-query path (Phase 8) ===\n");
    printf("N=%d DIM=%d GRAPH_K=%d QUERIES=%d K=%d EXPAND=%d lanes=%d\n",
           DATA_N_BASE, DATA_DIM, DATA_GRAPH_K,
           num_queries, SEARCH_TOP_K, SEARCH_EXPAND_COUNT, rover_config.lanes);
    printf("[SQMD evaluation] output_top_k=%d; use only Recall@%d for this run\n",
           SEARCH_TOP_K, SEARCH_TOP_K);
    printf("DATA_LAYOUT=%s DATA_DIR=%s\n", DATA_LAYOUT_NAME, DATA_DIR);
    printf("GRAPH_SCHEME=%d(%s) GRAPH_FILE=%s GRAPH_ENTRY=%d\n",
           RETRIEVAL_GRAPH_INDEX_SCHEME,
           DATA_GRAPH_INDEX_SCHEME_NAME, DATA_GRAPH_FILENAME,
           DATA_GRAPH_ENTRY_POINT);
    printf("%d entry x %d pool = %d configs\n\n", n_entry, n_pool, n_configs);
    fflush(stdout);

    printf("[1/5] Encoding base vectors...\n");
    fflush(stdout);
    {
        char path[4096];
        snprintf(path, sizeof(path), "%s", DATA_BASE_PATH);
        t_load = now_us();


#if RESIDUAL_REORDER_SCHEME == 0
        {
            const char *ufs_path = residual_resolve_ufs_path();
            const char *qp_path  = RESIDUAL_QPARAMS_CACHE_PATH;
#if RETRIEVAL_DEMO_ENCODE_SOURCE == 1
            printf("  loading pre-encoded L1+residual (encode skipped)\n");
            ret = retrieval_demo_load_l1_and_qparams_from_disk(
                L1_CODES_CACHE_PATH,
                qp_path,
                DATA_N_BASE, DATA_DIM, RETRIEVAL_DEMO_RESIDUAL_BITS,
                &g_dataset);
if (RETRIEVAL_DEMO_USE_DISK == 0) {
            if (ret == 0) {
                ret = retrieval_demo_load_residual_codes_from_block_file(
                    RESIDUAL_UFS_PATH, &g_dataset);
            }
}
            if (ret != 0) {
                printf("  ERROR: pre-encoded files missing/corrupt "
                       "(expected %s + %s + %s)\n",
                       L1_CODES_CACHE_PATH, RESIDUAL_UFS_PATH, qp_path);
                fflush(stdout);
                goto cleanup;
            }
#else
if (RETRIEVAL_DEMO_USE_DISK == 0) {
            ret = retrieval_demo_encode_dataset_from_fvecs_streaming(
                path, DATA_N_BASE, DATA_DIM, &g_dataset);
            if (ret != 0) { printf("  ERROR (in-memory encode)\n"); fflush(stdout); goto cleanup; }
} else {
            ret = retrieval_demo_encode_dataset_streaming_to_disk(
                path, DATA_N_BASE, DATA_DIM,
                ufs_path ? ufs_path : RESIDUAL_UFS_PATH,
                qp_path,
                NULL,
                &g_dataset);
            if (ret != 0) { printf("  ERROR (streaming encode)\n"); fflush(stdout); goto cleanup; }
}
#endif
        }
#else
        ret = retrieval_demo_encode_dataset_from_fvecs_streaming(
            path, DATA_N_BASE, DATA_DIM, &g_dataset);
        if (ret != 0) { printf("  ERROR\n"); fflush(stdout); goto cleanup; }
#endif
        printf("  L1 int%d scale=%.6f zero=%.4f bytes/vec=%d time=%llu ms\n",
               g_dataset.l1_bits,
               g_dataset.l1_qparams.scale, g_dataset.l1_qparams.zero_point,
               g_dataset.l1_code_bytes,
               (now_us() - t_load) / 1000ULL);
        fflush(stdout);
        if (demo_promote_l1_codes_to_shared(&g_dataset) != 0) {
            printf("  ERROR: cannot allocate shared l1_codes buffer\n");
            fflush(stdout);
            ret = -1;
            goto cleanup;
        }
        printf("  l1_codes storage: %s\n",
               g_dataset.l1_codes_uses_rpcmem ? "rpcmem" : "malloc fallback");
        fflush(stdout);
    }

    printf("[2/5] Loading %s graph (%s, n=%d, k=%d)...\n",
           DATA_GRAPH_INDEX_SCHEME_NAME, DATA_GRAPH_FILENAME,
           DATA_N_BASE, DATA_GRAPH_K);
    fflush(stdout);
    if (demo_resolve_graph_index(handle,
                                 domain_id,
                                 &g_graph,
                                 &g_graph_uses_rpcmem,
                                 &g_nsg_entry_point) != 0) {
        ret = -1; goto cleanup;
    }


    {
        const char *ufs_path = residual_resolve_ufs_path();
#if RESIDUAL_REORDER_SCHEME == 0
        if (ufs_path) {
            int rfd = open(ufs_path, O_RDONLY);
            if (rfd >= 0) {
                g_residual_fd = rfd;
                g_residual_vpb = RESIDUAL_BLOCK_SIZE / g_dataset.residual_code_bytes;
                g_residual_total_blocks = (g_dataset.num_vectors + g_residual_vpb - 1)
                                          / g_residual_vpb;
                printf("  Residual codes -> UFS (%s), %d blocks, vpb=%d\n",
                       ufs_path, g_residual_total_blocks, g_residual_vpb);
            } else {
                g_residual_fd = -1;
                printf("  WARN: open(%s) failed, using passthrough\n", ufs_path);
            }
        } else {
            g_residual_fd = -1;
        }
#else
        if (ufs_path &&
            residual_write_to_file(&g_dataset, ufs_path, g_reverse_map) == 0) {
            int rfd = open(ufs_path, O_RDONLY);
            if (rfd >= 0) {
                g_residual_fd = rfd;
                g_residual_vpb = RESIDUAL_BLOCK_SIZE / g_dataset.residual_code_bytes;
                g_residual_total_blocks = (g_dataset.num_vectors + g_residual_vpb - 1)
                                          / g_residual_vpb;
                free(g_dataset.residual_codes);
                g_dataset.residual_codes = NULL;
                printf("  Residual codes -> UFS (%s), %d blocks, vpb=%d\n",
                       ufs_path, g_residual_total_blocks, g_residual_vpb);
            } else {
                g_residual_fd = -1;
                printf("  WARN: open(%s) failed, using passthrough\n", ufs_path);
                unlink(ufs_path);
            }
        } else {
            g_residual_fd = -1;
if (RETRIEVAL_DEMO_USE_DISK != 0) {
            if (!ufs_path)
                printf("  WARN: residual UFS path disabled, using memory fallback\n");
            if (ufs_path) printf("  WARN: residual write failed, using passthrough\n");
}
        }
#endif
        if (g_residual_fd < 0) {
            if (!g_dataset.residual_codes) {
                printf("  ERROR: residual codes have no UFS or memory backing\n");
                ret = -1;
                goto cleanup;
            }
            printf("  Residual codes -> memory (%llu bytes)\n",
                   (unsigned long long)g_dataset.num_vectors *
                   (unsigned long long)g_dataset.residual_code_bytes);
            g_residual_vpb = RESIDUAL_BLOCK_SIZE / g_dataset.residual_code_bytes;
            g_residual_total_blocks = (g_dataset.num_vectors + g_residual_vpb - 1)
                                      / g_residual_vpb;
        }
        memset(&g_residual_stats, 0, sizeof(g_residual_stats));
        pthread_mutex_init(&g_residual_stats.mu, NULL);
        residual_stats_initialized = 1;
        fflush(stdout);
    }

    printf("[3/5] Loading queries and ground truth...\n");
    fflush(stdout);
    {
        char qp[4096], gtp[4096];
        snprintf(qp,  sizeof(qp),  "%s", DATA_QUERY_PATH);
        snprintf(gtp, sizeof(gtp), "%s", DATA_GT_PATH);
        ret  = load_fvecs_n(qp,  num_queries, DATA_DIM,   &g_queries);
        ret |= load_ivecs_n(gtp, num_queries, DEMO_GT_K,  &g_gt);
        if (ret != 0) { printf("  ERROR\n"); fflush(stdout); goto cleanup; }
        printf("  done. %d queries\n", num_queries);
        fflush(stdout);

#if RETRIEVAL_GRAPH_INDEX_SCHEME == 1
        {
            int seed_id = -1;
            float seed_dist = 0.0f;
            int seed_count = sqmd_make_seed(&g_dataset,
                                            g_queries,
                                            NULL,
                                            1,
                                            g_nsg_entry_point,
                                            ENTRY_POINT_SEED,
                                            &seed_id,
                                            &seed_dist,
                                            1);

            if (seed_count != 1 || seed_id != g_nsg_entry_point) {
                printf("  ERROR: SQMD seed diagnostic failed "
                       "(count=%d seed[0]=%d graph_entry=%d)\n",
                       seed_count, seed_id, g_nsg_entry_point);
                fflush(stdout);
                ret = -1;
                goto cleanup;
            }
            printf("  SQMD seed diagnostic: seed[0]=%d graph_entry=%d\n",
                   seed_id, g_nsg_entry_point);
            fflush(stdout);
        }
#endif
    }

    if (split_enabled) {
        unsigned long long load_start = now_us();

        if (g_dataset.l1_bits != 4 ||
            (g_dataset.residual_bits != 4 && g_dataset.residual_bits != 8) ||
            sqmd_load_correction_bias(SQMD_CORRECTION_BIAS_PATH,
                                      &g_dataset,
                                      &g_sqmd_correction_bias) != 0) {
            printf("  ERROR: split distance requires verified U4/U4 or U4/U8 bias "
                   "artifact %s\n", SQMD_CORRECTION_BIAS_PATH);
            fflush(stdout);
            ret = -1;
            goto cleanup;
        }
        split_bias_load_us = now_us() - load_start;
        printf("  SQMD split distance: ON bias=%s bytes=%llu load=%llu us\n",
               SQMD_CORRECTION_BIAS_PATH,
               (unsigned long long)g_dataset.num_vectors * sizeof(float),
               split_bias_load_us);
        fflush(stdout);
    } else {
        printf("  SQMD split distance: OFF (legacy full correction)\n");
        fflush(stdout);
    }

#if 1
    if (FINAL_REFINE_MODE) {
        const uint8_t *artifact_hash = FINAL_REFINE_SHA256_BYTES;
        RetrievalFinalRefineConfig fc;
        memset(&fc,0,sizeof(fc));
        fc.graph_to_original=rover_graph_to_original();
        fc.graph_to_storage=g_reorder_map;
        fc.mode=FINAL_REFINE_MODE; fc.num_vectors=g_dataset.num_vectors;
        fc.dim=g_dataset.dim; fc.capacity=FINAL_REFINE_CANDIDATES;
        fc.cache_slots=FINAL_REFINE_CACHE_SLOTS;
        fc.decode=sqmd_final_decode; fc.provider=&refine_base_cache;
        fc.artifact_path=FINAL_REFINE_PATH; fc.artifact_sha256=artifact_hash;
        fc.base_paths[0]=L1_CODES_CACHE_PATH; fc.base_paths[1]=RESIDUAL_UFS_PATH;
        fc.base_paths[2]=RESIDUAL_QPARAMS_CACHE_PATH;
        if (RESIDUAL_REORDER_SCHEME != 0 ||
            RETRIEVAL_DEMO_ENCODE_SOURCE != 1 ||
            retrieval_final_refine_open(&fc,&refine_ctx) != 0) {
            printf("  ERROR: Phase15 refinement artifact/provider validation failed\n");
            ret=-1; goto cleanup;
        }
        query_times=malloc((size_t)num_queries*sizeof(*query_times));
        if (!query_times) { ret=-1; goto cleanup; }
        if (FINAL_REFINE_TRACE) {
            refine_trace=fopen(FINAL_REFINE_TRACE_PATH,"wbx");
            if (!refine_trace || fwrite("P15TRC01",1,8,refine_trace)!=8) { ret=-1; goto cleanup; }
        }
        printf("[Phase15] mode=%d factor=%d budget=%d trace=%d numeric=f32-separate-f64-l2\n",
            FINAL_REFINE_MODE,FINAL_REFINE_FACTOR,FINAL_REFINE_CANDIDATES,FINAL_REFINE_TRACE);
    }
#endif

    printf("[4/5] Loading index to DSP for SQMD expand engine...\n");
    fflush(stdout);
    {
        t_load = now_us();

        if (g_dataset.l1_codes_uses_rpcmem && fastrpc_mmap) {
            int l1_fds[MAX_L1_CHUNKS];

            g_dma_domain_id = domain_id;
            if (demo_map_l1_chunks(domain_id, l1_fds) >= 0) {
                ret = calculator_retrieval_load_dma(
                    handle,
                    RETRIEVAL_DMA_NO_GRAPH_FD,
                    l1_fds, g_dataset.l1_chunk_count,
                    g_dataset.l1_chunk_vectors,
                    DATA_N_BASE,
                    DATA_DIM,
                    DATA_GRAPH_K,
                    g_nsg_entry_point,
                    g_dataset.l1_qparams.scale,
                    g_dataset.l1_qparams.zero_point);
                if (ret == 0) {
                    use_dma = 1;
                } else {
                    printf("  WARN: retrieval_load_dma failed 0x%x\n", ret);
                    fflush(stdout);
                    demo_dma_unmap();
                    ret = 0;
                }
            }
        }

        if (!use_dma) {

            size_t l1_total =
                (size_t)DATA_N_BASE * (size_t)g_dataset.l1_code_bytes;
            if (l1_total > (size_t)INT_MAX) {
                printf("  ERROR: DMA zero-copy load unavailable and L1 codes "
                       "(%llu bytes) exceed the by-value RPC limit; refusing "
                       "the by-value path (would OOM the device).\n",
                       (unsigned long long)l1_total);
                fflush(stdout);
                ret = -1;
                goto cleanup;
            }
            ret = calculator_retrieval_load(
                handle,
                g_graph, DATA_N_BASE * DATA_GRAPH_K,
                g_dataset.l1_codes, DATA_N_BASE * g_dataset.l1_code_bytes,
                DATA_N_BASE,
                DATA_DIM,
                DATA_GRAPH_K,
                g_nsg_entry_point,
                g_dataset.l1_qparams.scale,
                g_dataset.l1_qparams.zero_point);
        }

        if (ret != 0) {
            printf("  ERROR: retrieval_load%s failed 0x%x\n",
                   use_dma ? "_dma" : "", ret);
            fflush(stdout);
            goto cleanup;
        }
        if (use_dma) {
            printf("  done. time=%llu ms (DMA zero-copy, CPU-owned graph, "
                   "%d L1 chunks, vpc=%d)\n",
                   (now_us() - t_load) / 1000ULL,
                   g_dataset.l1_chunk_count, g_dataset.l1_chunk_vectors);
        } else {
            printf("  done. time=%llu ms (copy)\n",
                   (now_us() - t_load) / 1000ULL);
        }
        fflush(stdout);
    }

    printf("[5/5] Setting up SQMD shared control (%zu bytes)...\n",
           sizeof(SqmdSharedControl));
    fflush(stdout);
    if (demo_sqmd_buffer_init(&g_sqmd_shared,
                              domain_id,
                              rover_config.lanes) != 0) {
        printf("  ERROR: SQMD shared control init failed\n");
        fflush(stdout);
        ret = -1;
        goto cleanup;
    }
    printf("  done. lanes=%d buffers/lane=%d\n",
           rover_config.lanes, SQMD_BUFFERS_PER_LANE);
    fflush(stdout);
    printf("[SQMD] Running %d configs x %d queries"
           " (mode=%s, source_window=%d, split=%s, progress=%s, turn timeout=%s)...\n",
           n_configs,
           num_queries,
           SQMD_USE_SESSION ? "session" : "per-query",
           max_inflight_sources,
           split_enabled ? (g_dataset.residual_bits == 8 ? "u8-u8" : "u8-u4") : "off",
           SQMD_PROGRESS_EVERY_QUERIES > 0 ? "on" : "off",
           SQMD_TURN_WAIT_TIMEOUT_US > 0 ? "on" : "off");
    fflush(stdout);

    for (config_idx = 0; config_idx < n_configs; config_idx++) {
        const RetrievalDemoConfig *config = &configs[config_idx];
        DemoResult *res = &results[config_idx];
        RetrievalPipeline pipe;
        int num_out = FINAL_REFINE_MODE ? FINAL_REFINE_CANDIDATES : config->top_k * config->rerank_factor;
#if 1
        RetrievalFinalRefineStats refine_totals = {0};
        uint64_t refine_us = 0;
        num_out = rerank_count;
        retrieval_final_refine_reset(refine_ctx);
#endif
        int query_idx;
        unsigned long long t_cfg;
#if SQMD_USE_SESSION
        pthread_t sqmd_session_thread;
        RetrievalRpcSqmdSessionArgs sqmd_session_args;
        int sqmd_session_open_ok = 0;
#endif

        memset(res, 0, sizeof(*res));
        if (pipeline_init(&pipe, 1, rerank_count, results, configs) != 0) {
            ret = -1; goto cleanup;
        }
        if (residual_init_slot_cache(&pipe.slots[0], &g_dataset) != 0) {
            pipeline_destroy(&pipe);
            ret = -1; goto cleanup;
        }

        t_cfg = now_us();
#if SQMD_USE_SESSION
        if (sqmd_session_open(handle, &g_sqmd_shared,
                              &sqmd_session_thread, &sqmd_session_args) != 0) {
            printf("  ERROR: SQMD session open failed (config %d/%d)\n",
                   config_idx + 1, n_configs);
            fflush(stdout);
            residual_cache_destroy(pipe.slots[0].res_cache);
            pipe.slots[0].res_cache = NULL;
            pipeline_destroy(&pipe);
            ret = -1; goto cleanup;
        }
        sqmd_session_open_ok = 1;
#endif
        for (query_idx = 0; query_idx < num_queries; query_idx++) {
            RetrievalSlot *slot = &pipe.slots[0];
            int result_count;
            unsigned long long t0;
            unsigned long long t1;

            if (SQMD_PROGRESS_EVERY_QUERIES > 0 &&
                (query_idx == 0 ||
                 query_idx % SQMD_PROGRESS_EVERY_QUERIES == 0)) {
                printf("[SQMD] config %d/%d query %d/%d\n",
                       config_idx + 1,
                       n_configs,
                       query_idx + 1,
                       num_queries);
                fflush(stdout);
            }

            slot->query = g_queries + (size_t)query_idx * DATA_DIM;
            slot->gt_row = g_gt + (size_t)query_idx * DEMO_GT_K;
            slot->query_idx = query_idx;
            slot->config_idx = config_idx;
            slot->actual_count = 0;

            t0 = now_us();
#if SQMD_USE_SESSION
            result_count = sqmd_search_one_query_session(&g_dataset,
                                                         g_graph,
                                                         DATA_GRAPH_K,
                                                         g_nsg_entry_point,
                                                         slot->query,
                                                         config,
                                                         slot->res_cache,
                                                         &g_sqmd_shared,
                                                         &sqmd_session_args,
                                                         max_inflight_sources,
                                                         split_enabled,
                                                         res,
                                                         slot->candidate_ids,
                                                         slot->candidate_dists,
                                                         num_out);
#else
            result_count = sqmd_search_one_query_rpc(&g_dataset,
                                                     g_graph,
                                                     DATA_GRAPH_K,
                                                     g_nsg_entry_point,
                                                     slot->query,
                                                     config,
                                                     slot->res_cache,
                                                     handle,
                                                     &g_sqmd_shared,
                                                     max_inflight_sources,
                                                     slot->candidate_ids,
                                                     slot->candidate_dists,
                                                     num_out);
#endif
            t1 = now_us();
            if (result_count < 0) {
                ret = -1;
                break;
            }
#if !SQMD_USE_SESSION

            demo_sqmd_accumulate_stats(res, &g_sqmd_shared.ctrl->stats);
#endif
            slot->actual_count = result_count;
            res->phase1_us_sum += t1 - t0;

            {
                int top_k = config->top_k;
                int recall_limit;
                unsigned long long rr0 = now_us();

#if 1
                if (FINAL_REFINE_MODE) {
                    RetrievalFinalRefineStats rs;
                    uint64_t rt;
                    int count=slot->actual_count;
                    int refined;
                    if (refine_trace) {
                        uint32_t h[4]={(uint32_t)config->entry_points,(uint32_t)config->pool_size,
                            (uint32_t)query_idx,(uint32_t)count};
                        if (fwrite(h,sizeof(h),1,refine_trace)!=1 ||
                            fwrite(slot->candidate_ids,4,count,refine_trace)!=(size_t)count ||
                            fwrite(slot->candidate_dists,4,count,refine_trace)!=(size_t)count) {
                            ret=-1; break;
                        }
                    }
                    if (count>FINAL_REFINE_CANDIDATES) count=FINAL_REFINE_CANDIDATES;
                    refine_base_cache=slot->res_cache;
                    rt=now_us();
                    refined=retrieval_final_refine(refine_ctx,slot->query,slot->candidate_ids,
                        slot->candidate_dists,count,top_k,&rs);
                    refine_us+=now_us()-rt;
                    if (refined<0) {
                        printf("  ERROR: Phase15 refinement failed code=%d query=%d\n",refined,query_idx);
                        ret=-1; break;
                    }
                    slot->actual_count=refined;
                    refine_totals.candidates+=rs.candidates;
                    refine_totals.cache_hits+=rs.cache_hits;
                    refine_totals.code_blocks+=rs.code_blocks;
                    refine_totals.code_bytes+=rs.code_bytes;
                    refine_totals.qparams_bytes=rs.qparams_bytes;
                    refine_totals.resident_bytes=rs.resident_bytes;
                    refine_totals.artifact_bytes=rs.artifact_bytes;
                    if (refine_trace) {
                        uint32_t n=(uint32_t)refined;
                        if (fwrite(&n,4,1,refine_trace)!=1 ||
                            fwrite(slot->candidate_ids,4,n,refine_trace)!=n ||
                            fwrite(slot->candidate_dists,4,n,refine_trace)!=n) { ret=-1; break; }
                    }
                }
else if (split_enabled) {
                    int reranked = demo_rerank_candidates(
                        slot->query, config, slot->candidate_ids,
                        slot->candidate_dists, slot->actual_count,
                        NULL, slot->res_cache);
                    if (reranked < 0) {
                        ret = -1;
                        break;
                    }
                    slot->actual_count = reranked;
                }
#endif
                recall_limit = (slot->actual_count < top_k)
                                   ? slot->actual_count : top_k;

                if (rover_write_result(query_idx,slot->candidate_ids,slot->candidate_dists,recall_limit)) {ret=-1; break;}
                res->recall1_hits += demo_recall_hits_at_k(
                    slot->candidate_ids, recall_limit,
                    slot->gt_row, DEMO_GT_K, 1);
                res->recall10_hits += demo_recall_hits_at_k(
                    slot->candidate_ids, recall_limit,
                    slot->gt_row, DEMO_GT_K, 10);
                res->recall100_hits += demo_recall_hits_at_k(
                    slot->candidate_ids, recall_limit,
                    slot->gt_row, DEMO_GT_K, 100);
                res->phase2_us_sum += now_us() - rr0;
            }
#if 1
            if (query_times) query_times[query_idx]=now_us()-t0;
#endif
        }
#if SQMD_USE_SESSION
        if (sqmd_session_open_ok) {
            if (sqmd_session_close(&g_sqmd_shared, sqmd_session_thread,
                                   &sqmd_session_args) != 0 && ret == 0)
                ret = -1;

            demo_sqmd_accumulate_stats(res, &g_sqmd_shared.ctrl->stats);
        }
#endif
        res->total_us_sum = now_us() - t_cfg;

        residual_cache_flush_stats(pipe.slots[0].res_cache,
                                   &g_residual_stats,
                                   RESIDUAL_CACHE_ROLE_CORRECTION);
        residual_cache_destroy(pipe.slots[0].res_cache);
        pipe.slots[0].res_cache = NULL;
        pipeline_destroy(&pipe);
        if (ret != 0) goto cleanup;

        printf("  [%2d/%d] e=%3d p=%4d  R@1=%5.2f R@10=%5.2f R@100=%5.2f"
               "  wall=%4llu us (sqmd=%4llu rr=%4llu slot_count=1 lanes=%d"
               " work=%llu dedup=%llu empty=%llu dsp_compute=%llu cpu_corr=%llu"
               " cpu_wait=%llu dsp_wait=%llu polls=%llu abort=%llu)\n",
               config_idx + 1, n_configs,
               config->entry_points, config->pool_size,
               100.0*(double)res->recall1_hits   / (double)num_queries,
               100.0*(double)res->recall10_hits  / ((double)num_queries*10.0),
               100.0*(double)res->recall100_hits / ((double)num_queries*100.0),
               res->total_us_sum  / (unsigned long long)num_queries,
               res->phase1_us_sum / (unsigned long long)num_queries,
               res->phase2_us_sum / (unsigned long long)num_queries,
               rover_config.lanes,
               res->sqmd_work_items / (unsigned long long)num_queries,
               res->sqmd_dedup_skips / (unsigned long long)num_queries,
               res->sqmd_empty_turns / (unsigned long long)num_queries,
               res->sqmd_dsp_compute_us / (unsigned long long)num_queries,
               res->sqmd_cpu_correct_us / (unsigned long long)num_queries,
               res->sqmd_cpu_wait_dsp_us / (unsigned long long)num_queries,
               res->sqmd_dsp_wait_cpu_us / (unsigned long long)num_queries,
               res->sqmd_dsp_wait_poll_count / (unsigned long long)num_queries,
               res->sqmd_abort_count);
        printf("  [SQMD pipeline] window=%d jobs=%llu retired=%llu"
               " speculative=%llu starvation=%llu highwater=%llu\n",
               max_inflight_sources,
               res->sqmd_source_jobs_created,
               res->sqmd_source_jobs_retired,
               res->sqmd_speculative_sources,
               res->sqmd_source_window_starvation,
               res->sqmd_inflight_sources_highwater);
        if (split_enabled) {
            double rel = res->query_u8_norm_sq > 0.0
                ? sqrt(res->query_u8_error_sq / res->query_u8_norm_sq)
                : 0.0;
            printf("  [SQMD split] queries=%llu candidates=%llu bias_load_us=%llu"
                   " clip_low=%llu clip_high=%llu query_rel_l2=%.9g\n",
                   res->split_queries,
                   res->split_candidates,
                   split_bias_load_us,
                   res->query_u8_clipped_low,
                   res->query_u8_clipped_high,
                   rel);
        }
#if 1
        if (FINAL_REFINE_MODE) {
            struct rusage usage;
            memset(&usage,0,sizeof(usage));
            getrusage(RUSAGE_SELF,&usage);
            qsort(query_times,(size_t)num_queries,sizeof(*query_times),refine_time_compare);
            printf("  [Phase15 stats] top_k=%d mode=%d factor=%d budget=%d queries=%d candidates=%llu refine_us=%llu"
                " blocks=%llu bytes=%llu hits=%llu qparams_bytes=%llu resident_bytes=%llu artifact_bytes=%llu"
                " p50_us=%llu p95_us=%llu p99_us=%llu peak_rss_kib=%ld trace=%d\n",
                config->top_k,FINAL_REFINE_MODE,FINAL_REFINE_FACTOR,FINAL_REFINE_CANDIDATES,num_queries,
                (unsigned long long)refine_totals.candidates,(unsigned long long)refine_us,
                (unsigned long long)refine_totals.code_blocks,(unsigned long long)refine_totals.code_bytes,
                (unsigned long long)refine_totals.cache_hits,(unsigned long long)refine_totals.qparams_bytes,
                (unsigned long long)refine_totals.resident_bytes,(unsigned long long)refine_totals.artifact_bytes,
                (unsigned long long)query_times[(num_queries-1)*50/100],
                (unsigned long long)query_times[(num_queries-1)*95/100],
                (unsigned long long)query_times[(num_queries-1)*99/100],usage.ru_maxrss,FINAL_REFINE_TRACE);
        }
#endif
        fflush(stdout);
    }

cleanup:
#if 1
    if (refine_trace && fclose(refine_trace)!=0) ret=-1;
    free(query_times);
    retrieval_final_refine_close(refine_ctx);
#endif
    free(g_sqmd_correction_bias);
    g_sqmd_correction_bias = NULL;
    demo_sqmd_buffer_destroy(&g_sqmd_shared, domain_id);
    if (g_residual_fd >= 0) {
        close(g_residual_fd);
        g_residual_fd = -1;
    }
    if (residual_stats_initialized) {
        pthread_mutex_destroy(&g_residual_stats.mu);
        {
            int r;
            printf("\nResidual cache stats:\n");
            for (r = 0; r < RESIDUAL_CACHE_ROLE__COUNT; r++) {
                uint64_t h = g_residual_stats.hits[r];
                uint64_t m = g_residual_stats.misses[r];
                uint64_t e = g_residual_stats.errors[r];
                if (h + m + e > 0) {
                    printf("  role=%d: hits=%llu misses=%llu errors=%llu hit_rate=%.2f%%\n",
                           r, (unsigned long long)h, (unsigned long long)m,
                           (unsigned long long)e,
                           (h + m > 0) ? 100.0 * (double)h / (double)(h + m) : 0.0);
                }
            }
            if (g_residual_stats.io_error)
                printf("  ** I/O ERROR detected -- results may be invalid **\n");
            fflush(stdout);
        }
    }
    if (use_dma) {
        calculator_retrieval_unload_dma(handle);
        demo_dma_unmap();
    } else {
        calculator_retrieval_unload(handle);
    }
    demo_free_loaded_data();
    return ret;
}

int run_retrieval_search_rpc_demo_selected(remote_handle64 handle,
                                           int domain_id,
                                           const char *db_mode_env) {
    RetrievalDemoDbModeConfig mode_cfg;
#ifdef RETRIEVAL_ENABLE_FULL_BUILD

#endif

    if (retrieval_demo_db_mode_resolve(db_mode_env, &mode_cfg) != 0) {
        printf("ERROR: unsupported RETRIEVAL_DB_MODE=%s\n",
               db_mode_env ? db_mode_env : "(null)");
        return AEE_EBADPARM;
    }

    printf("RETRIEVAL_DB_MODE=%s\n", mode_cfg.mode_name);

    if (mode_cfg.mode == RETRIEVAL_DEMO_DB_MODE_PURE_PQ)
        return run_retrieval_search_rpc_demo_pure_pq(handle, domain_id);
    if (mode_cfg.mode == RETRIEVAL_DEMO_DB_MODE_SQMD)
        return run_retrieval_search_rpc_demo_sqmd(handle, domain_id);

    return run_retrieval_search_rpc_demo_db(handle, domain_id);
}

static int rover_open_sq_refine(ResidualBlockCache **cache,RetrievalFinalRefineContext **out) {
    RetrievalFinalRefineConfig fc={0};
    fc.graph_to_original=rover_graph_to_original();
        fc.graph_to_storage=g_reorder_map;
        fc.mode=FINAL_REFINE_MODE;fc.num_vectors=g_dataset.num_vectors;fc.dim=g_dataset.dim;
    fc.capacity=FINAL_REFINE_CANDIDATES;fc.cache_slots=FINAL_REFINE_CACHE_SLOTS;
    fc.decode=sqmd_final_decode;fc.provider=cache;fc.artifact_path=FINAL_REFINE_PATH;
    fc.artifact_sha256=rover_config.error4_hash;
    fc.base_paths[0]=L1_CODES_CACHE_PATH;fc.base_paths[1]=RESIDUAL_UFS_PATH;fc.base_paths[2]=RESIDUAL_QPARAMS_CACHE_PATH;
    return retrieval_final_refine_open(&fc,out);
}
