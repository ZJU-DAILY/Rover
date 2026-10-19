typedef struct PurePqSharedRuntime {
    PurePqSqmdSharedControl *control;
    PurePqQueryLutHeader *lut_header;
    float *lut;
    size_t control_bytes;
    size_t lut_bytes;
    int control_fd;
    int lut_fd;
    int control_mapped;
    int lut_mapped;
    int domain_id;
} PurePqSharedRuntime;

typedef struct PurePqSessionArgs {
    remote_handle64 handle;
    PurePqSharedRuntime *shared;
    int lane_count;
    int work_items_cap;
    int query_gen;
    volatile int done;
    int result;
} PurePqSessionArgs;

static int pure_pq_parse_fingerprint_hex(const char *hex,
                                         uint8_t fingerprint[32]) {
    int index;
    if (!hex || strlen(hex) != 64u || !fingerprint) {
        return -1;
    }
    for (index = 0; index < 32; index++) {
        int high = hex[index * 2];
        int low = hex[index * 2 + 1];
        high = high >= '0' && high <= '9' ? high - '0' :
               high >= 'a' && high <= 'f' ? high - 'a' + 10 :
               high >= 'A' && high <= 'F' ? high - 'A' + 10 : -1;
        low = low >= '0' && low <= '9' ? low - '0' :
              low >= 'a' && low <= 'f' ? low - 'a' + 10 :
              low >= 'A' && low <= 'F' ? low - 'A' + 10 : -1;
        if (high < 0 || low < 0) {
            return -1;
        }
        fingerprint[index] = (uint8_t)((high << 4) | low);
    }
    return 0;
}

static int pure_pq_online_residual_evaluator(void *context,
                                             int vector_id,
                                             float coarse_distance,
                                             float *out_distance) {
    return retrieval_pq_residual_query_evaluate(
        (RetrievalPqResidualQuery *)context,
        vector_id, coarse_distance, out_distance);
}

static void pure_pq_shared_destroy(PurePqSharedRuntime *shared) {
    if (!shared) {
        return;
    }
    if (shared->lut_mapped && fastrpc_munmap) {
        fastrpc_munmap(shared->domain_id, shared->lut_fd,
                       shared->lut_header, shared->lut_bytes);
    }
    if (shared->control_mapped && fastrpc_munmap) {
        fastrpc_munmap(shared->domain_id, shared->control_fd,
                       shared->control, shared->control_bytes);
    }
    if (shared->lut_header) {
        rpcmem_free(shared->lut_header);
    }
    if (shared->control) {
        rpcmem_free(shared->control);
    }
    memset(shared, 0, sizeof(*shared));
    shared->control_fd = -1;
    shared->lut_fd = -1;
    shared->domain_id = -1;
}

static int pure_pq_shared_init(PurePqSharedRuntime *shared,
                               int domain_id,
                               const RetrievalPqArtifactMetadata *metadata) {
    if (!shared || !metadata || !fastrpc_mmap) {
        return -1;
    }
    memset(shared, 0, sizeof(*shared));
    shared->control_fd = -1;
    shared->lut_fd = -1;
    shared->domain_id = domain_id;
    shared->control_bytes = sizeof(*shared->control);
    shared->lut_bytes = PURE_PQ_SQMD_CACHE_LINE_BYTES +
        (size_t)metadata->m * (size_t)metadata->ks * sizeof(float);
    if (shared->control_bytes > (size_t)INT_MAX ||
        shared->lut_bytes > (size_t)INT_MAX) {
        return -1;
    }
    shared->control = (PurePqSqmdSharedControl *)rpcmem_alloc(
        RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS,
        (int)shared->control_bytes);
    shared->lut_header = (PurePqQueryLutHeader *)rpcmem_alloc(
        RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS,
        (int)shared->lut_bytes);
    if (!shared->control || !shared->lut_header) {
        pure_pq_shared_destroy(shared);
        return -1;
    }
    memset(shared->control, 0, shared->control_bytes);
    memset(shared->lut_header, 0, shared->lut_bytes);
    shared->lut = (float *)((uint8_t *)shared->lut_header +
                            PURE_PQ_SQMD_CACHE_LINE_BYTES);
    shared->control_fd = rpcmem_to_fd(shared->control);
    shared->lut_fd = rpcmem_to_fd(shared->lut_header);
    if (shared->control_fd < 0 || shared->lut_fd < 0 ||
        fastrpc_mmap(domain_id, shared->control_fd, shared->control,
                     0, shared->control_bytes, FASTRPC_MAP_FD) != 0) {
        pure_pq_shared_destroy(shared);
        return -1;
    }
    shared->control_mapped = 1;
    if (fastrpc_mmap(domain_id, shared->lut_fd, shared->lut_header,
                     0, shared->lut_bytes, FASTRPC_MAP_FD) != 0) {
        pure_pq_shared_destroy(shared);
        return -1;
    }
    shared->lut_mapped = 1;
    return 0;
}

static void *pure_pq_session_thread(void *argument) {
    PurePqSessionArgs *args = (PurePqSessionArgs *)argument;
    if (!args || !args->shared) {
        return NULL;
    }
    args->result = calculator_retrieval_pure_pq_sqmd_run(
        args->handle,
        args->shared->control_fd,
        args->shared->lut_fd,
        args->lane_count,
        args->work_items_cap);
    DB_WRITE_BARRIER();
    args->done = 1;
    return NULL;
}

static int pure_pq_session_open(remote_handle64 handle,
                                PurePqSharedRuntime *shared,
                                int lane_count,
                                pthread_t *thread,
                                PurePqSessionArgs *args) {
    unsigned long long start;

    memset(args, 0, sizeof(*args));
    args->handle = handle;
    args->shared = shared;
    args->lane_count = lane_count;
    args->work_items_cap = PURE_PQ_SQMD_MAX_WORK_ITEMS;
    if (pthread_create(thread, NULL, pure_pq_session_thread, args) != 0) {
        return -1;
    }
    start = now_us();
    while (now_us() - start < 5000000ULL) {
        DB_READ_BARRIER();
        if (shared->control->control_state == PURE_PQ_SQMD_CONTROL_RUNNING) {
            return 0;
        }
        if (args->done ||
            shared->control->control_state == PURE_PQ_SQMD_CONTROL_ABORT) {
            pthread_join(*thread, NULL);
            return -1;
        }
        sched_yield();
    }
    shared->control->control_state = PURE_PQ_SQMD_CONTROL_ABORT;
    DB_WRITE_BARRIER();
    pthread_join(*thread, NULL);
    return -1;
}

static int pure_pq_session_close(PurePqSharedRuntime *shared,
                                 pthread_t thread,
                                 PurePqSessionArgs *args) {
    int aborted = shared->control->control_state ==
                  PURE_PQ_SQMD_CONTROL_ABORT;
    if (!aborted) {
        shared->control->control_state = PURE_PQ_SQMD_CONTROL_SHUTDOWN;
    }
    DB_WRITE_BARRIER();
    pthread_join(thread, NULL);
    return !aborted && args->result == AEE_SUCCESS ? 0 : -1;
}

static int pure_pq_make_seed_ids(int num_vectors, int entry_count,
                                 int graph_entry, int *seed_ids) {
    unsigned int seed = ENTRY_POINT_SEED;
    int count = entry_count < num_vectors ? entry_count : num_vectors;
    int index;

    for (index = 0; index < count; index++) {
        if (index == 0 && graph_entry >= 0 && graph_entry < num_vectors) {
            seed_ids[index] = graph_entry;
        } else {
            seed = seed * 1103515245u + 12345u;
            seed_ids[index] = (int)(((uint64_t)seed *
                                     (uint64_t)num_vectors) >> 32);
        }
    }
    return count;
}

static int pure_pq_run_one_query_limited(PurePqIndex *index,
                                 const int32_t *graph,
                                 int graph_degree,
                                 int graph_entry,
                                 const float *query,
                                 const RetrievalDemoConfig *config,
                                 PurePqSharedRuntime *shared,
                                 PurePqSessionArgs *session,
                                 RetrievalPqResidualQuery *residual_query,
                                 int *out_ids,
                                 float *out_distances,
                                 int out_capacity,
                                 uint64_t *out_work_items,
                                 RetrievalPqResidualStats *out_residual_stats,
                                 int source_window,
                                 PurePqSourcePipeline *out_pipeline,
                                 const RetrievalGraphReader *graph_reader, int prefix, PurePqCpuSearchState **reuse) {
    const RetrievalPqArtifactMetadata *metadata = pure_pq_index_metadata(index);
    PurePqCpuSearchState *state = reuse ? *reuse : NULL;
    PurePqDistancePolicy distance_policy;
    int *seed_ids = NULL;
    float *seed_distances = NULL;
    int seed_count;
    int result_count = -1;
    PurePqSourcePipeline pipeline;
    int query_gen;

    int logical=prefix>0?prefix:(int)metadata->num_vectors;
    if(logical>(int)metadata->num_vectors)return -1;
    if (pure_pq_source_pipeline_init(&pipeline, source_window) != 0) return -1;
    memset(&distance_policy, 0, sizeof(distance_policy));
    if (out_residual_stats) {
        memset(out_residual_stats, 0, sizeof(*out_residual_stats));
    }
    if (!metadata || !graph || !query || !config || !shared || !session ||
        !out_ids || !out_distances ||
        ((PURE_PQ_CORRECTION_MODE == 1) != (residual_query != NULL))) {
        return -1;
    }
    if (residual_query) {
        if (retrieval_pq_residual_query_bind(residual_query, query) != 0) {
            return -1;
        }
        distance_policy.mode = PURE_PQ_CORRECTION_ONLINE;
        distance_policy.evaluator = pure_pq_online_residual_evaluator;
        distance_policy.context = residual_query;
    }
    query_gen = ++session->query_gen;
    if (query_gen <= 0) {
        query_gen = 1;
        session->query_gen = 1;
    }
    if (pure_pq_index_build_lut(index, query, shared->lut,
                                (size_t)metadata->m * metadata->ks) != 0) {
        goto done;
    }
    shared->lut_header->m = (int32_t)metadata->m;
    shared->lut_header->ks = (int32_t)metadata->ks;
    shared->lut_header->lut_count = (int32_t)(metadata->m * metadata->ks);
    memcpy(shared->lut_header->index_uuid, metadata->index_uuid, 16);
    DB_WRITE_BARRIER();
    shared->lut_header->lut_gen = query_gen;
    shared->lut_header->query_gen = query_gen;
    shared->control->query_gen = query_gen;
    DB_WRITE_BARRIER();

    seed_ids = (int *)malloc((size_t)config->entry_points * sizeof(*seed_ids));
    seed_distances = (float *)malloc(
        (size_t)config->entry_points * sizeof(*seed_distances));
    if (!seed_ids || !seed_distances) {
        goto done;
    }
    seed_count = pure_pq_make_seed_ids(logical,
                                       config->entry_points,
                                       graph_entry, seed_ids);
    if (seed_count <= 0 ||
        pure_pq_index_score_ids(index, (const int32_t *)seed_ids,
                                seed_count, shared->lut,
                                seed_distances) != 0 ||
        ((!state && pure_pq_cpu_search_create((int)metadata->num_vectors,
            config->pool_size,config->expand_count,&state)) ||
         pure_pq_cpu_search_reset(state,logical,config->pool_size,config->expand_count)) ||
        pure_pq_cpu_search_set_distance_policy(
            state, residual_query ? &distance_policy : NULL) != 0 ||
        pure_pq_cpu_search_set_graph_reader(state, graph_reader) != 0 ||
        pure_pq_cpu_search_seed(state, seed_ids,
                                seed_distances, seed_count) != 0) {
        goto done;
    }

    {
        unsigned long long wait_start = now_us();
        uint64_t retired = 0;
        for (;;) {
            int rc;
            DB_READ_BARRIER();
            if (session->done || shared->control->control_state == PURE_PQ_SQMD_CONTROL_ABORT) goto done;
            rc = pure_pq_source_pipeline_step(&pipeline, state, graph,
                                              graph_degree, shared->control, query_gen);
            DB_WRITE_BARRIER();
            if (rc < 0) goto done;
            if (rc == 1) break;
            if (pipeline.retired != retired) {
                retired = pipeline.retired;
                wait_start = now_us();
            } else if (now_us() - wait_start > 30000000ULL) {
                goto done;
            } else {
                sched_yield();
            }
        }
    }
    shared->control->stats.work_items += pipeline.work_items;
    if (out_work_items) *out_work_items += pipeline.work_items;
    if (out_pipeline) *out_pipeline = pipeline;
    result_count = pure_pq_cpu_search_finish(state, out_ids,
                                             out_distances, out_capacity);
    if (result_count >= 0 && residual_query && out_residual_stats) {
        retrieval_pq_residual_query_stats(residual_query, out_residual_stats);
        if (out_residual_stats->cache_errors != 0u) {
            result_count = -1;
        }
    }

done:
    if(reuse)*reuse=state;else pure_pq_cpu_search_destroy(state);
    free(seed_ids);
    free(seed_distances);
    return result_count;
}
static int pure_pq_run_one_query_with_reader(PurePqIndex *index,
                                 const int32_t *graph,
                                 int graph_degree,
                                 int graph_entry,
                                 const float *query,
                                 const RetrievalDemoConfig *config,
                                 PurePqSharedRuntime *shared,
                                 PurePqSessionArgs *session,
                                 RetrievalPqResidualQuery *residual_query,
                                 int *out_ids,
                                 float *out_distances,
                                 int out_capacity,
                                 uint64_t *out_work_items,
                                 RetrievalPqResidualStats *out_residual_stats,
                                 int source_window,
                                 PurePqSourcePipeline *out_pipeline,
                                 const RetrievalGraphReader *graph_reader) {
    return pure_pq_run_one_query_limited(index,graph,graph_degree,graph_entry,query,config,
        shared,session,residual_query,out_ids,out_distances,out_capacity,out_work_items,
        out_residual_stats,source_window,out_pipeline,graph_reader,0,NULL);
}

static int pure_pq_run_one_query(PurePqIndex *index,
                                 const int32_t *graph,
                                 int graph_degree,
                                 int graph_entry,
                                 const float *query,
                                 const RetrievalDemoConfig *config,
                                 PurePqSharedRuntime *shared,
                                 PurePqSessionArgs *session,
                                 RetrievalPqResidualQuery *residual_query,
                                 int *out_ids,
                                 float *out_distances,
                                 int out_capacity,
                                 uint64_t *out_work_items,
                                 RetrievalPqResidualStats *out_residual_stats,
                                 int source_window,
                                 PurePqSourcePipeline *out_pipeline) {
    return pure_pq_run_one_query_with_reader(index,graph,graph_degree,graph_entry,query,config,
        shared,session,residual_query,out_ids,out_distances,out_capacity,out_work_items,
        out_residual_stats,source_window,out_pipeline,NULL);
}

static int pure_pq_refine_decode(void *provider, int id, float *out) {
    return retrieval_pq_residual_query_decode((RetrievalPqResidualQuery *)provider, id, out);
}

static uint64_t pure_pq_result_fingerprint(uint64_t hash, const int *ids,
                                           const float *distances, int count) {
    int i, byte;
    for (i = 0; i < count; ++i) {
        uint32_t values[2];
        values[0] = (uint32_t)ids[i];
        memcpy(&values[1], &distances[i], sizeof(float));
        for (byte = 0; byte < 8; ++byte) {
            hash ^= (values[byte / 4] >> (8 * (byte % 4))) & 255u;
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}

int run_retrieval_search_rpc_demo_pure_pq(remote_handle64 handle,
                                          int domain_id) {
    const int entry_values[] = RETRIEVAL_DEMO_ENTRY_SWEEP_VALUES;
    const int pool_values[] = RETRIEVAL_DEMO_POOL_SWEEP_VALUES;
    const int entry_count = (int)(sizeof(entry_values) / sizeof(entry_values[0]));
    const int pool_count = (int)(sizeof(pool_values) / sizeof(pool_values[0]));
    PurePqIndex *index = NULL;
    RetrievalPqResidualIndex *residual_index = NULL;
    RetrievalPqResidualQuery *residual_query = NULL;
    PurePqSharedRuntime shared;
    PurePqPrepareRequest prepare;
    uint8_t layout_fingerprint[32];
    FILE *result_file = NULL;
    FILE *refine_trace = NULL;
    RetrievalFinalRefineContext *refinement = NULL;
    const int refine_budget = FINAL_REFINE_MODE ? FINAL_REFINE_CANDIDATES : 0;
    int num_queries = RETRIEVAL_RPC_DEMO_QUERY_LIMIT;
    int result = -1;
    int entry_index;
    int pool_index;
    const char *window_env = getenv("SQMD_MAX_INFLIGHT_SOURCES");
    int source_window = 1;
    if (window_env && window_env[0]) {
        if (strcmp(window_env, "1") == 0) source_window = 1;
        else if (strcmp(window_env, "2") == 0) source_window = 2;
        else {
            printf("  ERROR: SQMD_MAX_INFLIGHT_SOURCES must be 1 or 2\n");
            return -1;
        }
    }

    memset(&shared, 0, sizeof(shared));
    shared.control_fd = -1;
    shared.lut_fd = -1;
    shared.domain_id = -1;
    if (num_queries <= 0 || num_queries > DATA_N_QUERIES) {
        num_queries = DATA_N_QUERIES;
    }
    printf("\n=== Pure-PQ SQMD Search (Phase 9) ===\n");
    printf("dataset=%s N=%d dim=%d bits=%d M=%d Ks=%d compression=%dx "
           "lanes=%d correction=%s residual_bits=%d artifacts=%s result=%s\n",
           PURE_PQ_DATASET_NAME, DATA_N_BASE, DATA_DIM, PURE_PQ_CODE_BITS,
           PURE_PQ_M, PURE_PQ_KS, PURE_PQ_COMPRESSION_X,
           PURE_PQ_SQMD_LANES, PURE_PQ_CORRECTION_TAG,
           PURE_PQ_CORRECTION_MODE == 1 ? PURE_PQ_RESIDUAL_BITS : 0,
           PURE_PQ_ARTIFACT_PREFIX,
           PURE_PQ_DEVICE_RESULT_PATH);
    fflush(stdout);

    if ((mkdir(PURE_PQ_DEVICE_RESULT_DIR, 0755) != 0 && errno != EEXIST) ||
        (result_file = fopen(PURE_PQ_DEVICE_RESULT_PATH, "w")) == NULL ||
        fprintf(result_file,
                "dataset,layout,code_bits,ks,m,code_bytes,compression_x,"
                "entry_points,pool_size,queries,recall_at_1,recall_at_10,"
                "recall_at_100,wall_us,work_items,dsp_adc_us,dsp_wait_polls,"
                "correction_mode,residual_bits,corrected_candidates,"
                "residual_cache_hits,residual_cache_misses,"
                "residual_cache_errors,"
                "artifact_prefix,window,source_jobs,source_retired,source_high_water,"
                "top_k,refine_mode,refine_factor,refine_budget,refine_candidates,refine_us,"
                "refine_blocks,refine_bytes,refine_hits,search_pool_fingerprint,total_wall_us\n") < 0) {
        printf("  ERROR: cannot create result CSV %s: %s\n",
               PURE_PQ_DEVICE_RESULT_PATH, strerror(errno));
        goto cleanup;
    }

    if (pure_pq_parse_fingerprint_hex(
            DATA_LAYOUT_FINGERPRINT_HEX, layout_fingerprint) != 0) {
        goto cleanup;
    }
    memset(&prepare, 0, sizeof(prepare));
    prepare.mode = (PurePqPrepareMode)PURE_PQ_PREPARE_MODE;
    prepare.base_fvecs_path = DATA_BASE_PATH;
    prepare.order_map_path = DATA_ORDER_MAP_PATH;
    prepare.artifact_prefix = PURE_PQ_ARTIFACT_PREFIX;
    prepare.expected_num_vectors = DATA_N_BASE;
    prepare.expected_dim = DATA_DIM;
    prepare.m = PURE_PQ_M;
    prepare.ks = PURE_PQ_KS;
    prepare.train_max = PURE_PQ_TRAIN_MAX;
    prepare.kmeans_iters = PURE_PQ_KMEANS_ITERS;
    prepare.encode_chunk_vectors = PURE_PQ_ENCODE_CHUNK_VECTORS;
    prepare.residual_bits = PURE_PQ_CORRECTION_MODE == 1
                                ? PURE_PQ_RESIDUAL_BITS : 0;
    prepare.memory_budget_bytes = PURE_PQ_MEMORY_BUDGET_BYTES;
    prepare.layout_id = DATA_LAYOUT_NAME;
    prepare.expected_layout_fingerprint = layout_fingerprint;
    if (pure_pq_index_prepare(&prepare, &index) != 0) {
        printf("  ERROR: Pure-PQ prepare failed (mode=%d prefix=%s)\n",
               PURE_PQ_PREPARE_MODE, PURE_PQ_ARTIFACT_PREFIX);
        goto cleanup;
    }
    if (demo_resolve_graph_index(handle, domain_id,
                                 &g_graph, &g_graph_uses_rpcmem,
                                 &g_nsg_entry_point) != 0) {
        goto cleanup;
    }
    if (load_fvecs_n(DATA_QUERY_PATH, num_queries, DATA_DIM, &g_queries) != 0 ||
        load_ivecs_n(DATA_GT_PATH, num_queries, DEMO_GT_K, &g_gt) != 0) {
        goto cleanup;
    }
    if (pure_pq_index_map_dsp(index, handle, domain_id) != 0) {
        printf("  ERROR: Pure-PQ DSP mapping/load failed\n");
        goto cleanup;
    }
    if (PURE_PQ_CORRECTION_MODE == 1 &&
        (pure_pq_index_open_residual(
             index, PURE_PQ_ARTIFACT_PREFIX, PURE_PQ_RESIDUAL_BITS,
             PURE_PQ_RESIDUAL_CACHE_SLOTS, &residual_index) != 0 ||
         retrieval_pq_residual_query_create(
             residual_index, &residual_query) != 0)) {
        printf("  ERROR: Pure-PQ ONLINE residual open failed "
               "(prefix=%s bits=%d)\n",
               PURE_PQ_ARTIFACT_PREFIX, PURE_PQ_RESIDUAL_BITS);
        goto cleanup;
    }
    if (FINAL_REFINE_MODE) {
        RetrievalFinalRefineConfig config = {0};
        const unsigned char *digest = (const unsigned char *)FINAL_REFINE_SHA256_BYTES;
        config.graph_to_original=rover_graph_to_original();
        config.mode = FINAL_REFINE_MODE;
        config.num_vectors = DATA_N_BASE; config.dim = DATA_DIM;
        config.capacity = FINAL_REFINE_CANDIDATES; config.cache_slots = FINAL_REFINE_CACHE_SLOTS;
        config.decode = pure_pq_refine_decode; config.provider = residual_query;
        config.artifact_path = FINAL_REFINE_PATH; config.artifact_sha256 = digest;
        config.base_paths[0] = rover_config.pq_codes;
        config.base_paths[1] = rover_config.pq_residual;
        config.base_paths[2] = rover_config.pq_qparams;
        if (!residual_query ||
            retrieval_final_refine_open_pq(&config, rover_config.pq_book, &refinement)) {
            printf("ERROR: PQ final refinement identity/open failed\n"); goto cleanup;
        }
    }
    if (PURE_PQ_REFINE_TRACE) {
        refine_trace = fopen(FINAL_REFINE_TRACE_PATH, "w");
        if (!refine_trace) goto cleanup;
    }
    if (pure_pq_shared_init(&shared, domain_id,
                            pure_pq_index_metadata(index)) != 0) {
        printf("  ERROR: Pure-PQ SQMD shared buffers failed\n");
        goto cleanup;
    }

    for (entry_index = 0; entry_index < entry_count; entry_index++) {
        for (pool_index = 0; pool_index < pool_count; pool_index++) {
            RetrievalDemoConfig config;
            PurePqSessionArgs session;
            pthread_t thread;
            int *out_ids = NULL;
            float *out_distances = NULL;
            int recall1 = 0;
            int recall10 = 0;
            int recall100 = 0;
            uint64_t work_items = 0u;
            uint64_t search_fingerprint = UINT64_C(14695981039346656037);
            uint64_t refine_candidates = 0, refine_us = 0, refine_blocks = 0, refine_bytes = 0, refine_hits = 0;
            unsigned long long total_wall_us;
            uint64_t source_jobs = 0u, source_retired = 0u;
            int source_high_water = 0;
            RetrievalPqResidualStats residual_totals;
            unsigned long long start;
            int query_index;
            int session_opened = 0;
            int query_failed = 0;
            double recall_at_1;
            double recall_at_10;
            double recall_at_100;
            unsigned long long wall_us;
            unsigned long long mean_work_items;
            unsigned long long mean_dsp_adc_us;
            unsigned long long mean_dsp_wait_polls;

            memset(&residual_totals, 0, sizeof(residual_totals));

            memset(&config, 0, sizeof(config));
            config.entry_points = entry_values[entry_index];
            config.expand_count = rover_config.expand_count;
            config.pool_size = pool_values[pool_index];
            config.top_k = SEARCH_TOP_K;

            out_ids = (int *)malloc((size_t)config.pool_size * sizeof(*out_ids));
            out_distances = (float *)malloc((size_t)config.pool_size * sizeof(*out_distances));
            retrieval_final_refine_reset(refinement);
            if (!out_ids || !out_distances ||
                pure_pq_session_open(handle, &shared, PURE_PQ_SQMD_LANES,
                                     &thread, &session) != 0) {
                free(out_ids);
                free(out_distances);
                goto cleanup;
            }
            session_opened = 1;
            start = now_us();
            for (query_index = 0; query_index < num_queries; query_index++) {
                const float *query = g_queries +
                    (size_t)query_index * (size_t)DATA_DIM;
                const int32_t *truth = g_gt +
                    (size_t)query_index * (size_t)DEMO_GT_K;
                RetrievalPqResidualStats query_residual_stats;
                PurePqSourcePipeline query_pipeline;
                int out_capacity = config.pool_size;
                int count = pure_pq_run_one_query(
                    index, g_graph, DATA_GRAPH_K, g_nsg_entry_point,
                    query, &config, &shared, &session, residual_query,
                    out_ids, out_distances, out_capacity, &work_items,
                    &query_residual_stats, source_window, &query_pipeline);
                if (count < 0) {
                    shared.control->control_state =
                        PURE_PQ_SQMD_CONTROL_ABORT;
                    DB_WRITE_BARRIER();
                    query_failed = 1;
                    break;
                }
                search_fingerprint = pure_pq_result_fingerprint(search_fingerprint, out_ids, out_distances, count);
                if (refine_trace) {
                    int i, d;
                    float decoded[DATA_DIM];
                    fprintf(refine_trace, "Q %d %d %d %d\n", config.pool_size, query_index, count, refine_budget);
                    for (i = 0; i < count; ++i)
                        fprintf(refine_trace, "C %d %a\n", out_ids[i], (double)out_distances[i]);
                    for (i = 0; i < count && i < (refine_budget ? refine_budget : SEARCH_TOP_K); ++i) {
                        if (pure_pq_refine_decode(residual_query, out_ids[i], decoded)) { query_failed = 1; break; }
                        fprintf(refine_trace, "V %d", out_ids[i]);
                        for (d = 0; d < DATA_DIM; ++d) fprintf(refine_trace, " %a", (double)decoded[d]);
                        fputc('\n', refine_trace);
                    }
                    if (query_failed || ferror(refine_trace)) { query_failed = 1; break; }
                }
                if (refinement) {
                    RetrievalFinalRefineStats stats;
                    unsigned long long refine_start = now_us();
                    int selected = count < refine_budget ? count : refine_budget;
                    count = retrieval_final_refine(refinement, query, out_ids, out_distances,
                                                     selected, SEARCH_TOP_K, &stats);
                    refine_us += now_us() - refine_start;
                    if (count < 0) { query_failed = 1; break; }
                    refine_candidates += stats.candidates; refine_blocks += stats.code_blocks;
                    refine_bytes += stats.code_bytes; refine_hits += stats.cache_hits;
                } else if (count > SEARCH_TOP_K) count = SEARCH_TOP_K;
                if (count != SEARCH_TOP_K) { query_failed = 1; break; }
                if (refine_trace) {
                    int i;
                    for (i = 0; i < count; ++i)
                        fprintf(refine_trace, "O %d %a\n", out_ids[i], (double)out_distances[i]);
                }
                source_jobs += query_pipeline.jobs;
                source_retired += query_pipeline.retired;
                if (query_pipeline.high_water > source_high_water)
                    source_high_water = query_pipeline.high_water;
                residual_totals.corrected_candidates +=
                    query_residual_stats.corrected_candidates;
                residual_totals.cache_hits += query_residual_stats.cache_hits;
                residual_totals.cache_misses +=
                    query_residual_stats.cache_misses;
                residual_totals.cache_errors +=
                    query_residual_stats.cache_errors;
                if (rover_write_result(query_index,out_ids,out_distances,
                        count < config.top_k ? count : config.top_k)) {query_failed=1;break;}
                recall1 += demo_recall_hits_at_k(out_ids, count,
                                                 truth, DEMO_GT_K, 1);
                recall10 += demo_recall_hits_at_k(out_ids, count,
                                                  truth, DEMO_GT_K, 10);
                recall100 += demo_recall_hits_at_k(out_ids, count,
                                                   truth, DEMO_GT_K, 100);
            }
            if (session_opened &&
                pure_pq_session_close(&shared, thread, &session) != 0) {
                free(out_ids);
                free(out_distances);
                goto cleanup;
            }
            if (query_failed) {
                free(out_ids);
                free(out_distances);
                goto cleanup;
            }
            recall_at_1 = 100.0 * (double)recall1 / (double)num_queries;
            recall_at_10 = 100.0 * (double)recall10 /
                           ((double)num_queries * 10.0);
            recall_at_100 = 100.0 * (double)recall100 /
                            ((double)num_queries * 100.0);
            total_wall_us = now_us() - start;
            wall_us = total_wall_us / (unsigned long long)num_queries;
            mean_work_items = (unsigned long long)(
                work_items / (uint64_t)num_queries);
            mean_dsp_adc_us = (unsigned long long)(
                shared.control->stats.dsp_compute_us /
                (uint64_t)num_queries);
            mean_dsp_wait_polls = (unsigned long long)(
                shared.control->stats.dsp_wait_polls /
                (uint64_t)num_queries);
            printf("  e=%3d p=%4d R@1=%5.2f R@10=%5.2f R@100=%5.2f "
                   "wall=%llu us work=%llu dsp_adc=%llu us polls=%llu "
                   "corr=%llu cache=%llu/%llu err=%llu\n",
                   config.entry_points, config.pool_size,
                   recall_at_1, recall_at_10, recall_at_100,
                   wall_us, mean_work_items, mean_dsp_adc_us,
                   mean_dsp_wait_polls,
                   (unsigned long long)(residual_totals.corrected_candidates /
                                        (uint64_t)num_queries),
                   (unsigned long long)residual_totals.cache_hits,
                   (unsigned long long)residual_totals.cache_misses,
                   (unsigned long long)residual_totals.cache_errors);
            if (fprintf(result_file,
                        "%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%.8f,%.8f,%.8f,"
                        "%llu,%llu,%llu,%llu,%s,%d,%llu,%llu,%llu,%llu,%s,%d,%llu,%llu,%d,"
                        "%d,%d,%d,%d,%llu,%llu,%llu,%llu,%llu,%016llx,%llu\n",
                        PURE_PQ_DATASET_NAME, DATA_LAYOUT_NAME,
                        PURE_PQ_CODE_BITS, PURE_PQ_KS, PURE_PQ_M,
                        PURE_PQ_CODE_BYTES, PURE_PQ_COMPRESSION_X,
                        config.entry_points, config.pool_size, num_queries,
                        recall_at_1, recall_at_10, recall_at_100,
                        wall_us, mean_work_items, mean_dsp_adc_us,
                        mean_dsp_wait_polls, PURE_PQ_CORRECTION_TAG,
                        PURE_PQ_CORRECTION_MODE == 1
                            ? PURE_PQ_RESIDUAL_BITS : 0,
                        (unsigned long long)
                            residual_totals.corrected_candidates,
                        (unsigned long long)residual_totals.cache_hits,
                        (unsigned long long)residual_totals.cache_misses,
                        (unsigned long long)residual_totals.cache_errors,
                        PURE_PQ_ARTIFACT_PREFIX, source_window,
                        (unsigned long long)source_jobs,
                        (unsigned long long)source_retired, source_high_water,
                        SEARCH_TOP_K, FINAL_REFINE_MODE, FINAL_REFINE_FACTOR, refine_budget,
                        (unsigned long long)refine_candidates, (unsigned long long)refine_us,
                        (unsigned long long)refine_blocks, (unsigned long long)refine_bytes,
                        (unsigned long long)refine_hits, (unsigned long long)search_fingerprint, total_wall_us) < 0 ||
                fflush(result_file) != 0) {
                free(out_ids);
                free(out_distances);
                goto cleanup;
            }
            printf("PQ_PIPELINE window=%d jobs=%llu retired=%llu high_water=%d\n",
                   source_window, (unsigned long long)source_jobs,
                   (unsigned long long)source_retired, source_high_water);
            printf("PQ_REFINE K=%d mode=%d factor=%d budget=%d candidates=%llu us=%llu search=%016llx\n",
                   SEARCH_TOP_K, FINAL_REFINE_MODE, FINAL_REFINE_FACTOR, refine_budget,
                   (unsigned long long)refine_candidates, (unsigned long long)refine_us,
                   (unsigned long long)search_fingerprint);
            fflush(stdout);
            free(out_ids);
            free(out_distances);
        }
    }
    result = AEE_SUCCESS;

cleanup:
    if (result_file && fclose(result_file) != 0) {
        result = -1;
    }
    if (refine_trace && fclose(refine_trace) != 0) result = -1;
    retrieval_final_refine_close(refinement);
    retrieval_pq_residual_query_destroy(residual_query);
    retrieval_pq_residual_index_close(residual_index);
    pure_pq_shared_destroy(&shared);
    if (index) {
        pure_pq_index_unmap_dsp(index);
        pure_pq_index_release(index);
    }
    demo_free_loaded_data();
    return result;
}
