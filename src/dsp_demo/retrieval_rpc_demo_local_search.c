static int demo_local_graph_row_count(const int32_t *neighbors,
                                      int slot_count,
                                      int num_vectors) {
    int count = slot_count;
    int index;

    if (!neighbors || slot_count < 0 || num_vectors <= 0)
        return -1;
    for (index = 0; index < slot_count; index++) {
        int neighbor_id = neighbors[index];

        if (neighbor_id == -1) {
            int suffix;

            count = index;
            for (suffix = index + 1; suffix < slot_count; suffix++) {
                if (neighbors[suffix] != -1)
                    return -1;
            }
            break;
        }
        if (neighbor_id < 0 || neighbor_id >= num_vectors)
            return -1;
    }
    return count;
}

static int demo_search_local_core(const RetrievalDemoEncodedDataset *dataset,
                                  const int32_t *graph,
                                  int graph_degree,
                                  const float *query,
                                  const RetrievalDemoConfig *config,
                                  unsigned int seed,
                                  int *result_ids,
                                  float *result_dists) {
    uint32_t *visited = NULL;
    DemoCandidate *pool = NULL;
    DemoFrontierEntry *frontier = NULL;
    ResidualBlockCache *local_cache = NULL;
    int frontier_size = 0;
    uint8_t query_l1[RETRIEVAL_ASYNC_DEMO_MAX_DIM];
    uint8_t batch_codes[RETRIEVAL_ASYNC_DEMO_MAX_BATCH * RETRIEVAL_ASYNC_DEMO_MAX_DIM];
    int batch_ids[RETRIEVAL_ASYNC_DEMO_MAX_BATCH];
    float batch_result_dists[RETRIEVAL_ASYNC_DEMO_MAX_BATCH];
    int pool_size = 0;
    int expand_limit;
    int result_count;
    int i;

    if (!dataset || !graph || !query || !config || !result_ids || !result_dists ||
        dataset->num_vectors <= 0 ||
        dataset->dim <= 0 || dataset->dim > RETRIEVAL_ASYNC_DEMO_MAX_DIM ||
        graph_degree <= 0 ||
        config->entry_points <= 0 ||
        config->pool_size <= 0 ||
        config->top_k <= 0 ||
        config->rerank_factor <= 0) {
        return -1;
    }

    expand_limit = config->expand_count;
    if (expand_limit > graph_degree) {
        expand_limit = graph_degree;
    }
    if (expand_limit <= 0 || expand_limit > RETRIEVAL_ASYNC_DEMO_MAX_BATCH) {
        return -1;
    }

    visited = (uint32_t *)calloc(VISITED_BITMAP_WORDS(dataset->num_vectors),
                                  sizeof(uint32_t));
    pool = (DemoCandidate *)malloc((size_t)config->pool_size * sizeof(DemoCandidate));
    {
        size_t frontier_cap = (size_t)config->entry_points +
                              (size_t)config->pool_size * (size_t)expand_limit;
        frontier = (DemoFrontierEntry *)malloc(frontier_cap *
                                               sizeof(DemoFrontierEntry));
    }
    if (!visited || !pool || !frontier) {
        free(visited);
        free(pool);
        free(frontier);
        return -1;
    }

    if (demo_quantize_query_l1(query, dataset, query_l1) != 0) {
        free(visited);
        free(pool);
        free(frontier);
        return -1;
    }

    {
        const uint8_t *base = (g_residual_fd < 0) ? dataset->residual_codes : NULL;
        local_cache = residual_cache_create(
            g_residual_fd, g_residual_vpb, g_residual_total_blocks,
            RESIDUAL_CACHE_SLOTS_LOCAL, base,
            dataset->residual_code_bytes,
            &g_residual_stats);
    }

    for (i = 0; i < config->entry_points && i < dataset->num_vectors; i++) {
        int entry_id;
        float distance;

        seed = seed * 1103515245u + 12345u;
        entry_id = (int)((seed / 65536u) % (unsigned int)dataset->num_vectors);

        if (VISITED_TEST(visited, entry_id)) {
            continue;
        }
        VISITED_SET(visited, entry_id);

        distance = demo_compute_l1_distance(
            dataset,
            query_l1,
            demo_l1_code_ptr(dataset, entry_id));
        if (demo_candidate_insert(pool,
                                  &pool_size,
                                  config->pool_size,
                                  entry_id,
                                  distance))
            demo_frontier_push(frontier, &frontier_size, entry_id, distance);
    }


    while (frontier_size > 0) {
        int current_id;
        const int32_t *neighbors;
        int valid_neighbor_count;
        int num_candidates = 0;
        int j;


        if (pool_size >= config->pool_size &&
            frontier[0].distance > pool[0].distance) {
            demo_frontier_pop(frontier, &frontier_size);
            continue;
        }

        current_id = demo_frontier_pop(frontier, &frontier_size);
        if (current_id < 0)
            break;

        neighbors = graph + (size_t)current_id * (size_t)graph_degree;
        valid_neighbor_count = demo_local_graph_row_count(
            neighbors, expand_limit, dataset->num_vectors);
        if (valid_neighbor_count < 0) {
            residual_cache_destroy(local_cache);
            free(visited);
            free(pool);
            free(frontier);
            return -1;
        }

        for (j = 0; j < valid_neighbor_count; j++) {
            int neighbor_id = neighbors[j];

            if (VISITED_TEST(visited, neighbor_id)) {
                continue;
            }
            VISITED_SET(visited, neighbor_id);
            batch_ids[num_candidates] = neighbor_id;
            memcpy(batch_codes + (size_t)num_candidates *
                   (size_t)dataset->l1_code_bytes,
                   demo_l1_code_ptr(dataset, neighbor_id),
                   (size_t)dataset->l1_code_bytes);
            num_candidates++;
        }

        if (num_candidates == 0)
            continue;

        if (demo_compute_l1_distances_batch(dataset,
                                            query_l1,
                                            batch_codes,
                                            num_candidates,
                                            batch_result_dists) != 0) {
            residual_cache_destroy(local_cache);
            free(visited);
            free(pool);
            free(frontier);
            return -1;
        }

        for (i = 0; i < num_candidates; i++) {
            if (demo_candidate_insert(pool,
                                      &pool_size,
                                      config->pool_size,
                                      batch_ids[i],
                                      batch_result_dists[i]))
                demo_frontier_push(frontier, &frontier_size,
                                   batch_ids[i], batch_result_dists[i]);
        }
    }

    demo_heap_sort(pool, pool_size);

    result_count = config->top_k * config->rerank_factor;
    if (result_count > pool_size) {
        result_count = pool_size;
    }
    for (i = 1; i < result_count; i++) {
        DemoCandidate tmp = pool[i];
        int j = i - 1;

        while (j >= 0 && pool[j].id > tmp.id) {
            pool[j + 1] = pool[j];
            j--;
        }
        pool[j + 1] = tmp;
    }
    for (i = 0; i < result_count; i++) {
        pool[i].distance = retrieval_demo_exact_distance_residual_cached(
                               query, dataset, pool[i].id, local_cache);
    }
    for (i = 1; i < result_count; i++) {
        DemoCandidate tmp = pool[i];
        int j = i - 1;

        while (j >= 0 && pool[j].distance > tmp.distance) {
            pool[j + 1] = pool[j];
            j--;
        }
        pool[j + 1] = tmp;
    }

    result_count = config->top_k;
    if (result_count > pool_size) {
        result_count = pool_size;
    }
    for (i = 0; i < result_count; i++) {
        result_ids[i] = pool[i].id;
        result_dists[i] = pool[i].distance;
    }

    residual_cache_flush_stats(local_cache, &g_residual_stats,
                               RESIDUAL_CACHE_ROLE_LOCAL);
    residual_cache_destroy(local_cache);
    free(frontier);
    free(visited);
    free(pool);
    return result_count;
}

int retrieval_demo_search_local(const RetrievalDemoEncodedDataset *dataset,
                                const int32_t *graph,
                                int graph_degree,
                                const float *query,
                                const RetrievalDemoConfig *config,
                                unsigned int seed,
                                int *result_ids,
                                float *result_dists) {
    return demo_search_local_core(dataset,
                                  graph,
                                  graph_degree,
                                  query,
                                  config,
                                  seed,
                                  result_ids,
                                  result_dists);
}
