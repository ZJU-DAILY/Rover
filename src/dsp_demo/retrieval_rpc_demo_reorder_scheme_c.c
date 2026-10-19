static double compute_or_g(const int32_t *graph, int N, int K,
                           const int32_t *block_of, int vpb) {
    double total = 0.0;
    int u;
    for (u = 0; u < N; u++) {
        int b = block_of[u];
        int cnt = 0;
        const int32_t *nb = graph + (size_t)u * (size_t)K;
        int j;
        for (j = 0; j < K; j++) {
            int v = nb[j];
            if (v >= 0 && v < N && v != u && block_of[v] == b)
                cnt++;
        }
        total += (double)cnt / (double)(vpb - 1);
    }
    return total / (double)N;
}

static int reorder_scheme_c(const int32_t *graph, int N, int K, int vpb,
                            int32_t *reorder_map, int32_t *reverse_map) {
    int *assigned = NULL;
    int *block_of = NULL;
    int *csize = NULL;
    int *freq = NULL;
    int *touched = NULL;
    int32_t *prev_block = NULL;
    int32_t *best_rmap = NULL;
    int32_t *best_rev = NULL;
    int *best_block_of = NULL;
    int M;
    int new_id, block_start;
    int i, j, u, iter;
    double t0, t1, t_bnp;
    double or_prev, or_new, or_best;
    int rc = -1;

    M = (N + vpb - 1) / vpb;

    assigned       = (int *)calloc((size_t)N, sizeof(int));
    block_of       = (int *)malloc((size_t)N * sizeof(int));
    csize          = (int *)calloc((size_t)M, sizeof(int));
    freq           = (int *)calloc((size_t)M, sizeof(int));
    touched        = (int *)malloc((size_t)M * sizeof(int));
    prev_block     = (int32_t *)malloc((size_t)N * sizeof(int32_t));
    best_rmap      = (int32_t *)malloc((size_t)N * sizeof(int32_t));
    best_rev       = (int32_t *)malloc((size_t)N * sizeof(int32_t));
    best_block_of  = (int *)malloc((size_t)N * sizeof(int));
    if (!assigned || !block_of || !csize || !freq || !touched ||
        !prev_block || !best_rmap || !best_rev || !best_block_of)
        goto fail;

    printf("[reorder C] M=%d vpb=%d\n", M, vpb);
    t0 = reorder_now_ms();


    new_id = 0;
    block_start = 0;
    for (u = 0; u < N; u++) {
        if (assigned[u]) continue;

        reorder_map[u] = new_id;
        reverse_map[new_id] = u;
        block_of[u] = new_id / vpb;
        new_id++;
        assigned[u] = 1;

        {
            const int32_t *nb = graph + (size_t)u * (size_t)K;
            for (j = 0; j < K; j++) {
                int v = nb[j];
                if (v >= 0 && v < N && !assigned[v] &&
                    (new_id - block_start) < vpb) {
                    reorder_map[v] = new_id;
                    reverse_map[new_id] = v;
                    block_of[v] = new_id / vpb;
                    new_id++;
                    assigned[v] = 1;
                }
            }
        }

        if ((new_id - block_start) >= vpb)
            block_start = new_id;
    }

    t1 = reorder_now_ms();
    t_bnp = t1 - t0;
    printf("[reorder C]   BNP: %.1f ms\n", t_bnp);


    or_prev = compute_or_g(graph, N, K, block_of, vpb);
    or_best = or_prev;
    memcpy(best_rmap, reorder_map, (size_t)N * sizeof(int32_t));
    memcpy(best_rev, reverse_map, (size_t)N * sizeof(int32_t));
    memcpy(best_block_of, block_of, (size_t)N * sizeof(int));
    printf("[reorder C]   BNP OR(G)= %.4f\n", or_prev);


    for (iter = 0; iter < BNF_ITERS; iter++) {
        double t_iter_start;


        memcpy(prev_block, block_of, (size_t)N * sizeof(int32_t));


        memset(csize, 0, (size_t)M * sizeof(int));

        t_iter_start = reorder_now_ms();

        for (u = 0; u < N; u++) {
            int touched_count = 0;
            int placed;


            {
                const int32_t *nb = graph + (size_t)u * (size_t)K;
                for (j = 0; j < K; j++) {
                    int v = nb[j];
                    if (v >= 0 && v < N && v != u) {
                        int b = prev_block[v];
                        if (freq[b] == 0)
                            touched[touched_count++] = b;
                        freq[b]++;
                    }
                }
            }


            {
                int t, k;
                for (t = 1; t < touched_count; t++) {
                    int key = touched[t];
                    int key_f = freq[key];
                    k = t - 1;
                    while (k >= 0 && freq[touched[k]] < key_f) {
                        touched[k + 1] = touched[k];
                        k--;
                    }
                    touched[k + 1] = key;
                }
            }


            placed = 0;
            for (i = 0; i < touched_count; i++) {
                int b = touched[i];
                if (csize[b] < vpb) {
                    int slot = b * vpb + csize[b];
                    reorder_map[u] = slot;
                    reverse_map[slot] = u;
                    block_of[u] = b;
                    csize[b]++;
                    placed = 1;
                    break;
                }
            }


            if (!placed) {
                for (i = 0; i < M; i++) {
                    if (csize[i] < vpb) {
                        int slot = i * vpb + csize[i];
                        reorder_map[u] = slot;
                        reverse_map[slot] = u;
                        block_of[u] = i;
                        csize[i]++;
                        placed = 1;
                        break;
                    }
                }
            }


            for (i = 0; i < touched_count; i++)
                freq[touched[i]] = 0;
        }

        or_new = compute_or_g(graph, N, K, block_of, vpb);
        printf("[reorder C]   BNF iter %d: OR(G)=%.4f -> %.4f delta=%.4f  %.1f ms\n",
               iter, or_prev, or_new, or_new - or_prev,
               reorder_now_ms() - t_iter_start);


        if (or_new > or_best) {
            or_best = or_new;
            memcpy(best_rmap, reorder_map, (size_t)N * sizeof(int32_t));
            memcpy(best_rev, reverse_map, (size_t)N * sizeof(int32_t));
            memcpy(best_block_of, block_of, (size_t)N * sizeof(int));
        }

        if (or_new - or_prev < BNF_OR_GAIN_TAU) {
            printf("[reorder C]   BNF early stop (gain < %.4f)\n", BNF_OR_GAIN_TAU);
            break;
        }
        or_prev = or_new;
    }


    memcpy(reorder_map, best_rmap, (size_t)N * sizeof(int32_t));
    memcpy(reverse_map, best_rev, (size_t)N * sizeof(int32_t));
    printf("[reorder C]   final OR(G)= %.4f\n", or_best);

    printf("[reorder C] total: %.1f s\n", (reorder_now_ms() - t0) / 1000.0);

    rc = 0;
fail:
    free(assigned);
    free(block_of);
    free(csize);
    free(freq);
    free(touched);
    free(prev_block);
    free(best_rmap);
    free(best_rev);
    free(best_block_of);
    return rc;
}
