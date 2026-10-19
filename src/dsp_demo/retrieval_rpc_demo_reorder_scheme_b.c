static int reorder_b_pick_Mc(int N, int vpb) {
    int fine = N / vpb;
    int target;
    int lo, hi, best, best_diff;
    int d, diff;

    if (KMEANS_L1_NUM_CLUSTERS_OVERRIDE > 0) {
        int v = KMEANS_L1_NUM_CLUSTERS_OVERRIDE;
        if (v < 2) v = 2;
        if (v > fine / 2) v = fine / 2;
        return v;
    }
    target = (int)sqrt((double)fine);
    if (target < 2) target = 2;
    lo = target / 2; if (lo < 2) lo = 2;
    hi = target * 2; if (hi > fine / 2) hi = fine / 2;
    best = target;
    best_diff = -1;
    for (d = lo; d <= hi; d++) {
        if (fine % d != 0) continue;
        diff = d > target ? d - target : target - d;
        if (best_diff < 0 || diff < best_diff) {
            best = d; best_diff = diff;
        }
    }
    if (best_diff < 0) {

        best = target;
    }
    return best;
}

static int reorder_scheme_b(const int32_t *graph, int N, int K, int vpb,
                            int32_t *reorder_map, int32_t *reverse_map) {
    int *in_degree = NULL;
    int *deg_order = NULL;
    int *label = NULL;
    int *csize = NULL;
    int *seeds = NULL;
    int *queue = NULL;
    int *score = NULL;
    int *touched = NULL;
    int touched_count = 0;
    int *cand = NULL;
    int *delta = NULL;
    int *delta_order = NULL;
    int *pair_count = NULL;
    int *cluster_offset = NULL;
    int *member_buf = NULL;
    int *assigned = NULL;
    int *block_buf = NULL;
    int *cluster_emit_order = NULL;
    int *cluster_total_score = NULL;

    int M_c, fine, L_c, cap_soft;
    int rc = -1;
    int i, j, c;
    unsigned int rng;
    double t0, t1, t_total0;

    t_total0 = reorder_now_ms();


    t0 = reorder_now_ms();
    in_degree = (int *)calloc((size_t)N, sizeof(int));
    deg_order = (int *)malloc((size_t)N * sizeof(int));
    if (!in_degree || !deg_order) goto fail;
    compute_in_degree(graph, N, K, in_degree);
    if (sort_by_degree_desc(N, K, in_degree, deg_order) != 0) goto fail;

    M_c  = reorder_b_pick_Mc(N, vpb);
    fine = N / vpb;
    L_c  = (fine / M_c) * vpb;
    cap_soft = (int)(((long)L_c * (100 + KMEANS_L1_SLACK_PCT) + 99) / 100);
    cap_soft = ((cap_soft + vpb - 1) / vpb) * vpb;

    label = (int *)malloc((size_t)N * sizeof(int));
    csize = (int *)calloc((size_t)M_c, sizeof(int));
    seeds = (int *)malloc((size_t)M_c * sizeof(int));

    score   = (int *)calloc((size_t)(N > M_c ? N : M_c), sizeof(int));
    touched = (int *)malloc((size_t)(N > M_c ? N : M_c) * sizeof(int));
    if (!label || !csize || !seeds || !score || !touched) goto fail;
    for (i = 0; i < N; i++) label[i] = -1;


    {
        int pool_size = N / 2;
        if (pool_size < M_c) pool_size = N;
        rng = (unsigned int)KMEANS_RNG_SEED;

        {
            int *pool = (int *)malloc((size_t)pool_size * sizeof(int));
            if (!pool) goto fail;
            for (i = 0; i < pool_size; i++) pool[i] = deg_order[i];
            for (i = 0; i < M_c; i++) {
                unsigned int r;
                int swap_idx;
                rng = rng * 1103515245u + 12345u;
                r = (rng >> 8);
                swap_idx = i + (int)(r % (unsigned int)(pool_size - i));

                {
                    int tmp = pool[i]; pool[i] = pool[swap_idx]; pool[swap_idx] = tmp;
                }
                seeds[i] = pool[i];
                label[seeds[i]] = i;
                csize[i] = 1;
            }
            free(pool);
        }
    }
    free(deg_order); deg_order = NULL;
    t1 = reorder_now_ms();
    printf("[reorder B] M_c=%d L_c=%d cap_soft=%d\n", M_c, L_c, cap_soft);
    printf("[reorder B]   in_degree+seeds: %.0f ms\n", t1 - t0);


    t0 = reorder_now_ms();
    queue = (int *)malloc((size_t)N * sizeof(int));
    if (!queue) goto fail;
    {
        int qh = 0, qt = 0;
        int unlabeled;
        for (i = 0; i < M_c; i++) queue[qt++] = seeds[i];
        while (qh < qt) {
            int v = queue[qh++];
            int lv = label[v];
            const int32_t *nb = graph + (size_t)v * (size_t)K;
            for (j = 0; j < K; j++) {
                int u = nb[j];
                if (u < 0 || u >= N) continue;
                if (label[u] != -1) continue;
                if (csize[lv] >= cap_soft) break;
                label[u] = lv;
                csize[lv]++;
                queue[qt++] = u;
            }
        }

        unlabeled = 0;
        for (i = 0; i < N; i++) {
            if (label[i] != -1) continue;
            unlabeled++;
            touched_count = 0;
            {
                const int32_t *nb = graph + (size_t)i * (size_t)K;
                for (j = 0; j < K; j++) {
                    int u = nb[j];
                    int lu;
                    if (u < 0 || u >= N) continue;
                    lu = label[u];
                    if (lu < 0) continue;
                    if (score[lu] == 0) touched[touched_count++] = lu;
                    score[lu]++;
                }
            }
            {
                int best_c = -1, best_s = -1;
                int t;
                for (t = 0; t < touched_count; t++) {
                    int cc = touched[t];
                    if (csize[cc] >= cap_soft) continue;
                    if (score[cc] > best_s ||
                        (score[cc] == best_s && cc < best_c)) {
                        best_s = score[cc]; best_c = cc;
                    }
                }
                if (best_c < 0) {

                    int min_s = INT_MAX, min_c = 0;
                    for (c = 0; c < M_c; c++) {
                        if (csize[c] < min_s) { min_s = csize[c]; min_c = c; }
                    }
                    best_c = min_c;
                }
                label[i] = best_c;
                csize[best_c]++;
            }
            for (j = 0; j < touched_count; j++) score[touched[j]] = 0;
        }
        t1 = reorder_now_ms();
        printf("[reorder B]   Round0 BFS:      %.0f ms (unlabeled=%d)\n",
               t1 - t0, unlabeled);
    }
    free(queue); queue = NULL;


    cand        = (int *)malloc((size_t)N * sizeof(int));
    delta       = (int *)malloc((size_t)N * sizeof(int));
    delta_order = (int *)malloc((size_t)N * sizeof(int));
    if (!cand || !delta || !delta_order) goto fail;

    {
        int iter;
        for (iter = 0; iter < KMEANS_L1_ITERS; iter++) {
            int moves = 0;
            int v;
            t0 = reorder_now_ms();


            for (v = 0; v < N; v++) {
                int cur = label[v];
                int cur_score, best_c, best_s;
                int t;
                touched_count = 0;
                {
                    const int32_t *nb = graph + (size_t)v * (size_t)K;
                    for (j = 0; j < K; j++) {
                        int u = nb[j];
                        int lu;
                        if (u < 0 || u >= N || u == v) continue;
                        lu = label[u];
                        if (score[lu] == 0) touched[touched_count++] = lu;
                        score[lu]++;
                    }
                }
                cur_score = score[cur];
                best_c = cur; best_s = cur_score;
                for (t = 0; t < touched_count; t++) {
                    int cc = touched[t];
                    if (score[cc] > best_s ||
                        (score[cc] == best_s && cc < best_c)) {
                        best_s = score[cc]; best_c = cc;
                    }
                }
                cand[v]  = best_c;
                delta[v] = best_s - cur_score;
                for (t = 0; t < touched_count; t++) score[touched[t]] = 0;
            }


            {
                int *bk = (int *)calloc((size_t)(2 * K + 1), sizeof(int));
                int *bs = (int *)malloc((size_t)(2 * K + 1) * sizeof(int));
                if (!bk || !bs) { free(bk); free(bs); goto fail; }
                for (v = 0; v < N; v++) {
                    int d = delta[v] + K;
                    if (d < 0) d = 0;
                    if (d > 2 * K) d = 2 * K;
                    bk[d]++;
                }
                bs[2 * K] = 0;
                for (i = 2 * K - 1; i >= 0; i--) bs[i] = bs[i + 1] + bk[i + 1];
                for (v = 0; v < N; v++) {
                    int d = delta[v] + K;
                    if (d < 0) d = 0;
                    if (d > 2 * K) d = 2 * K;
                    delta_order[bs[d]++] = v;
                }
                free(bk); free(bs);
            }


            for (i = 0; i < N; i++) {
                int vv = delta_order[i];
                int target_c;
                if (delta[vv] <= 0) break;
                target_c = cand[vv];
                if (target_c == label[vv]) continue;
                if (csize[target_c] < cap_soft) {
                    csize[label[vv]]--;
                    label[vv] = target_c;
                    csize[target_c]++;
                    moves++;
                } else {

                    int t;
                    int sec_c = -1, sec_s = -1;
                    touched_count = 0;
                    {
                        const int32_t *nb = graph + (size_t)vv * (size_t)K;
                        for (j = 0; j < K; j++) {
                            int u = nb[j];
                            int lu;
                            if (u < 0 || u >= N || u == vv) continue;
                            lu = label[u];
                            if (lu == target_c) continue;
                            if (score[lu] == 0) touched[touched_count++] = lu;
                            score[lu]++;
                        }
                    }
                    for (t = 0; t < touched_count; t++) {
                        int cc = touched[t];
                        if (cc == label[vv]) continue;
                        if (csize[cc] >= cap_soft) continue;
                        if (score[cc] > sec_s ||
                            (score[cc] == sec_s && cc < sec_c)) {
                            sec_s = score[cc]; sec_c = cc;
                        }
                    }
                    if (sec_c >= 0 && sec_s > score[label[vv]]) {
                        csize[label[vv]]--;
                        label[vv] = sec_c;
                        csize[sec_c]++;
                        moves++;
                    }
                    for (t = 0; t < touched_count; t++) score[touched[t]] = 0;
                }
            }

            t1 = reorder_now_ms();
            printf("[reorder B]   Lloyd iter %d:    moves=%d (%.1f%%) %.0f ms\n",
                   iter, moves, (double)moves * 100.0 / (double)N, t1 - t0);
            if ((long)moves * 100 < (long)N) {
                printf("[reorder B]   Lloyd early stop (moves < 1%%)\n");
                break;
            }
        }
    }
    free(cand); cand = NULL;
    free(delta); delta = NULL;
    free(delta_order); delta_order = NULL;


#if KMEANS_L1_KL_ENABLE
    t0 = reorder_now_ms();
    pair_count = (int *)calloc((size_t)M_c * (size_t)M_c, sizeof(int));
    if (!pair_count) goto fail;
    for (i = 0; i < N; i++) {
        int li = label[i];
        const int32_t *nb = graph + (size_t)i * (size_t)K;
        for (j = 0; j < K; j++) {
            int u = nb[j];
            int lu;
            if (u < 0 || u >= N) continue;
            lu = label[u];
            if (lu == li) continue;
            pair_count[(size_t)li * (size_t)M_c + (size_t)lu]++;
        }
    }
    {

        int *kl_offset = (int *)malloc((size_t)(M_c + 1) * sizeof(int));
        int *kl_members = NULL;
        int *kl_cursor = NULL;
        int swaps = 0;
        int pairs_processed = 0;
        const int min_pair = 4;
        int Ca, Cb;

        if (!kl_offset) { free(pair_count); pair_count = NULL; goto kl_done; }
        kl_offset[0] = 0;
        for (c = 0; c < M_c; c++) kl_offset[c + 1] = kl_offset[c] + csize[c];
        kl_members = (int *)malloc((size_t)N * sizeof(int));
        kl_cursor  = (int *)malloc((size_t)M_c * sizeof(int));
        if (!kl_members || !kl_cursor) {
            free(kl_offset); free(kl_members); free(kl_cursor);
            free(pair_count); pair_count = NULL;
            goto kl_done;
        }
        for (c = 0; c < M_c; c++) kl_cursor[c] = kl_offset[c];
        for (i = 0; i < N; i++) {
            int li = label[i];
            kl_members[kl_cursor[li]++] = i;
        }


        {
            int *pair_Ca = (int *)malloc((size_t)(M_c * M_c / 2) * sizeof(int));
            int *pair_Cb = (int *)malloc((size_t)(M_c * M_c / 2) * sizeof(int));
            int npairs = 0;
            const int max_pairs = M_c * 2;
            int p;

            if (!pair_Ca || !pair_Cb) {
                free(pair_Ca); free(pair_Cb);
                goto kl_cleanup;
            }
            for (Ca = 0; Ca < M_c; Ca++) {
                for (Cb = Ca + 1; Cb < M_c; Cb++) {
                    int pc_ab = pair_count[(size_t)Ca * (size_t)M_c + (size_t)Cb];
                    int pc_ba = pair_count[(size_t)Cb * (size_t)M_c + (size_t)Ca];
                    if (pc_ab < min_pair && pc_ba < min_pair) continue;
                    pair_Ca[npairs] = Ca;
                    pair_Cb[npairs] = Cb;
                    npairs++;
                }
            }


            for (p = 0; p < max_pairs && p < npairs; p++) {
                int best_idx = p;
                int best_pc = 0;
                int pp;
                for (pp = p; pp < npairs; pp++) {
                    int pc = pair_count[(size_t)pair_Ca[pp] * (size_t)M_c + (size_t)pair_Cb[pp]]
                           + pair_count[(size_t)pair_Cb[pp] * (size_t)M_c + (size_t)pair_Ca[pp]];
                    if (pc > best_pc) { best_pc = pc; best_idx = pp; }
                }
                if (best_idx != p) {
                    int tc = pair_Ca[p]; pair_Ca[p] = pair_Ca[best_idx]; pair_Ca[best_idx] = tc;
                    tc = pair_Cb[p]; pair_Cb[p] = pair_Cb[best_idx]; pair_Cb[best_idx] = tc;
                }
            }

            for (p = 0; p < max_pairs && p < npairs; p++) {
                int tries;
                int ia, ib;
                Ca = pair_Ca[p];
                Cb = pair_Cb[p];


                for (tries = 0; tries < 2; tries++) {
                    int src = (tries == 0) ? Ca : Cb;
                    int dst = (tries == 0) ? Cb : Ca;
                    int swap_budget = csize[src] < csize[dst] ? csize[src] / 4 : csize[dst] / 4;
                    int t2;
                    if (swap_budget < 1) continue;

                    for (t2 = 0, ia = kl_offset[src]; ia < kl_offset[src + 1] && t2 < swap_budget; ia++) {
                        int v = kl_members[ia];
                        int v_src_s = 0, v_dst_s = 0;
                        if (label[v] != src) continue;
                        {
                            const int32_t *nb = graph + (size_t)v * (size_t)K;
                            for (j = 0; j < K; j++) {
                                int u = nb[j];
                                if (u < 0 || u >= N || u == v) continue;
                                if (label[u] == src) v_src_s++;
                                else if (label[u] == dst) v_dst_s++;
                            }
                        }
                        if (v_dst_s < 2) continue;

                        for (ib = kl_offset[dst]; ib < kl_offset[dst + 1]; ib++) {
                            int u_idx = kl_members[ib];
                            int u_src_s = 0, u_dst_s = 0;
                            int gain;
                            if (label[u_idx] != dst) continue;
                            {
                                const int32_t *nb2 = graph + (size_t)u_idx * (size_t)K;
                                for (j = 0; j < K; j++) {
                                    int w = nb2[j];
                                    if (w < 0 || w >= N || w == u_idx) continue;
                                    if (label[w] == src) u_src_s++;
                                    else if (label[w] == dst) u_dst_s++;
                                }
                            }
                            if (u_src_s < 2) continue;
                            gain = v_dst_s + u_src_s - v_src_s - u_dst_s;
                            if (gain > 0) {
                                label[v] = dst;
                                label[u_idx] = src;
                                swaps++;
                                t2++;
                                break;
                            }
                        }
                    }
                }
                pairs_processed++;
            }
            free(pair_Ca); free(pair_Cb);
        }
        t1 = reorder_now_ms();
        printf("[reorder B]   KL refine:       swaps=%d %d pairs %.0f ms\n",
               swaps, pairs_processed, t1 - t0);
        free(kl_offset); free(kl_members); free(kl_cursor);
    }
    free(pair_count); pair_count = NULL;
kl_cleanup:
kl_done:
#endif


    t0 = reorder_now_ms();
    {
        int *surplus = (int *)calloc((size_t)M_c, sizeof(int));
        int *deficit = (int *)calloc((size_t)M_c, sizeof(int));
        int *donors  = NULL;
        int donors_count = 0;
        int donors_cap = 0;
        if (!surplus || !deficit) {
            free(surplus); free(deficit); goto fail;
        }
        for (c = 0; c < M_c; c++) {
            int r = csize[c] % vpb;
            surplus[c] = r;
            deficit[c] = 0;
            donors_cap += r;
        }

        {
            int remaining = donors_cap;

            for (c = 0; c < M_c && remaining > 0; c++) {
                if (surplus[c] > 0) continue;
                deficit[c] = vpb;
                remaining -= vpb;
            }

            for (c = 0; c < M_c && remaining > 0; c++) {
                if (surplus[c] == 0) continue;
                deficit[c] = vpb;
                remaining -= vpb;
            }
        }
        donors = (int *)malloc((size_t)(donors_cap > 0 ? donors_cap : 1) * sizeof(int));
        if (!donors) {
            free(surplus); free(deficit); goto fail;
        }


        for (c = 0; c < M_c; c++) {
            int s = surplus[c];
            int *cmembers = NULL;
            int *aff = NULL;
            int cm_count = 0;
            int t;
            if (s == 0) continue;
            cmembers = (int *)malloc((size_t)csize[c] * sizeof(int));
            aff      = (int *)malloc((size_t)csize[c] * sizeof(int));
            if (!cmembers || !aff) {
                free(cmembers); free(aff);
                free(surplus); free(deficit); free(donors);
                goto fail;
            }
            for (i = 0; i < N; i++) if (label[i] == c) cmembers[cm_count++] = i;

            for (t = 0; t < cm_count; t++) {
                int v = cmembers[t];
                int a = 0;
                const int32_t *nb = graph + (size_t)v * (size_t)K;
                for (j = 0; j < K; j++) {
                    int u = nb[j];
                    if (u < 0 || u >= N || u == v) continue;
                    if (label[u] == c) a++;
                }
                aff[t] = a;
            }

            for (t = 0; t < s; t++) {
                int min_idx = -1, min_aff = INT_MAX;
                int k2;
                for (k2 = 0; k2 < cm_count; k2++) {
                    if (aff[k2] < 0) continue;
                    if (aff[k2] < min_aff) { min_aff = aff[k2]; min_idx = k2; }
                }
                if (min_idx < 0) break;
                donors[donors_count++] = cmembers[min_idx];
                label[cmembers[min_idx]] = -1;
                csize[c]--;
                aff[min_idx] = -1;
            }
            free(cmembers); free(aff);
        }


        for (i = 0; i < donors_count; i++) {
            int v = donors[i];
            int t;
            int best_c = -1, best_s = -1;
            touched_count = 0;
            {
                const int32_t *nb = graph + (size_t)v * (size_t)K;
                for (j = 0; j < K; j++) {
                    int u = nb[j];
                    int lu;
                    if (u < 0 || u >= N) continue;
                    lu = label[u];
                    if (lu < 0) continue;
                    if (deficit[lu] <= 0) continue;
                    if (score[lu] == 0) touched[touched_count++] = lu;
                    score[lu]++;
                }
            }
            for (t = 0; t < touched_count; t++) {
                int cc = touched[t];
                if (score[cc] > best_s ||
                    (score[cc] == best_s && cc < best_c)) {
                    best_s = score[cc]; best_c = cc;
                }
            }
            if (best_c < 0) {

                for (c = 0; c < M_c; c++) {
                    if (deficit[c] > 0) { best_c = c; break; }
                }
            }
            label[v] = best_c;
            csize[best_c]++;
            deficit[best_c]--;
            for (t = 0; t < touched_count; t++) score[touched[t]] = 0;
        }
        t1 = reorder_now_ms();
        printf("[reorder B]   vpb align:       donors=%d %.0f ms\n",
               donors_count, t1 - t0);
        free(surplus); free(deficit); free(donors);
    }


    for (c = 0; c < M_c; c++) {
        if (csize[c] % vpb != 0) {
            printf("[reorder B] ERROR: cluster %d size %d not multiple of vpb %d\n",
                   c, csize[c], vpb);
            goto fail;
        }
    }


    cluster_offset = (int *)malloc((size_t)(M_c + 1) * sizeof(int));
    member_buf     = (int *)malloc((size_t)N * sizeof(int));
    if (!cluster_offset || !member_buf) goto fail;


    cluster_offset[0] = 0;
    for (c = 0; c < M_c; c++) cluster_offset[c + 1] = cluster_offset[c] + csize[c];
    {
        int *cursor = (int *)malloc((size_t)M_c * sizeof(int));
        if (!cursor) goto fail;
        for (c = 0; c < M_c; c++) cursor[c] = cluster_offset[c];
        for (i = 0; i < N; i++) {
            int li = label[i];
            member_buf[cursor[li]++] = i;
        }
        free(cursor);
    }


    t0 = reorder_now_ms();
    assigned = (int *)calloc((size_t)N, sizeof(int));
    block_buf = (int *)malloc((size_t)vpb * sizeof(int));
    if (!assigned || !block_buf) goto fail;

    {

        int *intra_deg = (int *)malloc((size_t)N * sizeof(int));
        int *idx_buf   = (int *)malloc((size_t)N * sizeof(int));
        if (!intra_deg || !idx_buf) {
            free(intra_deg); free(idx_buf); goto fail;
        }


        for (i = 0; i < N; i++) {
            int li = label[i];
            int a = 0;
            const int32_t *nb = graph + (size_t)i * (size_t)K;
            for (j = 0; j < K; j++) {
                int u = nb[j];
                if (u < 0 || u >= N || u == i) continue;
                if (label[u] == li) a++;
            }
            intra_deg[i] = a;
        }

        for (c = 0; c < M_c; c++) {
            int cs = cluster_offset[c];
            int ce = cluster_offset[c + 1];
            int t;

            {
                int *bk = (int *)calloc((size_t)(K + 1), sizeof(int));
                int *bs = (int *)malloc((size_t)(K + 1) * sizeof(int));
                if (!bk || !bs) { free(bk); free(bs); free(intra_deg); free(idx_buf); goto fail; }
                for (t = cs; t < ce; t++) bk[intra_deg[member_buf[t]]]++;
                bs[K] = 0;
                for (t = K - 1; t >= 0; t--) bs[t] = bs[t + 1] + bk[t + 1];
                for (t = cs; t < ce; t++) {
                    int v = member_buf[t];
                    idx_buf[cs + bs[intra_deg[v]]++] = v;
                }
                free(bk); free(bs);
            }


            {
                int write_cursor = cs;
                int t2;
                for (t2 = cs; t2 < ce; t2++) {
                    int seed_v = idx_buf[t2];
                    int block_size;
                    if (assigned[seed_v]) continue;

                    block_buf[0] = seed_v;
                    block_size = 1;
                    assigned[seed_v] = 1;
                    touched_count = 0;
                    {
                        const int32_t *nb = graph + (size_t)seed_v * (size_t)K;
                        for (j = 0; j < K; j++) {
                            int u = nb[j];
                            if (u < 0 || u >= N) continue;
                            if (label[u] != c || assigned[u]) continue;
                            if (score[u] == 0) touched[touched_count++] = u;
                            score[u]++;
                        }
                    }

                    while (block_size < vpb) {
                        int best_v = -1, best_s = 0;
                        int t3;
                        for (t3 = 0; t3 < touched_count; t3++) {
                            int u = touched[t3];
                            if (assigned[u]) continue;
                            if (score[u] > best_s) { best_s = score[u]; best_v = u; }
                        }
                        if (best_v < 0) {

                            int t4;
                            for (t4 = cs; t4 < ce; t4++) {
                                int u = idx_buf[t4];
                                if (!assigned[u]) { best_v = u; break; }
                            }
                            if (best_v < 0) break;
                        }
                        block_buf[block_size++] = best_v;
                        assigned[best_v] = 1;
                        {
                            const int32_t *nb = graph + (size_t)best_v * (size_t)K;
                            for (j = 0; j < K; j++) {
                                int u = nb[j];
                                if (u < 0 || u >= N) continue;
                                if (label[u] != c || assigned[u]) continue;
                                if (score[u] == 0) touched[touched_count++] = u;
                                score[u]++;
                            }
                        }
                    }


                    {
                        int t3;
                        for (t3 = 0; t3 < touched_count; t3++) score[touched[t3]] = 0;
                    }
                    {
                        int b;
                        for (b = 0; b < block_size; b++)
                            member_buf[write_cursor++] = block_buf[b];
                    }
                }

                if (write_cursor != ce) {
                    int t3;
                    for (t3 = cs; t3 < ce && write_cursor < ce; t3++) {
                        int u = idx_buf[t3];
                        if (!assigned[u]) {
                            assigned[u] = 1;
                            member_buf[write_cursor++] = u;
                        }
                    }
                }
            }
        }
        free(intra_deg); free(idx_buf);
    }
    t1 = reorder_now_ms();
    printf("[reorder B]   L2 packing:      %.0f ms\n", t1 - t0);


    t0 = reorder_now_ms();

    cluster_total_score = (int *)calloc((size_t)M_c, sizeof(int));
    cluster_emit_order  = (int *)malloc((size_t)M_c * sizeof(int));
    if (!cluster_total_score || !cluster_emit_order) goto fail;
    for (i = 0; i < N; i++) {
        int li = label[i];
        cluster_total_score[li] += K + in_degree[i];
    }
    {

        int *tmp = (int *)malloc((size_t)M_c * sizeof(int));
        if (!tmp) goto fail;
        for (c = 0; c < M_c; c++) tmp[c] = c;

        for (i = 0; i < M_c - 1; i++) {
            int max_idx = i;
            int kk;
            for (kk = i + 1; kk < M_c; kk++) {
                if (cluster_total_score[tmp[kk]] > cluster_total_score[tmp[max_idx]])
                    max_idx = kk;
            }
            if (max_idx != i) {
                int t = tmp[i]; tmp[i] = tmp[max_idx]; tmp[max_idx] = t;
            }
        }
        for (c = 0; c < M_c; c++) cluster_emit_order[c] = tmp[c];
        free(tmp);
    }
    {
        int new_id = 0;
        int oc;
        for (oc = 0; oc < M_c; oc++) {
            int cc = cluster_emit_order[oc];
            int cs = cluster_offset[cc];
            int ce = cluster_offset[cc + 1];
            int t;
            for (t = cs; t < ce; t++) {
                int v = member_buf[t];
                reorder_map[v] = new_id;
                reverse_map[new_id] = v;
                new_id++;
            }
        }
        if (new_id != N) {
            printf("[reorder B] ERROR: emit count %d != N %d\n", new_id, N);
            goto fail;
        }
    }
    t1 = reorder_now_ms();
    printf("[reorder B]   emit:            %.0f ms\n", t1 - t0);
    printf("[reorder B] total: %.1f s\n", (reorder_now_ms() - t_total0) / 1000.0);

    rc = 0;

fail:
    free(in_degree);
    free(deg_order);
    free(label);
    free(csize);
    free(seeds);
    free(queue);
    free(score);
    free(touched);
    free(cand);
    free(delta);
    free(delta_order);
    free(pair_count);
    free(cluster_offset);
    free(member_buf);
    free(assigned);
    free(block_buf);
    free(cluster_emit_order);
    free(cluster_total_score);
    return rc;
}
