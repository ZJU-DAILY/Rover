static float *g_rnn_l2_base = NULL;
static int g_rnn_l2_fd_base = -1;
static int g_rnn_l2_num_vectors = 0;
static int g_rnn_l2_dim = 0;

int calculator_retrieval_rnn_l2_unload_dma(remote_handle64 h) {
    (void)h;

    if (g_rnn_l2_fd_base >= 0) {
        HAP_mmap_put(g_rnn_l2_fd_base);
    }

    g_rnn_l2_base = NULL;
    g_rnn_l2_fd_base = -1;
    g_rnn_l2_num_vectors = 0;
    g_rnn_l2_dim = 0;
    FARF(HIGH, "retrieval_rnn_l2_unload_dma: done");
    return AEE_SUCCESS;
}

int calculator_retrieval_rnn_l2_load_dma(remote_handle64 h,
                                         int32 fd_base,
                                         int32 num_vectors,
                                         int32 dim) {
    void *base_vaddr = NULL;
    uint64 base_paddr = 0;
    int ret;

    (void)h;

    calculator_retrieval_rnn_l2_unload_dma(h);

    if (fd_base < 0 || num_vectors <= 1 || dim <= 0) {
        return AEE_EBADPARM;
    }

    ret = HAP_mmap_get(fd_base, &base_vaddr, &base_paddr);
    if (ret != 0) {
        FARF(ERROR, "retrieval_rnn_l2_load_dma: HAP_mmap_get(base fd=%d) failed 0x%x",
             fd_base, ret);
        return AEE_EFAILED;
    }

    g_rnn_l2_base = (float *)base_vaddr;
    g_rnn_l2_fd_base = (int)fd_base;
    g_rnn_l2_num_vectors = (int)num_vectors;
    g_rnn_l2_dim = (int)dim;

    FARF(HIGH, "retrieval_rnn_l2_load_dma: n=%d dim=%d",
         (int)num_vectors, (int)dim);
    return AEE_SUCCESS;
}

int calculator_retrieval_rnn_l2_batch(remote_handle64 h,
                                      const int *ids_a,
                                      int ids_aLen,
                                      const int *ids_b,
                                      int ids_bLen,
                                      int32 pair_count,
                                      float *distances,
                                      int distancesLen) {
    int i;

    (void)h;

    if (!g_rnn_l2_base) {
        return AEE_EFAILED;
    }
    if (!ids_a || !ids_b || !distances ||
        pair_count <= 0 ||
        ids_aLen < pair_count ||
        ids_bLen < pair_count ||
        distancesLen < pair_count) {
        return AEE_EBADPARM;
    }

    for (i = 0; i < pair_count; i++) {
        int a = ids_a[i];
        int b = ids_b[i];
        const float *va;
        const float *vb;
        float acc = 0.0f;
        int d;

        if (a < 0 || a >= g_rnn_l2_num_vectors ||
            b < 0 || b >= g_rnn_l2_num_vectors) {
            return AEE_EBADPARM;
        }

        va = g_rnn_l2_base + (size_t)a * (size_t)g_rnn_l2_dim;
        vb = g_rnn_l2_base + (size_t)b * (size_t)g_rnn_l2_dim;
        for (d = 0; d < g_rnn_l2_dim; d++) {
            float diff = va[d] - vb[d];
            acc += diff * diff;
        }
        distances[i] = acc;
    }

    return AEE_SUCCESS;
}
