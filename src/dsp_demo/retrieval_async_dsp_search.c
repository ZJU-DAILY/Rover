#include "qurt_timer.h"

#if defined(RETRIEVAL_ENABLE_FULL_BUILD) && RETRIEVAL_ENABLE_BUILD_PAUSE
static void build15_dsp_pause(RetrievalBuildPause *pause, int participant) {
    if(participant<0 || participant>=RETRIEVAL_BUILD_PAUSE_PARTICIPANTS)return;
    RetrievalBuildPauseSlot *ack=&pause->ack[participant];
    qurt_mem_cache_clean((qurt_addr_t)&pause->request,128,QURT_MEM_CACHE_INVALIDATE,QURT_MEM_DCACHE);
    DB_READ_BARRIER();
    if (!pause->request.value || ack->value<0) return;
    ack->value=1;DB_WRITE_BARRIER();
    qurt_mem_cache_clean((qurt_addr_t)ack,128,QURT_MEM_CACHE_FLUSH,QURT_MEM_DCACHE);
    do {
        if (qurt_timer_sleep(10000) != QURT_EOK) {

            ack->value=-1;DB_WRITE_BARRIER();
            qurt_mem_cache_clean((qurt_addr_t)ack,128,QURT_MEM_CACHE_FLUSH,QURT_MEM_DCACHE);
            return;
        }
        qurt_mem_cache_clean((qurt_addr_t)&pause->request,128,QURT_MEM_CACHE_INVALIDATE,QURT_MEM_DCACHE);
        DB_READ_BARRIER();
    } while (pause->request.value);
    ack->value=0;DB_WRITE_BARRIER();
    qurt_mem_cache_clean((qurt_addr_t)ack,128,QURT_MEM_CACHE_FLUSH,QURT_MEM_DCACHE);
}
#endif


typedef struct DspSearchCandidate {
    int   id;
    float distance;
} DspSearchCandidate;


typedef struct DspFrontierEntry {
    int   id;
    float distance;
} DspFrontierEntry;

typedef struct DspSearchContext {
    uint32_t *visited;
    volatile int in_use;
} DspSearchContext;

static int32_t          *g_search_graph         = NULL;
static uint8_t          *g_search_l1_codes      = NULL;
static DspSearchContext  g_search_ctx[DSP_SEARCH_MAX_CONCURRENT];
static int               g_search_ctx_count     = 0;
static int               g_search_num_vectors   = 0;
static int               g_search_dim           = 0;
static int               g_search_l1_bits       = 4;
static int               g_search_l1_code_bytes = 0;
static int               g_search_graph_k       = 0;
static int               g_search_entry_point   = DATA_GRAPH_ENTRY_POINT;
static float             g_search_l1_scale      = 0.0f;
static float             g_search_l1_zero_point = 0.0f;
static int               g_search_dma_mode      = 0;
static int               g_search_fd_graph      = -1;

static uint8_t          *g_search_l1_chunks[MAX_L1_CHUNKS] = { NULL };
static int               g_search_l1_fds[MAX_L1_CHUNKS];
static int               g_search_l1_chunk_count   = 0;
static int               g_search_l1_chunk_vectors = 0;
static uint8_t          *g_pure_pq_code_chunks[PURE_PQ_MAX_CODE_CHUNKS];
static int               g_pure_pq_code_fds[PURE_PQ_MAX_CODE_CHUNKS];
static int               g_pure_pq_chunk_counts[PURE_PQ_MAX_CODE_CHUNKS];
static int               g_pure_pq_chunk_count = 0;
static int               g_pure_pq_chunk_vectors = 0;
static float            *g_pure_pq_codebook = NULL;
static int               g_pure_pq_num_vectors = 0;
static int               g_pure_pq_dim = 0;
static int               g_pure_pq_m = 0;
static int               g_pure_pq_ks = 0;
static int               g_pure_pq_subdim = 0;
static int               g_pure_pq_code_bytes = 0;
static uint8_t           g_pure_pq_index_uuid[16];
static volatile int      g_pure_pq_session_live = 0;


int calculator_retrieval_unload_dma(remote_handle64 h);
int calculator_retrieval_pure_pq_unload_dma(remote_handle64 h);


static inline const uint8_t *dsp_l1_code_ptr(int vec) {
    int vpc = g_search_l1_chunk_vectors;
    int c   = (vpc > 0) ? (vec / vpc) : 0;
    int off = (vpc > 0) ? (vec % vpc) : vec;
    return g_search_l1_chunks[c] +
           (size_t)off * (size_t)g_search_l1_code_bytes;
}


static void dsp_l1_reset_copy(void) {
    free(g_search_l1_codes);
    g_search_l1_codes = NULL;
    g_search_l1_chunks[0]     = NULL;
    g_search_l1_chunk_count   = 0;
    g_search_l1_chunk_vectors = 0;
}


static void dsp_heap_sift_up(DspSearchCandidate *pool, int i) {
    DspSearchCandidate tmp = pool[i];
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (tmp.distance <= pool[parent].distance)
            break;
        pool[i] = pool[parent];
        i = parent;
    }
    pool[i] = tmp;
}

static void dsp_heap_sift_down(DspSearchCandidate *pool, int size, int i) {
    DspSearchCandidate tmp = pool[i];
    while (1) {
        int child = 2 * i + 1;
        if (child >= size)
            break;
        if (child + 1 < size &&
            pool[child + 1].distance > pool[child].distance)
            child++;
        if (tmp.distance >= pool[child].distance)
            break;
        pool[i] = pool[child];
        i = child;
    }
    pool[i] = tmp;
}


static int dsp_search_candidate_insert(DspSearchCandidate *pool,
                                       int *pool_size,
                                       int pool_capacity,
                                       int id,
                                       float distance) {
    int size = *pool_size;

    if (size < pool_capacity) {
        pool[size].id       = id;
        pool[size].distance = distance;
        dsp_heap_sift_up(pool, size);
        *pool_size = size + 1;
        return 1;
    } else if (distance < pool[0].distance) {
        pool[0].id       = id;
        pool[0].distance = distance;
        dsp_heap_sift_down(pool, size, 0);
        return 1;
    }
    return 0;
}


static void dsp_heap_sort(DspSearchCandidate *pool, int size) {
    int i;
    for (i = size - 1; i > 0; i--) {
        DspSearchCandidate tmp = pool[0];
        pool[0] = pool[i];
        pool[i] = tmp;
        dsp_heap_sift_down(pool, i, 0);
    }
}


static void dsp_frontier_sift_up(DspFrontierEntry *f, int i) {
    DspFrontierEntry tmp = f[i];
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (tmp.distance >= f[parent].distance)
            break;
        f[i] = f[parent];
        i = parent;
    }
    f[i] = tmp;
}

static void dsp_frontier_sift_down(DspFrontierEntry *f, int size, int i) {
    DspFrontierEntry tmp = f[i];
    while (1) {
        int child = 2 * i + 1;
        if (child >= size)
            break;
        if (child + 1 < size &&
            f[child + 1].distance < f[child].distance)
            child++;
        if (tmp.distance <= f[child].distance)
            break;
        f[i] = f[child];
        i = child;
    }
    f[i] = tmp;
}

static void dsp_frontier_push(DspFrontierEntry *f, int *f_size,
                              int id, float distance) {
    int pos = *f_size;
    f[pos].id       = id;
    f[pos].distance = distance;
    dsp_frontier_sift_up(f, pos);
    *f_size = pos + 1;
}


static int dsp_frontier_pop(DspFrontierEntry *f, int *f_size) {
    int top_id;

    if (*f_size <= 0)
        return -1;

    top_id = f[0].id;
    *f_size -= 1;
    if (*f_size > 0) {
        f[0] = f[*f_size];
        dsp_frontier_sift_down(f, *f_size, 0);
    }
    return top_id;
}

static int dsp_search_select_entry_id(int entry_idx, unsigned int *seed) {
#if 1
    if (entry_idx == 0 &&
        g_search_entry_point >= 0 &&
        g_search_entry_point < g_search_num_vectors) {
        return g_search_entry_point;
    }
#endif

    *seed = *seed * 1103515245u + 12345u;
    return (int)(((uint64_t)(*seed) * (uint64_t)g_search_num_vectors) >> 32);
}

static void dsp_search_release_context(DspSearchContext *ctx) {
    if (!ctx) {
        return;
    }
    __sync_lock_release(&ctx->in_use);
}

static DspSearchContext *dsp_search_acquire_context(void) {
    int i;

    for (i = 0; i < g_search_ctx_count; i++) {
        if (__sync_bool_compare_and_swap(&g_search_ctx[i].in_use, 0, 1)) {
            return &g_search_ctx[i];
        }
    }
    return NULL;
}


static void dsp_search_free_contexts(void) {
    int i;

    for (i = 0; i < DSP_SEARCH_MAX_CONCURRENT; i++) {
        free(g_search_ctx[i].visited);
        g_search_ctx[i].visited = NULL;
        g_search_ctx[i].in_use = 0;
    }
    g_search_ctx_count = 0;
}

static int dsp_search_quantize_query_l1(const float *query, uint8_t *query_l1) {
    if (g_search_l1_bits == 4) {
        return retrieval_quantize_query_u4(query,
                                           g_search_dim,
                                           g_search_l1_scale,
                                           g_search_l1_zero_point,
                                           query_l1);
    }

    return retrieval_quantize_query_u8(query,
                                       g_search_dim,
                                       g_search_l1_scale,
                                       g_search_l1_zero_point,
                                       query_l1);
}

static float dsp_search_compute_l1_distance(const uint8_t *query_l1,
                                            const uint8_t *db_codes) {
    if (g_search_l1_bits == 4) {
        return retrieval_compute_int4_packed_distance_u8(query_l1,
                                                         db_codes,
                                                         g_search_dim,
                                                         g_search_l1_scale);
    }

    return retrieval_compute_int8_native_distance_u8(query_l1,
                                                     db_codes,
                                                     g_search_dim,
                                                     g_search_l1_scale);
}

int calculator_retrieval_load(remote_handle64 h,
                              const int   *graph,    int graphLen,
                              const uint8 *l1_codes, int l1CodesLen,
                              int32 num_vectors, int32 dim, int32 graph_k,
                              int32 graph_entry_point,
                              float l1_scale,   float l1_zero_point) {
    size_t graph_bytes, codes_bytes;
    int l1_code_bytes;
    int i;

    (void)h;

    free(g_search_graph);    g_search_graph    = NULL;
    dsp_l1_reset_copy();
    dsp_search_free_contexts();

    if (num_vectors <= 0 || dim <= 0 || graph_k <= 0)
        return AEE_EBADPARM;
    l1_code_bytes = RETRIEVAL_DEMO_L1_CODE_BYTES(dim);
    if (graphLen   < num_vectors * graph_k)
        return AEE_EBADPARM;
    if (l1CodesLen < num_vectors * l1_code_bytes)
        return AEE_EBADPARM;

    graph_bytes = (size_t)num_vectors * (size_t)graph_k * sizeof(int32_t);
    codes_bytes = (size_t)num_vectors * (size_t)l1_code_bytes;

    g_search_graph    = (int32_t        *)malloc(graph_bytes);
    g_search_l1_codes = (uint8_t        *)malloc(codes_bytes);

    if (!g_search_graph || !g_search_l1_codes) {
        free(g_search_graph);    g_search_graph    = NULL;
        dsp_l1_reset_copy();
        return AEE_ENOMEMORY;
    }

    memcpy(g_search_graph,    graph,    graph_bytes);
    memcpy(g_search_l1_codes, l1_codes, codes_bytes);


    g_search_l1_chunks[0]     = g_search_l1_codes;
    g_search_l1_chunk_count   = 1;
    g_search_l1_chunk_vectors = (int)num_vectors;

    g_search_num_vectors   = (int)num_vectors;
    g_search_dim           = (int)dim;
    g_search_l1_bits       = RETRIEVAL_DEMO_L1_BITS;
    g_search_l1_code_bytes = l1_code_bytes;
    g_search_graph_k       = (int)graph_k;
    g_search_entry_point   = (graph_entry_point >= 0 &&
                              graph_entry_point < num_vectors)
                                 ? (int)graph_entry_point
                                 : DATA_GRAPH_ENTRY_POINT;
    g_search_l1_scale      = l1_scale;
    g_search_l1_zero_point = l1_zero_point;
    g_search_ctx_count     = DSP_SEARCH_MAX_CONCURRENT;

    for (i = 0; i < g_search_ctx_count; i++) {
        g_search_ctx[i].visited = (uint32_t *)calloc(
            VISITED_BITMAP_WORDS(num_vectors), sizeof(uint32_t));
        g_search_ctx[i].in_use = 0;
        if (!g_search_ctx[i].visited) {
            free(g_search_graph);    g_search_graph    = NULL;
            dsp_l1_reset_copy();
            dsp_search_free_contexts();
            return AEE_ENOMEMORY;
        }
    }

    FARF(HIGH, "retrieval_load: n=%d dim=%d l1=int%d code_bytes=%d k=%d scale=%.6f ctx=%d",
         num_vectors, dim, g_search_l1_bits, g_search_l1_code_bytes,
         graph_k, l1_scale, g_search_ctx_count);
    return AEE_SUCCESS;
}

int calculator_retrieval_search(remote_handle64 h,
                                const float *query, int queryLen,
                                int32 entry_points, int32 expand_count,
                                int32 pool_size,    int32 num_out,
                                int   *candidate_ids,   int candidate_idsLen,
                                float *candidate_dists, int candidate_distsLen,
                                int32 *actual_count) {
    uint8_t query_l1[RETRIEVAL_ASYNC_DEMO_MAX_DIM] __attribute__((aligned(128)));
    DspSearchCandidate *pool = NULL;
    DspFrontierEntry *frontier = NULL;
    int frontier_size = 0;
    DspSearchContext *ctx = NULL;
    uint32_t *visited = NULL;
    int pool_size_actual = 0;
    int expand_limit;
    int i;
    unsigned int seed = DSP_SEARCH_ENTRY_SEED;

    (void)h;

    if (!actual_count) {
        return AEE_EBADPARM;
    }
    *actual_count = 0;
    if (!g_search_graph || !g_search_l1_chunks[0] || g_search_ctx_count <= 0)
        return AEE_EFAILED;
    if (queryLen < g_search_dim || pool_size <= 0 || num_out <= 0)
        return AEE_EBADPARM;
    if (candidate_idsLen < num_out || candidate_distsLen < num_out)
        return AEE_EBADPARM;

    expand_limit = (expand_count < g_search_graph_k)
                       ? expand_count : g_search_graph_k;
    if (expand_limit <= 0)
        return AEE_EBADPARM;
    if (pool_size > DSP_SEARCH_MAX_POOL)
        pool_size = DSP_SEARCH_MAX_POOL;

    pool = (DspSearchCandidate *)malloc((size_t)DSP_SEARCH_MAX_POOL *
                                        sizeof(DspSearchCandidate));
    if (!pool)
        return AEE_ENOMEMORY;

    {
        size_t frontier_cap = (size_t)entry_points +
                              (size_t)pool_size * (size_t)expand_limit;
        frontier = (DspFrontierEntry *)malloc(frontier_cap *
                                              sizeof(DspFrontierEntry));
        if (!frontier) {
            free(pool);
            return AEE_ENOMEMORY;
        }
    }

    ctx = dsp_search_acquire_context();
    if (!ctx) {
        free(pool);
        free(frontier);
        return AEE_EFAILED;
    }
    visited = ctx->visited;
    memset(visited, 0, VISITED_BITMAP_BYTES(g_search_num_vectors));

    if (dsp_search_quantize_query_l1(query, query_l1) != 0) {
        free(pool);
        free(frontier);
        dsp_search_release_context(ctx);
        return AEE_EFAILED;
    }

    for (i = 0; i < entry_points && i < g_search_num_vectors; i++) {
        int entry_id;
        float dist;

        entry_id = dsp_search_select_entry_id(i, &seed);
        if (VISITED_TEST(visited, entry_id))
            continue;
        VISITED_SET(visited, entry_id);
        dist = dsp_search_compute_l1_distance(
            query_l1,
            dsp_l1_code_ptr(entry_id));
        if (dsp_search_candidate_insert(pool, &pool_size_actual,
                                        pool_size, entry_id, dist))
            dsp_frontier_push(frontier, &frontier_size, entry_id, dist);
    }


    while (frontier_size > 0) {
        int current_id;
        const int32_t *neighbors;
        int j;


        if (pool_size_actual >= pool_size &&
            frontier[0].distance > pool[0].distance) {
            dsp_frontier_pop(frontier, &frontier_size);
            continue;
        }

        current_id = dsp_frontier_pop(frontier, &frontier_size);
        if (current_id < 0)
            break;

        neighbors = g_search_graph +
            (size_t)current_id * (size_t)g_search_graph_k;

        for (j = 0; j < expand_limit; j++) {
            int neighbor_id = neighbors[j];
            float dist;
            const uint8_t *db_ptr;

            if (neighbor_id < 0 || neighbor_id >= g_search_num_vectors)
                continue;
            if (VISITED_TEST(visited, neighbor_id))
                continue;
            VISITED_SET(visited, neighbor_id);

            db_ptr = dsp_l1_code_ptr(neighbor_id);
            dist = dsp_search_compute_l1_distance(query_l1, db_ptr);
            if (dsp_search_candidate_insert(pool, &pool_size_actual,
                                            pool_size, neighbor_id, dist))
                dsp_frontier_push(frontier, &frontier_size,
                                  neighbor_id, dist);
        }
    }

    dsp_heap_sort(pool, pool_size_actual);

    {
        int out = (num_out < pool_size_actual) ? num_out : pool_size_actual;
        for (i = 0; i < out; i++) {
            candidate_ids[i]   = pool[i].id;
            candidate_dists[i] = pool[i].distance;
        }
        *actual_count = out;
    }

    free(frontier);
    free(pool);
    dsp_search_release_context(ctx);
    return AEE_SUCCESS;
}

int calculator_retrieval_unload(remote_handle64 h) {
    (void)h;
    if (g_search_dma_mode) {
        return calculator_retrieval_unload_dma(h);
    }
    free(g_search_graph);    g_search_graph    = NULL;
    dsp_l1_reset_copy();
    dsp_search_free_contexts();
    g_search_num_vectors   = 0;
    g_search_l1_code_bytes = 0;
    g_search_entry_point   = DATA_GRAPH_ENTRY_POINT;
    FARF(HIGH, "retrieval_unload: done");
    return AEE_SUCCESS;
}

int calculator_retrieval_unload_dma(remote_handle64 h) {
    int i;
    (void)h;

    if (!g_search_dma_mode)
        return AEE_SUCCESS;


    if (g_search_fd_graph >= 0) {
        HAP_mmap_put(g_search_fd_graph);
        g_search_fd_graph = -1;
    }
    for (i = 0; i < g_search_l1_chunk_count; i++) {
        if (g_search_l1_fds[i] >= 0) {
            HAP_mmap_put(g_search_l1_fds[i]);
            g_search_l1_fds[i] = -1;
        }
        g_search_l1_chunks[i] = NULL;
    }
    g_search_l1_chunk_count   = 0;
    g_search_l1_chunk_vectors = 0;
    g_search_graph         = NULL;
    g_search_l1_codes      = NULL;
    g_search_dma_mode      = 0;
    dsp_search_free_contexts();
    g_search_num_vectors   = 0;
    g_search_l1_code_bytes = 0;
    g_search_entry_point   = DATA_GRAPH_ENTRY_POINT;
    FARF(HIGH, "retrieval_unload_dma: done");
    return AEE_SUCCESS;
}

int calculator_retrieval_load_dma(remote_handle64 h,
                                  int32 fd_graph,
                                  const int32 *fd_l1_codes, int fd_l1_codesLen,
                                  int32 l1_chunk_vectors,
                                  int32 num_vectors, int32 dim, int32 graph_k,
                                  int32 graph_entry_point,
                                  float l1_scale,   float l1_zero_point) {
    void *graph_vaddr = NULL;
    uint64 graph_paddr = 0;
    int ret;
    int l1_code_bytes;
    int i;
    int mapped = 0;

    (void)h;


    calculator_retrieval_unload_dma(h);
    calculator_retrieval_unload(h);

    if (fd_graph < RETRIEVAL_DMA_NO_GRAPH_FD ||
        num_vectors <= 0 || dim <= 0 || graph_k <= 0)
        return AEE_EBADPARM;
    if (!fd_l1_codes || fd_l1_codesLen < 1 || fd_l1_codesLen > MAX_L1_CHUNKS ||
        l1_chunk_vectors <= 0)
        return AEE_EBADPARM;
    l1_code_bytes = RETRIEVAL_DEMO_L1_CODE_BYTES(dim);


    if (fd_graph != RETRIEVAL_DMA_NO_GRAPH_FD) {
        ret = HAP_mmap_get(fd_graph, &graph_vaddr, &graph_paddr);
        if (ret != 0) {
            FARF(ERROR, "retrieval_load_dma: HAP_mmap_get(graph fd=%d) failed 0x%x",
                 fd_graph, ret);
            return AEE_EFAILED;
        }
    }


    for (mapped = 0; mapped < fd_l1_codesLen; mapped++) {
        void *codes_vaddr = NULL;
        uint64 codes_paddr = 0;
        ret = HAP_mmap_get(fd_l1_codes[mapped], &codes_vaddr, &codes_paddr);
        if (ret != 0) {
            FARF(ERROR, "retrieval_load_dma: HAP_mmap_get(l1 chunk %d fd=%d) failed 0x%x",
                 mapped, fd_l1_codes[mapped], ret);
            if (fd_graph >= 0)
                HAP_mmap_put(fd_graph);
            for (i = 0; i < mapped; i++) {
                HAP_mmap_put(g_search_l1_fds[i]);
                g_search_l1_chunks[i] = NULL;
                g_search_l1_fds[i]    = -1;
            }
            return AEE_EFAILED;
        }
        g_search_l1_chunks[mapped] = (uint8_t *)codes_vaddr;
        g_search_l1_fds[mapped]    = fd_l1_codes[mapped];
    }


    g_search_graph            = (int32_t *)graph_vaddr;
    g_search_l1_codes         = NULL;
    g_search_dma_mode         = 1;
    g_search_fd_graph         = fd_graph;
    g_search_l1_chunk_count   = fd_l1_codesLen;
    g_search_l1_chunk_vectors = (int)l1_chunk_vectors;

    g_search_num_vectors   = (int)num_vectors;
    g_search_dim           = (int)dim;
    g_search_l1_bits       = RETRIEVAL_DEMO_L1_BITS;
    g_search_l1_code_bytes = l1_code_bytes;
    g_search_graph_k       = (int)graph_k;
    g_search_entry_point   = (graph_entry_point >= 0 &&
                              graph_entry_point < num_vectors)
                                 ? (int)graph_entry_point
                                 : DATA_GRAPH_ENTRY_POINT;
    g_search_l1_scale      = l1_scale;
    g_search_l1_zero_point = l1_zero_point;
    g_search_ctx_count     = DSP_SEARCH_MAX_CONCURRENT;

    for (i = 0; i < g_search_ctx_count; i++) {
        g_search_ctx[i].visited = (uint32_t *)calloc(
            VISITED_BITMAP_WORDS(num_vectors), sizeof(uint32_t));
        g_search_ctx[i].in_use = 0;
        if (!g_search_ctx[i].visited) {
            int c;
            g_search_graph    = NULL;
            g_search_dma_mode = 0;
            if (fd_graph >= 0)
                HAP_mmap_put(fd_graph);
            g_search_fd_graph    = -1;
            for (c = 0; c < g_search_l1_chunk_count; c++) {
                HAP_mmap_put(g_search_l1_fds[c]);
                g_search_l1_chunks[c] = NULL;
                g_search_l1_fds[c]    = -1;
            }
            g_search_l1_chunk_count   = 0;
            g_search_l1_chunk_vectors = 0;
            dsp_search_free_contexts();
            return AEE_ENOMEMORY;
        }
    }

    FARF(HIGH, "retrieval_load_dma: n=%d dim=%d l1=int%d code_bytes=%d k=%d scale=%.6f chunks=%d vpc=%d ctx=%d (zero-copy)",
         num_vectors, dim, g_search_l1_bits, g_search_l1_code_bytes,
         graph_k, l1_scale, g_search_l1_chunk_count, g_search_l1_chunk_vectors,
         g_search_ctx_count);
    return AEE_SUCCESS;
}

int calculator_retrieval_pure_pq_unload_dma(remote_handle64 h) {
    int chunk;
    (void)h;

    if (g_pure_pq_session_live) {
        return AEE_EFAILED;
    }
    for (chunk = 0; chunk < g_pure_pq_chunk_count; chunk++) {
        if (g_pure_pq_code_fds[chunk] >= 0) {
            HAP_mmap_put(g_pure_pq_code_fds[chunk]);
        }
        g_pure_pq_code_chunks[chunk] = NULL;
        g_pure_pq_code_fds[chunk] = -1;
        g_pure_pq_chunk_counts[chunk] = 0;
    }
    free(g_pure_pq_codebook);
    g_pure_pq_codebook = NULL;
    g_pure_pq_chunk_count = 0;
    g_pure_pq_chunk_vectors = 0;
    g_pure_pq_num_vectors = 0;
    g_pure_pq_dim = 0;
    g_pure_pq_m = 0;
    g_pure_pq_ks = 0;
    g_pure_pq_subdim = 0;
    g_pure_pq_code_bytes = 0;
    memset(g_pure_pq_index_uuid, 0, sizeof(g_pure_pq_index_uuid));
    FARF(HIGH, "retrieval_pure_pq_unload_dma: done");
    return AEE_SUCCESS;
}

int calculator_retrieval_pure_pq_load_dma(
        remote_handle64 h,
        const int32 *fd_pq_codes, int fd_pq_codesLen,
        const int32 *pq_chunk_vector_counts, int pq_chunk_vector_countsLen,
        int32 pq_chunk_vectors,
        const float *pq_codebook, int pq_codebookLen,
        int32 num_vectors, int32 dim,
        int32 pq_m, int32 pq_ks, int32 pq_subdim, int32 pq_code_bytes,
        const uint8 *index_uuid, int index_uuidLen) {
    float *codebook_copy = NULL;
    size_t codebook_count;
    uint64_t total_vectors = 0u;
    int mapped = 0;
    int index;

    if (!fd_pq_codes || !pq_chunk_vector_counts || !pq_codebook ||
        !index_uuid || fd_pq_codesLen <= 0 ||
        fd_pq_codesLen > PURE_PQ_MAX_CODE_CHUNKS ||
        pq_chunk_vector_countsLen != fd_pq_codesLen ||
        pq_chunk_vectors <= 0 || num_vectors <= 0 || dim <= 0 ||
        pq_m <= 0 || (pq_ks != 16 && pq_ks != 256) || pq_subdim <= 0 ||
        pq_m > INT32_MAX / pq_subdim || pq_m * pq_subdim != dim ||
        pq_code_bytes != (pq_ks == 16 ? pq_m / 2 + pq_m % 2 : pq_m) ||
        index_uuidLen != 16 ||
        g_pure_pq_session_live) {
        return AEE_EBADPARM;
    }
    codebook_count = (size_t)pq_m * (size_t)pq_ks * (size_t)pq_subdim;
    if (codebook_count > (size_t)INT32_MAX ||
        pq_codebookLen != (int)codebook_count) {
        return AEE_EBADPARM;
    }
    for (index = 0; index < fd_pq_codesLen; index++) {
        int count = pq_chunk_vector_counts[index];
        if (fd_pq_codes[index] < 0 || count <= 0 ||
            count > pq_chunk_vectors ||
            total_vectors > (uint64_t)num_vectors - (uint64_t)count) {
            return AEE_EBADPARM;
        }
        if (index + 1 < fd_pq_codesLen && count != pq_chunk_vectors) {
            return AEE_EBADPARM;
        }
        total_vectors += (uint64_t)count;
    }
    if (total_vectors != (uint64_t)num_vectors) {
        return AEE_EBADPARM;
    }
    for (index = 0; index < pq_codebookLen; index++) {
        float value = pq_codebook[index];
        if (!(value == value) || value > 3.402823466e+38F ||
            value < -3.402823466e+38F) {
            return AEE_EBADPARM;
        }
    }

    if (calculator_retrieval_pure_pq_unload_dma(h) != AEE_SUCCESS) {
        return AEE_EFAILED;
    }
    for (mapped = 0; mapped < fd_pq_codesLen; mapped++) {
        void *address = NULL;
        uint64 physical = 0;
        int result = HAP_mmap_get(fd_pq_codes[mapped], &address, &physical);
        if (result != 0 || !address) {
            int cleanup;
            for (cleanup = 0; cleanup < mapped; cleanup++) {
                HAP_mmap_put(g_pure_pq_code_fds[cleanup]);
                g_pure_pq_code_chunks[cleanup] = NULL;
                g_pure_pq_code_fds[cleanup] = -1;
                g_pure_pq_chunk_counts[cleanup] = 0;
            }
            FARF(ERROR,
                 "retrieval_pure_pq_load_dma: map chunk=%d fd=%d failed 0x%x",
                 mapped, fd_pq_codes[mapped], result);
            return AEE_EFAILED;
        }
        g_pure_pq_code_chunks[mapped] = (uint8_t *)address;
        g_pure_pq_code_fds[mapped] = fd_pq_codes[mapped];
        g_pure_pq_chunk_counts[mapped] = pq_chunk_vector_counts[mapped];
    }

    codebook_copy = (float *)malloc(codebook_count * sizeof(float));
    if (!codebook_copy) {
        g_pure_pq_chunk_count = fd_pq_codesLen;
        calculator_retrieval_pure_pq_unload_dma(h);
        return AEE_ENOMEMORY;
    }
    memcpy(codebook_copy, pq_codebook, codebook_count * sizeof(float));
    g_pure_pq_codebook = codebook_copy;
    g_pure_pq_chunk_count = fd_pq_codesLen;
    g_pure_pq_chunk_vectors = pq_chunk_vectors;
    g_pure_pq_num_vectors = num_vectors;
    g_pure_pq_dim = dim;
    g_pure_pq_m = pq_m;
    g_pure_pq_ks = pq_ks;
    g_pure_pq_subdim = pq_subdim;
    g_pure_pq_code_bytes = pq_code_bytes;
    memcpy(g_pure_pq_index_uuid, index_uuid, 16);
    FARF(HIGH,
         "retrieval_pure_pq_load_dma: n=%d dim=%d M=%d Ks=%d chunks=%d vpc=%d",
         num_vectors, dim, pq_m, pq_ks, fd_pq_codesLen, pq_chunk_vectors);
    return AEE_SUCCESS;
}


typedef struct DspSearchState {
    DspSearchCandidate *pool;
    DspFrontierEntry   *frontier;
    int    pool_size_actual;
    int    pool_capacity;
    int    frontier_size;
    int    frontier_capacity;
    uint32_t *visited;
    DspSearchContext *ctx;
    uint8_t query_l1[RETRIEVAL_ASYNC_DEMO_MAX_DIM] __attribute__((aligned(128)));
    int    finished;
    int    expand_limit;
    int    num_out;
} DspSearchState;

static void db_shared_invalidate(ExpansionBuffer *buf) {
    if (!buf)
        return;
    qurt_mem_cache_clean((qurt_addr_t)buf,
                         (qurt_size_t)sizeof(DbSharedLayout),
                         QURT_MEM_CACHE_INVALIDATE,
                         QURT_MEM_DCACHE);
}

static void db_shared_flush_slot(ExpansionBuffer *buf) {
    if (!buf)
        return;
    qurt_mem_cache_clean((qurt_addr_t)buf,
                         (qurt_size_t)sizeof(ExpansionBuffer),
                         QURT_MEM_CACHE_FLUSH,
                         QURT_MEM_DCACHE);
}

static void db_shared_flush_stats(DbSharedStats *stats) {
    if (!stats)
        return;
    qurt_mem_cache_clean((qurt_addr_t)stats,
                         (qurt_size_t)sizeof(*stats),
                         QURT_MEM_CACHE_FLUSH,
                         QURT_MEM_DCACHE);
}

static void sqmd_shared_invalidate_control_header(SqmdSharedControl *ctrl) {
    if (!ctrl)
        return;
    qurt_mem_cache_clean((qurt_addr_t)ctrl,
                         (qurt_size_t)SQMD_CONTROL_HEADER_BYTES,
                         QURT_MEM_CACHE_INVALIDATE,
                         QURT_MEM_DCACHE);
}

static void sqmd_shared_flush_control_header(SqmdSharedControl *ctrl) {
    if (!ctrl)
        return;
    qurt_mem_cache_clean((qurt_addr_t)ctrl,
                         (qurt_size_t)SQMD_CONTROL_HEADER_BYTES,
                         QURT_MEM_CACHE_FLUSH,
                         QURT_MEM_DCACHE);
}

static void sqmd_shared_flush_initialized_data(SqmdSharedControl *ctrl) {
    if (!ctrl)
        return;

    qurt_mem_cache_clean((qurt_addr_t)ctrl,
                         (qurt_size_t)offsetof(SqmdSharedControl,
                                              startup_slot),
                         QURT_MEM_CACHE_FLUSH,
                         QURT_MEM_DCACHE);
}

static void sqmd_shared_invalidate_startup_slot(SqmdStartupSlot *slot) {
    if (!slot)
        return;
    qurt_mem_cache_clean((qurt_addr_t)slot,
                         (qurt_size_t)sizeof(*slot),
                         QURT_MEM_CACHE_INVALIDATE,
                         QURT_MEM_DCACHE);
}

static void sqmd_shared_flush_sqmd_stats(SqmdSharedStats *stats) {
    if (!stats)
        return;
    qurt_mem_cache_clean((qurt_addr_t)stats,
                         (qurt_size_t)sizeof(*stats),
                         QURT_MEM_CACHE_FLUSH,
                         QURT_MEM_DCACHE);
}

static void sqmd_shared_flush_lane_buffer(SqmdLaneBuffer *buf) {
    if (!buf)
        return;
    qurt_mem_cache_clean((qurt_addr_t)buf,
                         (qurt_size_t)sizeof(*buf),
                         QURT_MEM_CACHE_FLUSH,
                         QURT_MEM_DCACHE);
}

static void sqmd_shared_invalidate_lane_buffer(SqmdLaneBuffer *buf) {
    if (!buf)
        return;
    qurt_mem_cache_clean((qurt_addr_t)buf,
                         (qurt_size_t)sizeof(*buf),
                         QURT_MEM_CACHE_INVALIDATE,
                         QURT_MEM_DCACHE);
}


static void sqmd_shared_invalidate_lane_state(SqmdLaneBuffer *buf) {
    if (!buf)
        return;
    qurt_mem_cache_clean((qurt_addr_t)buf,
                         (qurt_size_t)SQMD_CACHE_LINE_BYTES,
                         QURT_MEM_CACHE_INVALIDATE,
                         QURT_MEM_DCACHE);
}

static void sqmd_shared_flush_lane_state(SqmdLaneBuffer *buf) {
    if (!buf)
        return;
    qurt_mem_cache_clean((qurt_addr_t)buf,
                         (qurt_size_t)SQMD_CACHE_LINE_BYTES,
                         QURT_MEM_CACHE_FLUSH,
                         QURT_MEM_DCACHE);
}


static void sqmd_shared_flush_lane_result(SqmdLaneBuffer *buf) {
    size_t end;

    if (!buf)
        return;
    end = offsetof(SqmdLaneBuffer, result) +
          (size_t)buf->result_count * sizeof(SqmdResultItem);
    end = SQMD_ALIGN_UP_BYTES(end, SQMD_CACHE_LINE_BYTES);
    if (end > sizeof(*buf))
        end = sizeof(*buf);
    qurt_mem_cache_clean((qurt_addr_t)buf,
                         (qurt_size_t)end,
                         QURT_MEM_CACHE_FLUSH,
                         QURT_MEM_DCACHE);
}


static void sqmd_shared_invalidate_sqmd_stats(SqmdSharedStats *stats) {
    if (!stats)
        return;
    qurt_mem_cache_clean((qurt_addr_t)stats,
                         (qurt_size_t)sizeof(*stats),
                         QURT_MEM_CACHE_INVALIDATE,
                         QURT_MEM_DCACHE);
}


static void sqmd_shared_invalidate_query_gen(SqmdQuerySlot *slot) {
    if (!slot)
        return;
    qurt_mem_cache_clean((qurt_addr_t)slot,
                         (qurt_size_t)SQMD_CACHE_LINE_BYTES,
                         QURT_MEM_CACHE_INVALIDATE,
                         QURT_MEM_DCACHE);
}


static void sqmd_shared_invalidate_query_slot(SqmdQuerySlot *slot) {
    if (!slot)
        return;
    qurt_mem_cache_clean((qurt_addr_t)slot,
                         (qurt_size_t)sizeof(*slot),
                         QURT_MEM_CACHE_INVALIDATE,
                         QURT_MEM_DCACHE);
}

typedef struct SqmdDspDistanceCtx {
    const uint8_t *query_l1;
    int contract_id;
} SqmdDspDistanceCtx;

typedef struct SqmdDspLaneWorkerCtx {
    SqmdSharedControl *ctrl;
    int lane_idx;
    int num_vectors;
    int work_items_cap;
    SqmdDspDistanceCtx *dist_ctx;
    int persistent;
    uint8_t *local_query_l1;
    uint64_t compute_us;
    uint64_t wait_cpu_us;
    uint64_t wait_poll_count;
} SqmdDspLaneWorkerCtx;

static int sqmd_dsp_distance_fn(const SqmdWorkItem *work,
                                float *coarse_dist,
                                void *ctx) {
    SqmdDspDistanceCtx *dist_ctx = (SqmdDspDistanceCtx *)ctx;
    const uint8_t *db_ptr;

    if (!work || !coarse_dist || !dist_ctx || !dist_ctx->query_l1)
        return -1;
    if (work->neighbor_id < 0 || work->neighbor_id >= g_search_num_vectors)
        return -1;

    db_ptr = dsp_l1_code_ptr(work->neighbor_id);
    if (dist_ctx->contract_id == SQMD_QUERY_CONTRACT_L1_ALIGNED_U8) {
        if (g_search_l1_bits != 4)
            return SQMD_STATUS_BAD_PARAM;
        *coarse_dist = retrieval_compute_u8_u4_l1_distance(
            dist_ctx->query_l1, db_ptr, g_search_dim, g_search_l1_scale);
    } else {
        *coarse_dist = dsp_search_compute_l1_distance(dist_ctx->query_l1,
                                                      db_ptr);
    }
    return 0;
}


static void sqmd_dsp_worker_record_abort(SqmdDspLaneWorkerCtx *ctx,
                                         SqmdLaneBuffer *buf,
                                         int buf_idx,
                                         int rc) {
    (void)buf_idx;
    if (!ctx || !buf)
        return;


    buf->result_count = (rc < 0) ? rc : SQMD_STATUS_BAD_PARAM;
    DB_WRITE_BARRIER();
    sqmd_shared_flush_lane_buffer(buf);
    DB_WRITE_BARRIER();
    buf->state = SQMD_BUF_DSP_ABORT;
    DB_WRITE_BARRIER();
    sqmd_shared_flush_lane_state(buf);
}

static void sqmd_dsp_lane_worker_fn(void *arg) {
    SqmdDspLaneWorkerCtx *ctx = (SqmdDspLaneWorkerCtx *)arg;
    SqmdDspDistanceCtx local_dist;
    int local_gen = -1;
    int next_consume_buf = 0;
    uint64_t wait_start_us = 0;

    if (!ctx || !ctx->ctrl ||
        ctx->lane_idx < 0 || ctx->lane_idx >= ctx->ctrl->lane_count ||
        (!ctx->persistent && !ctx->dist_ctx) ||
        (ctx->persistent && !ctx->local_query_l1)) {
        qurt_thread_exit(QURT_EFAILED);
        return;
    }
    local_dist.query_l1 = ctx->local_query_l1;
    local_dist.contract_id = SQMD_QUERY_CONTRACT_LEGACY;

    while (1) {
        SqmdLane *lane = &ctx->ctrl->lanes[ctx->lane_idx];
        int did_work = 0;

#if defined(RETRIEVAL_ENABLE_FULL_BUILD) && RETRIEVAL_ENABLE_BUILD_PAUSE
        if (ctx->persistent) build15_dsp_pause(&ctx->ctrl->build_pause,ctx->lane_idx);
#endif
        sqmd_shared_invalidate_control_header(ctx->ctrl);
        DB_READ_BARRIER();
        if (ctx->persistent) {
            if (ctx->ctrl->session_state == SQMD_SESSION_SHUTDOWN ||
                ctx->ctrl->session_state == SQMD_SESSION_ABORT ||
                ctx->ctrl->ctrl_state == SQMD_CTRL_ABORT)
                break;
        } else {
            if (ctx->ctrl->ctrl_state == SQMD_CTRL_DONE ||
                ctx->ctrl->ctrl_state == SQMD_CTRL_ABORT)
                break;
        }


        {
            SqmdLaneBuffer *buf = &lane->buf[next_consume_buf];
            SqmdDspDistanceCtx *use_ctx;
            int rc;
            int selected_turn;

            sqmd_shared_invalidate_lane_state(buf);
            DB_READ_BARRIER();
            if (buf->state != SQMD_BUF_CPU_READY)
                goto no_lane_work;


            sqmd_shared_invalidate_lane_buffer(buf);
            DB_READ_BARRIER();
            if (buf->state != SQMD_BUF_CPU_READY)
                continue;
            selected_turn = buf->turn_id;

            if (ctx->persistent) {

                sqmd_shared_invalidate_query_gen(&ctx->ctrl->query_slot);
                DB_READ_BARRIER();
                if (ctx->ctrl->query_slot.query_gen != local_gen) {
                    sqmd_shared_invalidate_query_slot(&ctx->ctrl->query_slot);
                    DB_READ_BARRIER();
                    if (ctx->ctrl->query_slot.contract_id ==
                            SQMD_QUERY_CONTRACT_L1_ALIGNED_U8) {
                        if (ctx->ctrl->query_slot.query_dim != g_search_dim ||
                            g_search_l1_bits != 4) {
                            sqmd_dsp_worker_record_abort(ctx, buf,
                                                         next_consume_buf,
                                                         SQMD_STATUS_BAD_PARAM);
                            qurt_thread_exit(QURT_EFAILED);
                            return;
                        }
                        memcpy(ctx->local_query_l1,
                               ctx->ctrl->query_slot.query_u8,
                               (size_t)g_search_dim);
                        local_dist.contract_id =
                            SQMD_QUERY_CONTRACT_L1_ALIGNED_U8;
                    } else if (ctx->ctrl->query_slot.contract_id !=
                                   SQMD_QUERY_CONTRACT_LEGACY ||
                               dsp_search_quantize_query_l1(
                                   ctx->ctrl->query_slot.query,
                                   ctx->local_query_l1) != 0) {
                        sqmd_dsp_worker_record_abort(ctx, buf,
                                                     next_consume_buf,
                                                     SQMD_STATUS_BAD_PARAM);
                        qurt_thread_exit(QURT_EFAILED);
                        return;
                    } else {
                        local_dist.contract_id = SQMD_QUERY_CONTRACT_LEGACY;
                    }
                    local_gen = ctx->ctrl->query_slot.query_gen;
                }
                use_ctx = &local_dist;
            } else {
                use_ctx = ctx->dist_ctx;
            }

            if (wait_start_us != 0) {
                ctx->wait_cpu_us += HAP_perf_get_time_us() - wait_start_us;
                wait_start_us = 0;
            }
            lane->dsp_active_buf = next_consume_buf;
            {
                uint64_t compute_t0 = HAP_perf_get_time_us();

                rc = sqmd_process_lane_buffer_payload_with_cap(
                    buf,
                    selected_turn,
                    ctx->num_vectors,
                    ctx->work_items_cap,
                    sqmd_dsp_distance_fn,
                    use_ctx);
                ctx->compute_us += HAP_perf_get_time_us() - compute_t0;
            }
            lane->dsp_active_buf = -1;
            if (rc != SQMD_STATUS_OK) {
                FARF(ERROR,
                     "SQMD lane %d buffer %d process failed rc=%d state=%d turn=%d work=%d cap=%d",
                     ctx->lane_idx, next_consume_buf, rc, buf->state,
                     buf->turn_id,
                     buf->work_count, ctx->work_items_cap);
                sqmd_dsp_worker_record_abort(ctx, buf, next_consume_buf, rc);
                qurt_thread_exit(QURT_EFAILED);
                return;
            }
            if (ctx->persistent)
                sqmd_shared_flush_lane_result(buf);
            else
                sqmd_shared_flush_lane_buffer(buf);


            DB_WRITE_BARRIER();
            buf->state = SQMD_BUF_DSP_DONE;
            DB_WRITE_BARRIER();
            sqmd_shared_flush_lane_state(buf);
            next_consume_buf =
                (next_consume_buf + 1) % SQMD_BUFFERS_PER_LANE;
            did_work = 1;
        }

no_lane_work:
        if (!did_work) {
            if (wait_start_us == 0)
                wait_start_us = HAP_perf_get_time_us();
            ctx->wait_poll_count++;
            if ((ctx->wait_poll_count & 255u) == 0u)
                qurt_timer_sleep(1);
        }
    }

    if (wait_start_us != 0)
        ctx->wait_cpu_us += HAP_perf_get_time_us() - wait_start_us;
    qurt_thread_exit(QURT_EOK);
}


static int db_search_reserve_frontier(DspSearchState *st, int additional) {
    size_t required, capacity;
    DspFrontierEntry *frontier;

    if (additional < 0 || st->frontier_size < 0 ||
        st->frontier_size > g_search_num_vectors ||
        additional > g_search_num_vectors - st->frontier_size)
        return -1;
    required = (size_t)st->frontier_size + (size_t)additional;
    if (required <= (size_t)st->frontier_capacity)
        return 0;
    capacity = (size_t)st->frontier_capacity * 2u;
    if (capacity < required)
        capacity = required;
    if (capacity > (size_t)g_search_num_vectors)
        capacity = (size_t)g_search_num_vectors;
    if (capacity > SIZE_MAX / sizeof(*frontier))
        return -1;
    frontier = (DspFrontierEntry *)realloc(st->frontier,
                                          capacity * sizeof(*frontier));
    if (!frontier)
        return -1;
    st->frontier = frontier;
    st->frontier_capacity = (int)capacity;
    return 0;
}


static int db_search_state_init(DspSearchState *st,
                                const float *query, int queryLen,
                                int32 entry_points, int32 expand_count,
                                int32 pool_size, int32 num_out) {
    int i;
    unsigned int seed = DSP_SEARCH_ENTRY_SEED;
    int expand_limit;

    memset(st, 0, sizeof(*st));

    if (!query || queryLen < g_search_dim || entry_points <= 0 ||
        pool_size <= 0 || num_out <= 0 || g_search_num_vectors <= 0)
        return -1;

    expand_limit = (expand_count < g_search_graph_k)
                       ? expand_count : g_search_graph_k;
    if (expand_limit <= 0)
        return -1;
    if (pool_size > DSP_SEARCH_MAX_POOL)
        pool_size = DSP_SEARCH_MAX_POOL;

    st->pool_capacity = pool_size;
    st->expand_limit  = expand_limit;
    st->num_out       = num_out;

    st->pool = (DspSearchCandidate *)malloc(
        (size_t)DSP_SEARCH_MAX_POOL * sizeof(DspSearchCandidate));
    if (!st->pool)
        return -1;

    {
        size_t cap = (size_t)entry_points +
                     (size_t)pool_size * (size_t)expand_limit;
        if (cap > (size_t)g_search_num_vectors)
            cap = (size_t)g_search_num_vectors;
        st->frontier_capacity = (int)cap;
        st->frontier = (DspFrontierEntry *)malloc(
            cap * sizeof(DspFrontierEntry));
        if (!st->frontier) {
            free(st->pool); st->pool = NULL;
            return -1;
        }
    }

    st->ctx = dsp_search_acquire_context();
    if (!st->ctx) {
        free(st->frontier); st->frontier = NULL;
        free(st->pool);     st->pool = NULL;
        return -1;
    }
    st->visited = st->ctx->visited;
    memset(st->visited, 0, VISITED_BITMAP_BYTES(g_search_num_vectors));

    if (dsp_search_quantize_query_l1(query, st->query_l1) != 0) {
        dsp_search_release_context(st->ctx); st->ctx = NULL;
        free(st->frontier); st->frontier = NULL;
        free(st->pool);     st->pool = NULL;
        return -1;
    }


    for (i = 0; i < entry_points && i < g_search_num_vectors; i++) {
        int entry_id;
        float dist;

        entry_id = dsp_search_select_entry_id(i, &seed);
        if (VISITED_TEST(st->visited, entry_id))
            continue;
        VISITED_SET(st->visited, entry_id);

        dist = dsp_search_compute_l1_distance(
            st->query_l1,
            dsp_l1_code_ptr(entry_id));

        if (dsp_search_candidate_insert(st->pool, &st->pool_size_actual,
                                        st->pool_capacity, entry_id, dist))
            dsp_frontier_push(st->frontier, &st->frontier_size,
                              entry_id, dist);
    }

    return 0;
}


static int db_search_expand_one_step(DspSearchState *st,
                                     ExpansionBuffer *buf,
                                     DbSharedStats *stats) {
    int current_id = -1;
    int count = 0;
    int out_count = 0;
    const int32_t *neighbors;
    int neighbor_ids[DB_MAX_EXPAND];
    float coarse_dists[DB_MAX_EXPAND];
    int j;


    while (st->frontier_size > 0) {
        if (st->pool_size_actual >= st->pool_capacity &&
            st->frontier[0].distance > st->pool[0].distance) {
            dsp_frontier_pop(st->frontier, &st->frontier_size);
            continue;
        }
        current_id = dsp_frontier_pop(st->frontier,
                                      &st->frontier_size);
        break;
    }


    if (current_id < 0)
        return 0;


    neighbors = g_search_graph +
        (size_t)current_id * (size_t)g_search_graph_k;

    for (j = 0; j < st->expand_limit; j++) {
        int neighbor_id = neighbors[j];
        const uint8_t *db_ptr;
        float dist;

        if (neighbor_id < 0 || neighbor_id >= g_search_num_vectors)
            continue;
        if (VISITED_TEST(st->visited, neighbor_id))
            continue;
        VISITED_SET(st->visited, neighbor_id);

        db_ptr = dsp_l1_code_ptr(neighbor_id);
        dist = dsp_search_compute_l1_distance(st->query_l1, db_ptr);

        neighbor_ids[count] = neighbor_id;
        coarse_dists[count] = dist;
        count++;
    }

    for (j = 0; j < count; j++) {
        buf->neighbor_ids[out_count] = neighbor_ids[j];
        buf->coarse_dists[out_count] = coarse_dists[j];
        out_count++;
    }

    buf->count = out_count;
    (void)stats;
    return 1;
}


static int db_search_apply_corrections(DspSearchState *st,
                                        ExpansionBuffer *buf) {
    int i;

    if (buf->count < 0 || buf->count > DB_MAX_EXPAND ||
        db_search_reserve_frontier(st, buf->count) != 0)
        return -1;
    for (i = 0; i < buf->count; i++) {
        if (dsp_search_candidate_insert(
                st->pool, &st->pool_size_actual,
                st->pool_capacity,
                buf->neighbor_ids[i],
                buf->corrected_dists[i]))
            dsp_frontier_push(st->frontier, &st->frontier_size,
                              buf->neighbor_ids[i],
                              buf->corrected_dists[i]);
    }
    return 0;
}

static void db_search_state_cleanup(DspSearchState *st) {
    if (!st)
        return;
    free(st->frontier); st->frontier = NULL;
    free(st->pool);     st->pool = NULL;
    if (st->ctx) {
        dsp_search_release_context(st->ctx);
        st->ctx = NULL;
    }
}


int calculator_retrieval_search_db(
        remote_handle64 h,
        const float *query_a, int query_aLen,
        const float *query_b, int query_bLen,
        int32 fd_expansion_buf,
        int32 entry_points, int32 expand_count,
        int32 pool_size,    int32 num_out,
        int   *candidate_ids_a,   int candidate_ids_aLen,
        float *candidate_dists_a, int candidate_dists_aLen,
        int32 *actual_count_a,
        int   *candidate_ids_b,   int candidate_ids_bLen,
        float *candidate_dists_b, int candidate_dists_bLen,
        int32 *actual_count_b) {

    DspSearchState st[2];
    DbSharedLayout *shared = NULL;
    ExpansionBuffer *buf = NULL;
    DbSharedStats *stats = NULL;
    uint64 paddr = 0;
    int out, i;
    int ret = AEE_SUCCESS;
    uint64_t wait_cpu_done_us = 0;
    uint64_t wait_cpu_done_start_us = 0;
    uint64_t invalidate_us = 0;
    uint64_t wait_poll_count = 0;

    (void)h;

    memset(st, 0, sizeof(st));

    if (!actual_count_a || !actual_count_b)
        return AEE_EBADPARM;
    *actual_count_a = 0;
    *actual_count_b = 0;

    ret = HAP_mmap_get(fd_expansion_buf, (void **)&shared, &paddr);
    if (ret != 0) {
        FARF(ERROR, "retrieval_search_db: HAP_mmap_get(fd=%d) failed 0x%x",
             fd_expansion_buf, ret);
        return AEE_EFAILED;
    }

    if (!shared) {
        ret = AEE_EFAILED;
        goto cleanup;
    }
    buf = shared->buf;
    stats = &shared->stats;
    if (!g_search_graph || !g_search_l1_chunks[0] || g_search_ctx_count <= 0) {
        ret = AEE_EFAILED;
        goto cleanup;
    }
    if (candidate_ids_aLen < num_out || candidate_dists_aLen < num_out ||
        candidate_ids_bLen < num_out || candidate_dists_bLen < num_out) {
        ret = AEE_EBADPARM;
        goto cleanup;
    }


    if (db_search_state_init(&st[0], query_a, query_aLen,
                             entry_points, expand_count,
                             pool_size, num_out) != 0) {
        ret = AEE_ENOMEMORY;
        goto cleanup;
    }

    if (db_search_state_init(&st[1], query_b, query_bLen,
                             entry_points, expand_count,
                             pool_size, num_out) != 0) {
        ret = AEE_ENOMEMORY;
        goto cleanup;
    }


    buf[0].state    = DB_STATE_EMPTY;
    buf[0].finished = 0;
    buf[0].count    = 0;
    buf[1].state    = DB_STATE_EMPTY;
    buf[1].finished = 0;
    buf[1].count    = 0;
    if (stats) {
        stats->dsp_wait_cpu_done_us = 0;
        stats->dsp_invalidate_us = 0;
        stats->dsp_wait_poll_count = 0;
    }
    DB_WRITE_BARRIER();


    while (!st[0].finished || !st[1].finished) {
        int cur = -1;
        uint64_t inv_t0 = HAP_perf_get_time_us();

        db_shared_invalidate(buf);
        invalidate_us += HAP_perf_get_time_us() - inv_t0;


        for (i = 0; i < 2; i++) {
            if (st[i].finished)
                continue;

            if (buf[i].state == DB_STATE_CPU_DONE) {
                DB_READ_BARRIER();
                if (db_search_apply_corrections(&st[i], &buf[i]) != 0) {
                    ret = AEE_ENOMEMORY;
                    goto cleanup;
                }
                buf[i].state = DB_STATE_EMPTY;
                DB_WRITE_BARRIER();
                db_shared_flush_slot(&buf[i]);
                cur = i;
                break;
            }
            if (buf[i].state == DB_STATE_EMPTY) {
                cur = i;
                break;
            }
        }

        if (cur < 0) {
            if (wait_cpu_done_start_us == 0) {
                wait_cpu_done_start_us = HAP_perf_get_time_us();
            }
            wait_poll_count++;
            continue;
        }

        if (wait_cpu_done_start_us != 0) {
            wait_cpu_done_us += HAP_perf_get_time_us() - wait_cpu_done_start_us;
            wait_cpu_done_start_us = 0;
        }

        if (db_search_expand_one_step(&st[cur], &buf[cur], stats)) {
            DB_WRITE_BARRIER();
            buf[cur].state = DB_STATE_DSP_DONE;
            db_shared_flush_slot(&buf[cur]);
        } else {

            st[cur].finished  = 1;
            buf[cur].finished = 1;
            buf[cur].count    = 0;
            DB_WRITE_BARRIER();
            buf[cur].state = DB_STATE_DSP_DONE;
            db_shared_flush_slot(&buf[cur]);
        }
    }


    dsp_heap_sort(st[0].pool, st[0].pool_size_actual);
    out = (num_out < st[0].pool_size_actual)
              ? num_out : st[0].pool_size_actual;
    for (i = 0; i < out; i++) {
        candidate_ids_a[i]   = st[0].pool[i].id;
        candidate_dists_a[i] = st[0].pool[i].distance;
    }
    *actual_count_a = out;

    dsp_heap_sort(st[1].pool, st[1].pool_size_actual);
    out = (num_out < st[1].pool_size_actual)
              ? num_out : st[1].pool_size_actual;
    for (i = 0; i < out; i++) {
        candidate_ids_b[i]   = st[1].pool[i].id;
        candidate_dists_b[i] = st[1].pool[i].distance;
    }
    *actual_count_b = out;

    if (wait_cpu_done_start_us != 0) {
        wait_cpu_done_us += HAP_perf_get_time_us() - wait_cpu_done_start_us;
        wait_cpu_done_start_us = 0;
    }
    if (stats) {
        stats->dsp_wait_cpu_done_us = wait_cpu_done_us;
        stats->dsp_invalidate_us = invalidate_us;
        stats->dsp_wait_poll_count = wait_poll_count;
        DB_WRITE_BARRIER();
        db_shared_flush_stats(stats);
    }

cleanup:
    db_search_state_cleanup(&st[0]);
    db_search_state_cleanup(&st[1]);
    HAP_mmap_put(fd_expansion_buf);
    return ret;
}


int calculator_retrieval_expand_sqmd(remote_handle64 h,
                                     const float *query,
                                     int queryLen,
                                     int32 query_dim,
                                     int32 fd_sqmd_ctrl,
                                     int32 lane_count,
                                     int32 work_items_cap) {
    SqmdSharedControl *ctrl = NULL;
    SqmdDspDistanceCtx dist_ctx;
    uint64 paddr = 0;
    uint8_t query_l1[RETRIEVAL_ASYNC_DEMO_MAX_DIM] __attribute__((aligned(128)));
    int ret = AEE_SUCCESS;
    int lane;
    int buf_idx;
    qurt_thread_t workers[SQMD_DSP_LANES];
    void *worker_stacks[SQMD_DSP_LANES];
    SqmdDspLaneWorkerCtx worker_ctx[SQMD_DSP_LANES];
    uint64_t invalidate_us = 0;
    uint64_t abort_count = 0;

    (void)h;
    memset(workers, 0, sizeof(workers));
    memset(worker_stacks, 0, sizeof(worker_stacks));
    memset(worker_ctx, 0, sizeof(worker_ctx));

    if (!query || queryLen != query_dim || query_dim <= 0 ||
        query_dim > RETRIEVAL_ASYNC_DEMO_MAX_DIM ||
        fd_sqmd_ctrl < 0 ||
        lane_count <= 0 || lane_count > SQMD_DSP_LANES ||
        work_items_cap <= 0 || work_items_cap > SQMD_MAX_WORK_ITEMS) {
        FARF(ERROR,
             "retrieval_expand_sqmd: bad params query=%p queryLen=%d query_dim=%d fd=%d lanes=%d cap=%d max_lanes=%d max_cap=%d",
             query, queryLen, query_dim, fd_sqmd_ctrl, lane_count,
             work_items_cap, SQMD_DSP_LANES, SQMD_MAX_WORK_ITEMS);
        return AEE_EBADPARM;
    }

    if (!g_search_l1_chunks[0] || g_search_ctx_count <= 0) {
        FARF(ERROR,
             "retrieval_expand_sqmd: L1 index not loaded codes=%p ctx=%d",
             g_search_l1_chunks[0], g_search_ctx_count);
        return AEE_EFAILED;
    }

    if (dsp_search_quantize_query_l1(query, query_l1) != 0) {
        FARF(ERROR,
             "retrieval_expand_sqmd: query quantize failed dim=%d l1_bits=%d scale=%.6f",
             g_search_dim, g_search_l1_bits, g_search_l1_scale);
        return AEE_EBADPARM;
    }
    dist_ctx.query_l1 = query_l1;
    dist_ctx.contract_id = SQMD_QUERY_CONTRACT_LEGACY;

    ret = HAP_mmap_get(fd_sqmd_ctrl, (void **)&ctrl, &paddr);
    if (ret != 0) {
        FARF(ERROR, "retrieval_expand_sqmd: HAP_mmap_get(fd=%d) failed 0x%x",
             fd_sqmd_ctrl, ret);
        return AEE_EFAILED;
    }

    if (!ctrl) {
        ret = AEE_EFAILED;
        goto cleanup;
    }

    ctrl->ctrl_state = SQMD_CTRL_PREPARING;
    ctrl->lane_count = lane_count;
    ctrl->turn_id = 0;
    memset((void *)&ctrl->stats, 0, sizeof(ctrl->stats));
    for (lane = 0; lane < SQMD_DSP_LANES; lane++) {
        for (buf_idx = 0; buf_idx < SQMD_BUFFERS_PER_LANE; buf_idx++) {
            ctrl->lanes[lane].buf[buf_idx].state = SQMD_BUF_EMPTY;
            ctrl->lanes[lane].buf[buf_idx].turn_id = 0;
            ctrl->lanes[lane].buf[buf_idx].work_count = 0;
            ctrl->lanes[lane].buf[buf_idx].result_count = 0;
        }
        ctrl->lanes[lane].dsp_active_buf = -1;
        ctrl->lanes[lane].cpu_fill_buf = 0;
    }
    DB_WRITE_BARRIER();
    sqmd_shared_flush_initialized_data(ctrl);
    for (lane = 0; lane < lane_count; lane++) {
        qurt_thread_attr_t attr;
        char name[QURT_THREAD_ATTR_NAME_MAXLEN];

        worker_stacks[lane] = malloc(RETRIEVAL_ASYNC_DEMO_STACK_SIZE);
        if (!worker_stacks[lane]) {
            FARF(ERROR,
                 "retrieval_expand_sqmd: malloc lane stack failed lane=%d size=%d",
                 lane, RETRIEVAL_ASYNC_DEMO_STACK_SIZE);
            ret = AEE_ENOMEMORY;
            abort_count++;
            DB_WRITE_BARRIER();
            ctrl->ctrl_state = SQMD_CTRL_ABORT;
            DB_WRITE_BARRIER();
            sqmd_shared_flush_control_header(ctrl);
            goto cleanup;
        }

        worker_ctx[lane].ctrl = ctrl;
        worker_ctx[lane].lane_idx = lane;
        worker_ctx[lane].num_vectors = g_search_num_vectors;
        worker_ctx[lane].work_items_cap = work_items_cap;
        worker_ctx[lane].dist_ctx = &dist_ctx;

        qurt_thread_attr_init(&attr);
        snprintf(name, sizeof(name), "sqmd_lane_%d", lane);
        qurt_thread_attr_set_name(&attr, name);
        qurt_thread_attr_set_stack_addr(&attr, worker_stacks[lane]);
        qurt_thread_attr_set_stack_size(&attr, RETRIEVAL_ASYNC_DEMO_STACK_SIZE);
        qurt_thread_attr_set_priority(&attr,
                                      QURT_THREAD_ATTR_PRIORITY_DEFAULT / 2);
        ret = qurt_thread_create(&workers[lane],
                                 &attr,
                                 sqmd_dsp_lane_worker_fn,
                                 &worker_ctx[lane]);
        if (ret != QURT_EOK) {
            FARF(ERROR,
                 "retrieval_expand_sqmd: qurt_thread_create lane=%d failed ret=0x%x stack=%p size=%d",
                 lane, ret, worker_stacks[lane],
                 RETRIEVAL_ASYNC_DEMO_STACK_SIZE);
            ret = AEE_EFAILED;
            abort_count++;
            DB_WRITE_BARRIER();
            ctrl->ctrl_state = SQMD_CTRL_ABORT;
            DB_WRITE_BARRIER();
            sqmd_shared_flush_control_header(ctrl);
            goto cleanup;
        }
    }


    sqmd_shared_invalidate_startup_slot(&ctrl->startup_slot);
    DB_READ_BARRIER();
    if (ctrl->startup_slot.cancel) {
        ret = AEE_EFAILED;
        abort_count++;
        ctrl->ctrl_state = SQMD_CTRL_ABORT;
        DB_WRITE_BARRIER();
        sqmd_shared_flush_control_header(ctrl);
        goto cleanup;
    }
    DB_WRITE_BARRIER();
    ctrl->ctrl_state = SQMD_CTRL_RUNNING;
    DB_WRITE_BARRIER();
    sqmd_shared_flush_control_header(ctrl);

    while (1) {
        uint64_t inv_t0 = HAP_perf_get_time_us();

        sqmd_shared_invalidate_control_header(ctrl);
        invalidate_us += HAP_perf_get_time_us() - inv_t0;
        DB_READ_BARRIER();
        if (!ctrl->startup_slot.acknowledged) {
            sqmd_shared_invalidate_startup_slot(&ctrl->startup_slot);
            DB_READ_BARRIER();
            if (ctrl->startup_slot.cancel) {
                ret = AEE_EFAILED;
                abort_count++;
                ctrl->ctrl_state = SQMD_CTRL_ABORT;
                DB_WRITE_BARRIER();
                sqmd_shared_flush_control_header(ctrl);
                break;
            }
        }
        if (ctrl->ctrl_state == SQMD_CTRL_DONE)
            break;
        if (ctrl->ctrl_state == SQMD_CTRL_ABORT) {
            ret = AEE_EFAILED;
            break;
        }
    }

cleanup:
    for (lane = 0; lane < lane_count; lane++) {
        if (workers[lane] != 0) {
            int status = 0;

            qurt_thread_join(workers[lane], &status);
            if (status != QURT_EOK && ret == AEE_SUCCESS)
                ret = AEE_EFAILED;
        }
    }
    if (ctrl) {

        sqmd_shared_invalidate_sqmd_stats(&ctrl->stats);
        DB_READ_BARRIER();
        for (lane = 0; lane < lane_count; lane++) {
            ctrl->stats.sqmd_dsp_compute_us += worker_ctx[lane].compute_us;
            ctrl->stats.sqmd_dsp_wait_cpu_us += worker_ctx[lane].wait_cpu_us;
            ctrl->stats.sqmd_dsp_wait_poll_count +=
                worker_ctx[lane].wait_poll_count;
        }
        ctrl->stats.sqmd_dsp_invalidate_us += invalidate_us;
        ctrl->stats.sqmd_abort_count += abort_count;
        DB_WRITE_BARRIER();
        sqmd_shared_flush_sqmd_stats(&ctrl->stats);
        sqmd_shared_flush_control_header(ctrl);
    }
    for (lane = 0; lane < SQMD_DSP_LANES; lane++) {
        free(worker_stacks[lane]);
    }
    HAP_mmap_put(fd_sqmd_ctrl);
    return ret;
}


int calculator_retrieval_expand_sqmd_run(remote_handle64 h,
                                         int32 fd_sqmd_ctrl,
                                         int32 lane_count,
                                         int32 work_items_cap) {
    SqmdSharedControl *ctrl = NULL;
    uint64 paddr = 0;
    int ret = AEE_SUCCESS;
    int lane;
    int buf_idx;
    qurt_thread_t workers[SQMD_DSP_LANES];
    void *worker_stacks[SQMD_DSP_LANES];
    uint8_t *worker_query_l1[SQMD_DSP_LANES];
    SqmdDspLaneWorkerCtx worker_ctx[SQMD_DSP_LANES];
    uint64_t invalidate_us = 0;
    uint64_t abort_count = 0;

    (void)h;
    memset(workers, 0, sizeof(workers));
    memset(worker_stacks, 0, sizeof(worker_stacks));
    memset(worker_query_l1, 0, sizeof(worker_query_l1));
    memset(worker_ctx, 0, sizeof(worker_ctx));

    if (fd_sqmd_ctrl < 0 ||
        lane_count <= 0 || lane_count > SQMD_DSP_LANES ||
        work_items_cap <= 0 || work_items_cap > SQMD_MAX_WORK_ITEMS) {
        FARF(ERROR,
             "retrieval_expand_sqmd_run: bad params fd=%d lanes=%d cap=%d",
             fd_sqmd_ctrl, lane_count, work_items_cap);
        return AEE_EBADPARM;
    }
    if (!g_search_l1_chunks[0] || g_search_ctx_count <= 0) {
        FARF(ERROR,
             "retrieval_expand_sqmd_run: L1 index not loaded codes=%p ctx=%d",
             g_search_l1_chunks[0], g_search_ctx_count);
        return AEE_EFAILED;
    }

    ret = HAP_mmap_get(fd_sqmd_ctrl, (void **)&ctrl, &paddr);
    if (ret != 0) {
        FARF(ERROR, "retrieval_expand_sqmd_run: HAP_mmap_get(fd=%d) failed 0x%x",
             fd_sqmd_ctrl, ret);
        return AEE_EFAILED;
    }
    if (!ctrl) {
        ret = AEE_EFAILED;
        goto cleanup;
    }


    ctrl->session_state = SQMD_SESSION_IDLE;
    ctrl->ctrl_state = SQMD_CTRL_DONE;
    ctrl->lane_count = lane_count;
    ctrl->turn_id = 0;
    memset((void *)&ctrl->stats, 0, sizeof(ctrl->stats));
    ctrl->query_slot.query_gen = 0;
    for (lane = 0; lane < SQMD_DSP_LANES; lane++) {
        for (buf_idx = 0; buf_idx < SQMD_BUFFERS_PER_LANE; buf_idx++) {
            ctrl->lanes[lane].buf[buf_idx].state = SQMD_BUF_EMPTY;
            ctrl->lanes[lane].buf[buf_idx].turn_id = 0;
            ctrl->lanes[lane].buf[buf_idx].work_count = 0;
            ctrl->lanes[lane].buf[buf_idx].result_count = 0;
        }
        ctrl->lanes[lane].dsp_active_buf = -1;
        ctrl->lanes[lane].cpu_fill_buf = 0;
    }
    DB_WRITE_BARRIER();
    sqmd_shared_flush_initialized_data(ctrl);

    for (lane = 0; lane < lane_count; lane++) {
        qurt_thread_attr_t attr;
        char name[QURT_THREAD_ATTR_NAME_MAXLEN];

        worker_stacks[lane] = malloc(RETRIEVAL_ASYNC_DEMO_STACK_SIZE);
        worker_query_l1[lane] =
            (uint8_t *)malloc(RETRIEVAL_ASYNC_DEMO_MAX_DIM);
        if (!worker_stacks[lane] || !worker_query_l1[lane]) {
            FARF(ERROR,
                 "retrieval_expand_sqmd_run: malloc lane=%d failed stack=%p q=%p",
                 lane, worker_stacks[lane], worker_query_l1[lane]);
            ret = AEE_ENOMEMORY;
            abort_count++;
            DB_WRITE_BARRIER();
            ctrl->session_state = SQMD_SESSION_ABORT;
            DB_WRITE_BARRIER();
            sqmd_shared_flush_control_header(ctrl);
            goto cleanup;
        }

        worker_ctx[lane].ctrl = ctrl;
        worker_ctx[lane].lane_idx = lane;
        worker_ctx[lane].num_vectors = g_search_num_vectors;
        worker_ctx[lane].work_items_cap = work_items_cap;
        worker_ctx[lane].dist_ctx = NULL;
        worker_ctx[lane].persistent = 1;
        worker_ctx[lane].local_query_l1 = worker_query_l1[lane];

        qurt_thread_attr_init(&attr);
        snprintf(name, sizeof(name), "sqmd_run_%d", lane);
        qurt_thread_attr_set_name(&attr, name);
        qurt_thread_attr_set_stack_addr(&attr, worker_stacks[lane]);
        qurt_thread_attr_set_stack_size(&attr, RETRIEVAL_ASYNC_DEMO_STACK_SIZE);
        qurt_thread_attr_set_priority(&attr,
                                      QURT_THREAD_ATTR_PRIORITY_DEFAULT / 2);
        ret = qurt_thread_create(&workers[lane],
                                 &attr,
                                 sqmd_dsp_lane_worker_fn,
                                 &worker_ctx[lane]);
        if (ret != QURT_EOK) {
            FARF(ERROR,
                 "retrieval_expand_sqmd_run: qurt_thread_create lane=%d failed ret=0x%x",
                 lane, ret);
            ret = AEE_EFAILED;
            abort_count++;
            DB_WRITE_BARRIER();
            ctrl->session_state = SQMD_SESSION_ABORT;
            DB_WRITE_BARRIER();
            sqmd_shared_flush_control_header(ctrl);
            goto cleanup;
        }
    }


    sqmd_shared_invalidate_startup_slot(&ctrl->startup_slot);
    DB_READ_BARRIER();
    if (ctrl->startup_slot.cancel) {
        ret = AEE_EFAILED;
        abort_count++;
        ctrl->session_state = SQMD_SESSION_ABORT;
        DB_WRITE_BARRIER();
        sqmd_shared_flush_control_header(ctrl);
        goto cleanup;
    }
    DB_WRITE_BARRIER();
    ctrl->session_state = SQMD_SESSION_RUNNING;
    DB_WRITE_BARRIER();
    sqmd_shared_flush_control_header(ctrl);


    while (1) {
#if defined(RETRIEVAL_ENABLE_FULL_BUILD) && RETRIEVAL_ENABLE_BUILD_PAUSE
        build15_dsp_pause(&ctrl->build_pause,RETRIEVAL_BUILD_PAUSE_SUPERVISOR);
#endif
        uint64_t inv_t0 = HAP_perf_get_time_us();

        sqmd_shared_invalidate_control_header(ctrl);
        invalidate_us += HAP_perf_get_time_us() - inv_t0;
        DB_READ_BARRIER();
        if (!ctrl->startup_slot.acknowledged) {
            sqmd_shared_invalidate_startup_slot(&ctrl->startup_slot);
            DB_READ_BARRIER();
            if (ctrl->startup_slot.cancel) {
                ret = AEE_EFAILED;
                abort_count++;
                ctrl->session_state = SQMD_SESSION_ABORT;
                DB_WRITE_BARRIER();
                sqmd_shared_flush_control_header(ctrl);
                break;
            }
        }
        if (ctrl->session_state == SQMD_SESSION_SHUTDOWN)
            break;
        if (ctrl->session_state == SQMD_SESSION_ABORT) {
            ret = AEE_EFAILED;
            break;
        }
    }

cleanup:
    for (lane = 0; lane < lane_count; lane++) {
        if (workers[lane] != 0) {
            int status = 0;

            qurt_thread_join(workers[lane], &status);
            if (status != QURT_EOK && ret == AEE_SUCCESS)
                ret = AEE_EFAILED;
        }
    }
    if (ctrl) {

        sqmd_shared_invalidate_sqmd_stats(&ctrl->stats);
        DB_READ_BARRIER();
        for (lane = 0; lane < lane_count; lane++) {
            ctrl->stats.sqmd_dsp_compute_us += worker_ctx[lane].compute_us;
            ctrl->stats.sqmd_dsp_wait_cpu_us += worker_ctx[lane].wait_cpu_us;
            ctrl->stats.sqmd_dsp_wait_poll_count +=
                worker_ctx[lane].wait_poll_count;
        }
        ctrl->stats.sqmd_dsp_invalidate_us += invalidate_us;
        ctrl->stats.sqmd_abort_count += abort_count;
        DB_WRITE_BARRIER();
        sqmd_shared_flush_sqmd_stats(&ctrl->stats);
        sqmd_shared_flush_control_header(ctrl);
    }
    for (lane = 0; lane < SQMD_DSP_LANES; lane++) {
        free(worker_stacks[lane]);
        free(worker_query_l1[lane]);
    }
    HAP_mmap_put(fd_sqmd_ctrl);
    return ret;
}

typedef struct PurePqDspLaneWorkerContext {
    PurePqSqmdSharedControl *control;
    PurePqQueryLutHeader *lut_header;
    const float *shared_lut;
    struct PurePqDspLutCache *lut_cache;
    void *hvx_gather_buffer;
    int lane_index;
    int work_items_cap;
    uint64_t compute_us;
    uint64_t wait_cpu_us;
    uint64_t wait_polls;
    uint64_t abort_count;
} PurePqDspLaneWorkerContext;

typedef struct __attribute__((aligned(PURE_PQ_SQMD_CACHE_LINE_BYTES)))
PurePqDspLutCache {
    float *values;
    size_t value_count;
    size_t lut_bytes_aligned;
    void *gather_buffers;
    int use_hvx;
    volatile int query_gen;
    volatile int copy_lock;
} PurePqDspLutCache;

static void pure_pq_dsp_invalidate_control(
        PurePqSqmdSharedControl *control) {
    qurt_mem_cache_clean((qurt_addr_t)control,
                         (qurt_size_t)PURE_PQ_SQMD_CACHE_LINE_BYTES,
                         QURT_MEM_CACHE_INVALIDATE,
                         QURT_MEM_DCACHE);
}

static void pure_pq_dsp_flush_control(
        PurePqSqmdSharedControl *control) {
    qurt_mem_cache_clean((qurt_addr_t)control,
                         (qurt_size_t)PURE_PQ_SQMD_CACHE_LINE_BYTES,
                         QURT_MEM_CACHE_FLUSH,
                         QURT_MEM_DCACHE);
}

static void pure_pq_dsp_invalidate_buffer(PurePqSqmdLaneBuffer *buffer,
                                          int full_buffer) {
    qurt_mem_cache_clean(
        (qurt_addr_t)buffer,
        (qurt_size_t)(full_buffer ? sizeof(*buffer)
                                 : PURE_PQ_SQMD_CACHE_LINE_BYTES),
        QURT_MEM_CACHE_INVALIDATE,
        QURT_MEM_DCACHE);
}

static void pure_pq_dsp_flush_buffer(PurePqSqmdLaneBuffer *buffer) {
    qurt_mem_cache_clean((qurt_addr_t)buffer,
                         (qurt_size_t)sizeof(*buffer),
                         QURT_MEM_CACHE_FLUSH,
                         QURT_MEM_DCACHE);
}

static void pure_pq_dsp_invalidate_lut(PurePqQueryLutHeader *header,
                                       int full_lut) {
    size_t bytes = PURE_PQ_SQMD_CACHE_LINE_BYTES;
    if (full_lut) {
        bytes += (size_t)g_pure_pq_m * (size_t)g_pure_pq_ks * sizeof(float);
    }
    qurt_mem_cache_clean((qurt_addr_t)header,
                         (qurt_size_t)bytes,
                         QURT_MEM_CACHE_INVALIDATE,
                         QURT_MEM_DCACHE);
}

static void pure_pq_dsp_flush_stats(PurePqSqmdStats *stats) {
    qurt_mem_cache_clean((qurt_addr_t)stats,
                         (qurt_size_t)sizeof(*stats),
                         QURT_MEM_CACHE_FLUSH,
                         QURT_MEM_DCACHE);
}

static void pure_pq_dsp_abort(PurePqDspLaneWorkerContext *worker) {
    worker->abort_count++;
    worker->control->control_state = PURE_PQ_SQMD_CONTROL_ABORT;
    DB_WRITE_BARRIER();
    pure_pq_dsp_flush_control(worker->control);
}

static int pure_pq_dsp_prepare_local_lut(
        PurePqDspLaneWorkerContext *worker,
        int query_gen) {
    PurePqDspLutCache *cache;
    size_t lut_bytes;

    if (!worker || !worker->lut_header || !worker->shared_lut ||
        !(cache = worker->lut_cache) || !cache->use_hvx || !cache->values ||
        query_gen <= 0 ||
        cache->value_count !=
            (size_t)g_pure_pq_m * (size_t)g_pure_pq_ks) {
        return -1;
    }
    lut_bytes = cache->value_count * sizeof(float);

    for (;;) {
        DB_READ_BARRIER();
        if (cache->query_gen == query_gen) {
            return 0;
        }
        if (__sync_bool_compare_and_swap(&cache->copy_lock, 0, 1)) {
            int result = 0;

            if (cache->query_gen != query_gen) {
                pure_pq_dsp_invalidate_lut(worker->lut_header, 1);
                DB_READ_BARRIER();
                if (worker->lut_header->query_gen != query_gen ||
                    worker->lut_header->lut_gen != query_gen ||
                    worker->lut_header->m != g_pure_pq_m ||
                    worker->lut_header->ks != g_pure_pq_ks ||
                    worker->lut_header->lut_count !=
                        g_pure_pq_m * g_pure_pq_ks ||
                    memcmp(worker->lut_header->index_uuid,
                           g_pure_pq_index_uuid,
                           sizeof(g_pure_pq_index_uuid)) != 0) {
                    result = -1;
                } else {
                    memcpy(cache->values, worker->shared_lut, lut_bytes);
                    DB_WRITE_BARRIER();
                    cache->query_gen = query_gen;
                }
            }
            DB_WRITE_BARRIER();
            __sync_lock_release(&cache->copy_lock);
            return result;
        }
        pure_pq_dsp_invalidate_control(worker->control);
        DB_READ_BARRIER();
        if (worker->control->control_state ==
                PURE_PQ_SQMD_CONTROL_ABORT ||
            worker->control->control_state ==
                PURE_PQ_SQMD_CONTROL_SHUTDOWN) {
            return -1;
        }
    }
}

static void pure_pq_dsp_lane_worker(void *argument) {
    PurePqDspLaneWorkerContext *worker =
        (PurePqDspLaneWorkerContext *)argument;
    int local_query_gen = -1;
    uint64_t wait_start = 0u;

    if (!worker || !worker->control || !worker->lut_header ||
        !worker->shared_lut || !worker->lut_cache ||
        worker->lane_index < 0 ||
        worker->lane_index >= worker->control->lane_count) {
        qurt_thread_exit(QURT_EFAILED);
        return;
    }

    for (;;) {
        PurePqSqmdLane *lane =
            &worker->control->lanes[worker->lane_index];
        int did_work = 0;
        int buffer_index;

#if defined(RETRIEVAL_ENABLE_FULL_BUILD) && RETRIEVAL_ENABLE_BUILD_PAUSE
        build15_dsp_pause(&worker->control->build_pause,worker->lane_index);
#endif
        pure_pq_dsp_invalidate_control(worker->control);
        DB_READ_BARRIER();
        if (worker->control->control_state == PURE_PQ_SQMD_CONTROL_SHUTDOWN ||
            worker->control->control_state == PURE_PQ_SQMD_CONTROL_ABORT) {
            break;
        }
        if (worker->control->control_state != PURE_PQ_SQMD_CONTROL_RUNNING) {
            worker->wait_polls++;
            if ((worker->wait_polls & 255u) == 0u)
                qurt_timer_sleep(1);
            continue;
        }

        for (buffer_index = 0;
             buffer_index < PURE_PQ_SQMD_BUFFERS_PER_LANE;
             buffer_index++) {
            PurePqSqmdLaneBuffer *buffer = &lane->buffers[buffer_index];
            PurePqSqmdAdcContext adc_context;
            int result;

            pure_pq_dsp_invalidate_buffer(buffer, 0);
            DB_READ_BARRIER();
            if (buffer->state != PURE_PQ_SQMD_BUFFER_CPU_READY) {
                continue;
            }
            pure_pq_dsp_invalidate_buffer(buffer, 1);
            pure_pq_dsp_invalidate_control(worker->control);
            pure_pq_dsp_invalidate_lut(worker->lut_header, 0);
            DB_READ_BARRIER();
            if (buffer->query_gen != worker->control->query_gen ||
                buffer->work_count > worker->work_items_cap) {
                pure_pq_dsp_abort(worker);
                qurt_thread_exit(QURT_EFAILED);
                return;
            }
            if (local_query_gen != buffer->query_gen) {
                if (worker->lut_cache->use_hvx) {
                    if (pure_pq_dsp_prepare_local_lut(
                            worker, buffer->query_gen) != 0) {
                        pure_pq_dsp_abort(worker);
                        qurt_thread_exit(QURT_EFAILED);
                        return;
                    }
                } else {
                    pure_pq_dsp_invalidate_lut(worker->lut_header, 1);
                    DB_READ_BARRIER();
                }
                local_query_gen = buffer->query_gen;
            }
            if (wait_start != 0u) {
                worker->wait_cpu_us += HAP_perf_get_time_us() - wait_start;
                wait_start = 0u;
            }
            memset(&adc_context, 0, sizeof(adc_context));
            adc_context.code_chunks =
                (const uint8_t *const *)g_pure_pq_code_chunks;
            adc_context.chunk_vector_counts = g_pure_pq_chunk_counts;
            adc_context.chunk_count = g_pure_pq_chunk_count;
            adc_context.chunk_vectors = g_pure_pq_chunk_vectors;
            adc_context.num_vectors = g_pure_pq_num_vectors;
            adc_context.m = g_pure_pq_m;
            adc_context.ks = g_pure_pq_ks;
            adc_context.index_uuid = g_pure_pq_index_uuid;
            adc_context.lut_header = worker->lut_header;
            adc_context.lut = worker->lut_cache->use_hvx
                ? worker->lut_cache->values
                : worker->shared_lut;
            adc_context.use_hvx = worker->lut_cache->use_hvx;
            adc_context.hvx_gather_buffer = worker->hvx_gather_buffer;
            lane->dsp_active_buffer = buffer_index;
            {
                uint64_t start = HAP_perf_get_time_us();
                result = pure_pq_sqmd_process_buffer_payload(
                    buffer, buffer->turn_id, buffer->query_gen, &adc_context);
                worker->compute_us += HAP_perf_get_time_us() - start;
            }
            lane->dsp_active_buffer = -1;
            if (result != 0) {
                FARF(ERROR,
                     "pure PQ lane=%d buffer=%d process failed turn=%d gen=%d work=%d",
                     worker->lane_index, buffer_index, buffer->turn_id,
                     buffer->query_gen, buffer->work_count);
                pure_pq_dsp_abort(worker);
                qurt_thread_exit(QURT_EFAILED);
                return;
            }
            DB_WRITE_BARRIER();
            pure_pq_dsp_flush_buffer(buffer);

            DB_WRITE_BARRIER();
            buffer->state = PURE_PQ_SQMD_BUFFER_DSP_DONE;
            DB_WRITE_BARRIER();
            qurt_mem_cache_clean((qurt_addr_t)buffer,
                (qurt_size_t)PURE_PQ_SQMD_CACHE_LINE_BYTES,
                QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
            did_work = 1;
        }

        if (!did_work) {
            if (wait_start == 0u) {
                wait_start = HAP_perf_get_time_us();
            }
            worker->wait_polls++;
            if ((worker->wait_polls & 255u) == 0u)
                qurt_timer_sleep(1);
        }
    }
    if (wait_start != 0u) {
        worker->wait_cpu_us += HAP_perf_get_time_us() - wait_start;
    }
    qurt_thread_exit(QURT_EOK);
}

int calculator_retrieval_pure_pq_sqmd_run(remote_handle64 h,
                                           int32 fd_sqmd_ctrl,
                                           int32 fd_query_lut,
                                           int32 lane_count,
                                           int32 work_items_cap) {
    PurePqSqmdSharedControl *control = NULL;
    PurePqQueryLutHeader *lut_header = NULL;
    const float *lut = NULL;
    compute_res_attr_t vtcm_attributes;
    unsigned int vtcm_context_id = 0u;
    PurePqDspLutCache lut_cache;
    uint64 control_physical = 0u;
    uint64 lut_physical = 0u;
    qurt_thread_t threads[PURE_PQ_SQMD_MAX_LANES];
    void *stacks[PURE_PQ_SQMD_MAX_LANES];
    PurePqDspLaneWorkerContext workers[PURE_PQ_SQMD_MAX_LANES];
    int result = AEE_SUCCESS;
    int hvx_eligible;
    int lane;
    (void)h;

    memset(threads, 0, sizeof(threads));
    memset(stacks, 0, sizeof(stacks));
    memset(workers, 0, sizeof(workers));
    memset(&vtcm_attributes, 0, sizeof(vtcm_attributes));
    memset(&lut_cache, 0, sizeof(lut_cache));
    if (fd_sqmd_ctrl < 0 || fd_query_lut < 0 || lane_count <= 0 ||
        lane_count > PURE_PQ_SQMD_MAX_LANES || work_items_cap <= 0 ||
        work_items_cap > PURE_PQ_SQMD_MAX_WORK_ITEMS ||
        !g_pure_pq_codebook || g_pure_pq_chunk_count <= 0 ||
        !__sync_bool_compare_and_swap(&g_pure_pq_session_live, 0, 1)) {
        return AEE_EBADPARM;
    }
    result = HAP_mmap_get(fd_sqmd_ctrl, (void **)&control, &control_physical);
    if (result != 0 || !control) {
        result = AEE_EFAILED;
        goto cleanup;
    }
    result = HAP_mmap_get(fd_query_lut, (void **)&lut_header, &lut_physical);
    if (result != 0 || !lut_header) {
        result = AEE_EFAILED;
        goto cleanup;
    }
    lut = (const float *)((const uint8_t *)lut_header +
                          PURE_PQ_SQMD_CACHE_LINE_BYTES);
    lut_cache.value_count =
        (size_t)g_pure_pq_m * (size_t)g_pure_pq_ks;
    if (lut_cache.value_count == 0u ||
        lut_cache.value_count > UINT_MAX / sizeof(float)) {
        result = AEE_EBADPARM;
        goto cleanup;
    }
    lut_cache.lut_bytes_aligned = PURE_PQ_SQMD_ALIGN_UP(
        lut_cache.value_count * sizeof(float),
        PURE_PQ_SQMD_CACHE_LINE_BYTES);
    hvx_eligible = g_pure_pq_m >= 32 &&
                   (g_pure_pq_ks == 16 || g_pure_pq_ks == 256);
    if (hvx_eligible &&
        HAP_compute_res_attr_init(&vtcm_attributes) == AEE_SUCCESS &&
        HAP_compute_res_attr_set_vtcm_param(
            &vtcm_attributes,
            (unsigned int)(lut_cache.lut_bytes_aligned +
                           (size_t)lane_count *
                               PURE_PQ_SQMD_CACHE_LINE_BYTES),
            1) == AEE_SUCCESS) {
        vtcm_context_id = HAP_compute_res_acquire(&vtcm_attributes, 10000u);
        if (vtcm_context_id != 0u) {
            lut_cache.values = (float *)HAP_compute_res_attr_get_vtcm_ptr(
                &vtcm_attributes);
            if (lut_cache.values) {
                lut_cache.gather_buffers =
                    (uint8_t *)lut_cache.values +
                    lut_cache.lut_bytes_aligned;
            }
        }
    }
    lut_cache.use_hvx = hvx_eligible && lut_cache.values != NULL &&
                        lut_cache.gather_buffers != NULL;
    lut_cache.query_gen = -1;
    lut_cache.copy_lock = 0;
    if (!lut_cache.use_hvx) {
        if (vtcm_context_id != 0u) {
            HAP_compute_res_release(vtcm_context_id);
            vtcm_context_id = 0u;
        }
        if (!hvx_eligible) {
            FARF(ALWAYS,
                 "pure PQ HVX disabled: M=%d is smaller than 32",
                 g_pure_pq_m);
        } else {
            FARF(ALWAYS,
                 "pure PQ HVX disabled: single-page VTCM allocation failed for %u bytes",
                 (unsigned int)(lut_cache.lut_bytes_aligned +
                                (size_t)lane_count *
                                    PURE_PQ_SQMD_CACHE_LINE_BYTES));
        }
    }
    pure_pq_sqmd_control_init(control, lane_count);
    control->control_state = PURE_PQ_SQMD_CONTROL_IDLE;
    DB_WRITE_BARRIER();
    qurt_mem_cache_clean((qurt_addr_t)control,
                         (qurt_size_t)sizeof(*control),
                         QURT_MEM_CACHE_FLUSH,
                         QURT_MEM_DCACHE);

    for (lane = 0; lane < lane_count; lane++) {
        qurt_thread_attr_t attributes;
        char name[QURT_THREAD_ATTR_NAME_MAXLEN];

        stacks[lane] = malloc(RETRIEVAL_ASYNC_DEMO_STACK_SIZE);
        if (!stacks[lane]) {
            result = AEE_ENOMEMORY;
            control->control_state = PURE_PQ_SQMD_CONTROL_ABORT;
            DB_WRITE_BARRIER();
            pure_pq_dsp_flush_control(control);
            goto cleanup;
        }
        workers[lane].control = control;
        workers[lane].lut_header = lut_header;
        workers[lane].shared_lut = lut;
        workers[lane].lut_cache = &lut_cache;
        workers[lane].hvx_gather_buffer = lut_cache.use_hvx
            ? (uint8_t *)lut_cache.gather_buffers +
                (size_t)lane * PURE_PQ_SQMD_CACHE_LINE_BYTES
            : NULL;
        workers[lane].lane_index = lane;
        workers[lane].work_items_cap = work_items_cap;
        qurt_thread_attr_init(&attributes);
        snprintf(name, sizeof(name), "pure_pq_%d", lane);
        qurt_thread_attr_set_name(&attributes, name);
        qurt_thread_attr_set_stack_addr(&attributes, stacks[lane]);
        qurt_thread_attr_set_stack_size(&attributes,
                                        RETRIEVAL_ASYNC_DEMO_STACK_SIZE);
        qurt_thread_attr_set_priority(&attributes,
                                      QURT_THREAD_ATTR_PRIORITY_DEFAULT / 2);
        if (qurt_thread_create(&threads[lane], &attributes,
                               pure_pq_dsp_lane_worker,
                               &workers[lane]) != QURT_EOK) {
            result = AEE_EFAILED;
            control->control_state = PURE_PQ_SQMD_CONTROL_ABORT;
            DB_WRITE_BARRIER();
            pure_pq_dsp_flush_control(control);
            goto cleanup;
        }
    }
    control->control_state = PURE_PQ_SQMD_CONTROL_RUNNING;
    DB_WRITE_BARRIER();
    pure_pq_dsp_flush_control(control);

    for (;;) {
#if defined(RETRIEVAL_ENABLE_FULL_BUILD) && RETRIEVAL_ENABLE_BUILD_PAUSE
        build15_dsp_pause(&control->build_pause,RETRIEVAL_BUILD_PAUSE_SUPERVISOR);
#endif
        pure_pq_dsp_invalidate_control(control);
        DB_READ_BARRIER();
        if (control->control_state == PURE_PQ_SQMD_CONTROL_SHUTDOWN) {
            break;
        }
        if (control->control_state == PURE_PQ_SQMD_CONTROL_ABORT) {
            result = AEE_EFAILED;
            break;
        }
    }

cleanup:
    if (control && result != AEE_SUCCESS &&
        control->control_state != PURE_PQ_SQMD_CONTROL_ABORT) {
        control->control_state = PURE_PQ_SQMD_CONTROL_ABORT;
        DB_WRITE_BARRIER();
        pure_pq_dsp_flush_control(control);
    }
    for (lane = 0; lane < lane_count; lane++) {
        if (threads[lane] != 0) {
            int status = 0;
            qurt_thread_join(threads[lane], &status);
            if (status != QURT_EOK && result == AEE_SUCCESS) {
                result = AEE_EFAILED;
            }
        }
    }
    if (control) {
        qurt_mem_cache_clean((qurt_addr_t)&control->stats,
                             (qurt_size_t)sizeof(control->stats),
                             QURT_MEM_CACHE_INVALIDATE,
                             QURT_MEM_DCACHE);
        DB_READ_BARRIER();
        for (lane = 0; lane < lane_count; lane++) {
            control->stats.dsp_compute_us += workers[lane].compute_us;
            control->stats.dsp_wait_cpu_us += workers[lane].wait_cpu_us;
            control->stats.dsp_wait_polls += workers[lane].wait_polls;
            control->stats.abort_count += workers[lane].abort_count;
        }
        DB_WRITE_BARRIER();
        pure_pq_dsp_flush_stats(&control->stats);
        pure_pq_dsp_flush_control(control);
    }
    for (lane = 0; lane < PURE_PQ_SQMD_MAX_LANES; lane++) {
        free(stacks[lane]);
    }
    if (vtcm_context_id != 0u) {
        HAP_compute_res_release(vtcm_context_id);
    }
    if (lut_header) {
        HAP_mmap_put(fd_query_lut);
    }
    if (control) {
        HAP_mmap_put(fd_sqmd_ctrl);
    }
    __sync_lock_release(&g_pure_pq_session_live);
    return result;
}
