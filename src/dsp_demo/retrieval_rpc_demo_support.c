typedef enum {
    RESIDUAL_CACHE_ROLE_RERANK     = 0,
    RESIDUAL_CACHE_ROLE_CORRECTION = 1,
    RESIDUAL_CACHE_ROLE_LOCAL      = 2,
    RESIDUAL_CACHE_ROLE__COUNT
} ResidualCacheRole;

typedef struct {
    uint64_t hits  [RESIDUAL_CACHE_ROLE__COUNT];
    uint64_t misses[RESIDUAL_CACHE_ROLE__COUNT];
    uint64_t errors[RESIDUAL_CACHE_ROLE__COUNT];
    volatile int io_error;
    pthread_mutex_t mu;
} ResidualAggregateStats;

typedef struct {
    int     block_idx;
    uint8_t data[RESIDUAL_BLOCK_SIZE];
} ResidualCacheSlot;

struct ResidualBlockCache {
    int  fd;
    int  vectors_per_block;
    int  total_blocks;
    int  residual_code_bytes;
    int  cache_slots;
    const uint8_t *mem_base;
    uint64_t hits;
    uint64_t misses;
    uint64_t errors;
    ResidualAggregateStats *stats;
    ResidualCacheSlot *slots;
};

static int g_residual_fd           = -1;
static int g_residual_vpb          = 0;
static int g_residual_total_blocks = 0;
static ResidualAggregateStats g_residual_stats;

static int pwrite_full(int fd, const void *buf, size_t n, off_t off) {
    size_t done = 0;
    while (done < n) {
        ssize_t k = pwrite(fd, (const char *)buf + done, n - done, off + (off_t)done);
        if (k < 0) { if (errno == EINTR) continue; return -1; }
        if (k == 0) { errno = EIO; return -1; }
        done += (size_t)k;
    }
    return 0;
}

static int pread_full(int fd, void *buf, size_t n, off_t off) {
    size_t done = 0;
    while (done < n) {
        ssize_t k = pread(fd, (char *)buf + done, n - done, off + (off_t)done);
        if (k < 0) { if (errno == EINTR) continue; return -1; }
        if (k == 0) { errno = EIO; return -1; }
        done += (size_t)k;
    }
    return 0;
}

static int residual_write_to_file(
    const RetrievalDemoEncodedDataset *dataset,
    const char *file_path,
    const int32_t *reverse_map)
{
    int vpb, total_blocks, fd, b;
    uint8_t block_buf[RESIDUAL_BLOCK_SIZE];

    if (!dataset || !dataset->residual_codes ||
        dataset->residual_code_bytes <= 0 ||
        dataset->residual_code_bytes > RESIDUAL_BLOCK_SIZE) {
        return -1;
    }

    vpb = RESIDUAL_BLOCK_SIZE / dataset->residual_code_bytes;
    total_blocks = (dataset->num_vectors + vpb - 1) / vpb;

    fd = open(file_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return -1;

    for (b = 0; b < total_blocks; b++) {
        int start_new = b * vpb;
        int nvec = vpb;
        int v;
        if (start_new + nvec > dataset->num_vectors)
            nvec = dataset->num_vectors - start_new;

        memset(block_buf, 0, RESIDUAL_BLOCK_SIZE);
        if (reverse_map) {
            for (v = 0; v < nvec; v++) {
                int orig_id = reverse_map[start_new + v];
                memcpy(block_buf + (size_t)v * (size_t)dataset->residual_code_bytes,
                       dataset->residual_codes
                           + (size_t)orig_id * (size_t)dataset->residual_code_bytes,
                       (size_t)dataset->residual_code_bytes);
            }
        } else {
            memcpy(block_buf,
                   dataset->residual_codes
                       + (size_t)start_new * (size_t)dataset->residual_code_bytes,
                   (size_t)nvec * (size_t)dataset->residual_code_bytes);
        }

        if (pwrite_full(fd, block_buf, RESIDUAL_BLOCK_SIZE,
                        (off_t)b * RESIDUAL_BLOCK_SIZE) != 0) {
            goto fail;
        }
    }

    if (fsync(fd) != 0) goto fail;
    if (close(fd) != 0) { fd = -1; goto fail_unlink; }
    return 0;

fail:
    close(fd);
fail_unlink:
    unlink(file_path);
    return -1;
}

static ResidualBlockCache *residual_cache_create(
    int fd,
    int vectors_per_block,
    int total_blocks,
    int cache_slots,
    const uint8_t *mem_base,
    int residual_code_bytes,
    ResidualAggregateStats *stats)
{
    ResidualBlockCache *c;
    int i;

    if (cache_slots <= 0 || (cache_slots & (cache_slots - 1)) != 0)
        return NULL;

    c = (ResidualBlockCache *)calloc(1, sizeof(*c));
    if (!c) return NULL;

    c->fd                 = fd;
    c->vectors_per_block  = vectors_per_block;
    c->total_blocks       = total_blocks;
    c->residual_code_bytes = residual_code_bytes;
    c->cache_slots        = cache_slots;
    c->mem_base           = mem_base;
    c->stats              = stats;

    if (fd >= 0) {
        c->slots = (ResidualCacheSlot *)malloc((size_t)cache_slots * sizeof(ResidualCacheSlot));
        if (!c->slots) { free(c); return NULL; }

        for (i = 0; i < cache_slots; i++)
            c->slots[i].block_idx = -1;
    }

    return c;
}

static void residual_cache_destroy(ResidualBlockCache *cache) {
    if (!cache) return;
    free(cache->slots);
    free(cache);
}

static void residual_cache_reset(ResidualBlockCache *cache) {
    if (!cache) return;
    int i;
    for (i = 0; i < cache->cache_slots; i++)
        cache->slots[i].block_idx = -1;
    cache->hits = cache->misses = cache->errors = 0;
}

static const uint8_t *residual_cache_get(ResidualBlockCache *cache, int vec_idx) {
    if (!cache) return NULL;

    if (cache->fd < 0) {
        if (!cache->mem_base) return NULL;
        return cache->mem_base + (size_t)(g_reorder_map?g_reorder_map[vec_idx]:vec_idx) * (size_t)cache->residual_code_bytes;
    }

    {
        int file_pos  = g_reorder_map ? g_reorder_map[vec_idx] : vec_idx;
        int block_idx = file_pos / cache->vectors_per_block;
        int within    = (file_pos % cache->vectors_per_block)
                        * cache->residual_code_bytes;
        int slot_idx  = block_idx & (cache->cache_slots - 1);
        ResidualCacheSlot *slot = &cache->slots[slot_idx];

        if (slot->block_idx == block_idx) {
            cache->hits++;
            return slot->data + within;
        }

        cache->misses++;
        slot->block_idx = -1;

        if (pread_full(cache->fd, slot->data, RESIDUAL_BLOCK_SIZE,
                       (off_t)block_idx * RESIDUAL_BLOCK_SIZE) != 0) {
            cache->errors++;
            __atomic_store_n(&cache->stats->io_error, 1, __ATOMIC_RELEASE);
            return NULL;
        }

        slot->block_idx = block_idx;
        return slot->data + within;
    }
}

typedef struct __attribute__((packed)) SqmdCorrectionBiasHeader {
    uint8_t magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t num_vectors;
    uint32_t dim;
    uint32_t l1_bits;
    uint32_t residual_bits;
    uint64_t payload_bytes;
    uint64_t payload_fnv64;
    uint8_t source_fingerprint[16];
} SqmdCorrectionBiasHeader;

#define SQMD_CORRECTION_BIAS_VERSION 1u
#define SQMD_CORRECTION_BIAS_HEADER_BYTES 64u

static uint64_t sqmd_fnv1a64(const void *data, size_t bytes) {
    const uint8_t *p = (const uint8_t *)data;
    uint64_t hash = UINT64_C(14695981039346656037);
    size_t i;

    for (i = 0; i < bytes; i++) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int sqmd_load_correction_bias(
        const char *path,
        const RetrievalDemoEncodedDataset *dataset,
        float **out_bias) {
    static const uint8_t expected_magic[8] = {
        'S', 'Q', 'M', 'D', 'B', 'I', 'A', 'S'
    };
    uint8_t expected_fingerprint[32], hashes[96];
    if(retrieval_build_hash(rover_config.l1,hashes) ||
       retrieval_build_hash(rover_config.residual,hashes+32) ||
       retrieval_build_hash(rover_config.qparams,hashes+64) ||
       retrieval_pq_artifact_hash_bytes(hashes,sizeof(hashes),expected_fingerprint))return -1;
    SqmdCorrectionBiasHeader header;
    struct stat st;
    float *bias = NULL;
    size_t payload_bytes;
    int fd = -1;
    int i;

    if (!path || !dataset || !out_bias || dataset->num_vectors <= 0)
        return -1;
    *out_bias = NULL;
    payload_bytes = (size_t)dataset->num_vectors * sizeof(float);
    fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0 || fstat(fd, &st) != 0 ||
        st.st_size != (off_t)(sizeof(header) + payload_bytes) ||
        pread_full(fd, &header, sizeof(header), 0) != 0 ||
        memcmp(header.magic, expected_magic, sizeof(expected_magic)) != 0 ||
        header.version != SQMD_CORRECTION_BIAS_VERSION ||
        header.header_bytes != SQMD_CORRECTION_BIAS_HEADER_BYTES ||
        header.num_vectors != (uint32_t)dataset->num_vectors ||
        header.dim != (uint32_t)dataset->dim ||
        header.l1_bits != (uint32_t)dataset->l1_bits ||
        header.residual_bits != (uint32_t)dataset->residual_bits ||
        header.payload_bytes != (uint64_t)payload_bytes ||
        memcmp(header.source_fingerprint, expected_fingerprint,
               16) != 0) {
        if (fd >= 0)
            close(fd);
        return -1;
    }

    bias = (float *)malloc(payload_bytes);
    if (!bias || pread_full(fd, bias, payload_bytes,
                            (off_t)sizeof(header)) != 0) {
        free(bias);
        close(fd);
        return -1;
    }
    close(fd);
    if (sqmd_fnv1a64(bias, payload_bytes) != header.payload_fnv64) {
        free(bias);
        return -1;
    }
    for (i = 0; i < dataset->num_vectors; i++) {
        if (!isfinite(bias[i])) {
            free(bias);
            return -1;
        }
    }
    *out_bias = bias;
    return 0;
}

static void residual_cache_flush_stats(
    ResidualBlockCache *cache,
    ResidualAggregateStats *stats,
    ResidualCacheRole role)
{
    if (!cache || !stats) return;
    pthread_mutex_lock(&stats->mu);
    stats->hits  [role] += cache->hits;
    stats->misses[role] += cache->misses;
    stats->errors[role] += cache->errors;
    pthread_mutex_unlock(&stats->mu);
    cache->hits = cache->misses = cache->errors = 0;
}

#ifndef RESIDUAL_UFS_PATH
#define RESIDUAL_UFS_PATH ""
#endif

static const char *residual_resolve_ufs_path(void) {
if (RETRIEVAL_DEMO_USE_DISK == 0) {
    return NULL;
} else {
    const char *env = getenv("RESIDUAL_UFS_PATH");
    if (env && env[0] != '\0') {
        if (env[0] == '0' && env[1] == '\0') return NULL;
        return env;
    }
    return RESIDUAL_UFS_PATH;
}
}

static int residual_init_slot_cache(
    RetrievalSlot *slot,
    const RetrievalDemoEncodedDataset *dataset)
{
    const uint8_t *base = (g_residual_fd < 0) ? dataset->residual_codes : NULL;
    slot->res_cache = residual_cache_create(
        g_residual_fd, g_residual_vpb, g_residual_total_blocks,
        RESIDUAL_CACHE_SLOTS, base,
        dataset->residual_code_bytes,
        &g_residual_stats);
    return (slot->res_cache == NULL) ? -1 : 0;
}

static unsigned long long now_us(void) {
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return (unsigned long long)tv.tv_sec * 1000000ULL +
           (unsigned long long)tv.tv_usec;
}


struct demo_dma_heap_allocation_data {
    uint64_t len;
    uint32_t fd;
    uint32_t fd_flags;
    uint64_t heap_flags;
};
#ifndef DEMO_DMA_HEAP_IOCTL_ALLOC
#define DEMO_DMA_HEAP_IOC_MAGIC 'H'
#define DEMO_DMA_HEAP_IOCTL_ALLOC \
    _IOWR(DEMO_DMA_HEAP_IOC_MAGIC, 0x0, struct demo_dma_heap_allocation_data)
#endif


struct demo_dma_buf_sync {
    uint64_t flags;
};
#ifndef DEMO_DMA_BUF_IOCTL_SYNC
#define DEMO_DMA_BUF_SYNC_READ  (1 << 0)
#define DEMO_DMA_BUF_SYNC_WRITE (2 << 0)
#define DEMO_DMA_BUF_SYNC_RW    (DEMO_DMA_BUF_SYNC_READ | DEMO_DMA_BUF_SYNC_WRITE)
#define DEMO_DMA_BUF_SYNC_START (0 << 2)
#define DEMO_DMA_BUF_SYNC_END   (1 << 2)
#define DEMO_DMA_BUF_IOCTL_SYNC _IOW('b', 0, struct demo_dma_buf_sync)
#endif

#define DEMO_DMABUF_REGISTRY_MAX 4
typedef struct DemoDmabufEntry {
    void  *ptr;
    int    fd;
    size_t length;
} DemoDmabufEntry;
static DemoDmabufEntry g_demo_dmabuf_registry[DEMO_DMABUF_REGISTRY_MAX];

static const char *const DEMO_DMA_HEAP_PATHS[] = {
    "/dev/dma_heap/system",
    "/dev/dma_heap/qcom,system",
};

static int demo_dmabuf_registry_add(void *ptr, int fd, size_t length) {
    int i;
    for (i = 0; i < DEMO_DMABUF_REGISTRY_MAX; i++) {
        if (g_demo_dmabuf_registry[i].ptr == NULL) {
            g_demo_dmabuf_registry[i].ptr    = ptr;
            g_demo_dmabuf_registry[i].fd     = fd;
            g_demo_dmabuf_registry[i].length = length;
            return 0;
        }
    }
    return -1;
}

static DemoDmabufEntry *demo_dmabuf_registry_find(void *ptr) {
    int i;
    if (!ptr) {
        return NULL;
    }
    for (i = 0; i < DEMO_DMABUF_REGISTRY_MAX; i++) {
        if (g_demo_dmabuf_registry[i].ptr == ptr) {
            return &g_demo_dmabuf_registry[i];
        }
    }
    return NULL;
}


static void demo_dmabuf_sync(int fd, uint64_t flags) {
    struct demo_dma_buf_sync sync;
    if (fd < 0) {
        return;
    }
    memset(&sync, 0, sizeof(sync));
    sync.flags = flags;
    if (ioctl(fd, DEMO_DMA_BUF_IOCTL_SYNC, &sync) != 0) {
        printf("  WARN: dma_buf sync(0x%llx) failed (errno=%d)\n",
               (unsigned long long)flags, errno);
    }
}


static void *demo_dmaheap_alloc(size_t bytes) {
    struct demo_dma_heap_allocation_data alloc;
    void *ptr;
    int heap_fd = -1;
    unsigned h;

    for (h = 0; h < sizeof(DEMO_DMA_HEAP_PATHS) / sizeof(DEMO_DMA_HEAP_PATHS[0]);
         h++) {
        heap_fd = open(DEMO_DMA_HEAP_PATHS[h], O_RDONLY | O_CLOEXEC);
        if (heap_fd >= 0) {
            break;
        }
    }
    if (heap_fd < 0) {
        printf("  dma_heap: cannot open any /dev/dma_heap/* (errno=%d)\n", errno);
        return NULL;
    }

    memset(&alloc, 0, sizeof(alloc));
    alloc.len       = (uint64_t)bytes;
    alloc.fd_flags  = (uint32_t)(O_RDWR | O_CLOEXEC);
    alloc.heap_flags = 0;
    if (ioctl(heap_fd, DEMO_DMA_HEAP_IOCTL_ALLOC, &alloc) != 0) {
        printf("  dma_heap: ALLOC(%llu bytes) failed (errno=%d)\n",
               (unsigned long long)bytes, errno);
        close(heap_fd);
        return NULL;
    }
    close(heap_fd);

    ptr = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, (int)alloc.fd, 0);
    if (ptr == MAP_FAILED) {
        printf("  dma_heap: mmap(%llu bytes) failed (errno=%d)\n",
               (unsigned long long)bytes, errno);
        close((int)alloc.fd);
        return NULL;
    }
    if (demo_dmabuf_registry_add(ptr, (int)alloc.fd, bytes) != 0) {
        printf("  dma_heap: registry full (max %d)\n", DEMO_DMABUF_REGISTRY_MAX);
        munmap(ptr, bytes);
        close((int)alloc.fd);
        return NULL;
    }

    demo_dmabuf_sync((int)alloc.fd, DEMO_DMA_BUF_SYNC_START | DEMO_DMA_BUF_SYNC_RW);
    return ptr;
}

static void demo_dmaheap_free(DemoDmabufEntry *e) {
    if (!e || !e->ptr) {
        return;
    }
    munmap(e->ptr, e->length);
    if (e->fd >= 0) {
        close(e->fd);
    }
    e->ptr    = NULL;
    e->fd     = -1;
    e->length = 0;
}


static void demo_shared_buf_flush_after_fill(void *ptr) {
    DemoDmabufEntry *e = demo_dmabuf_registry_find(ptr);
    if (!e) {
        return;
    }
    demo_dmabuf_sync(e->fd, DEMO_DMA_BUF_SYNC_END | DEMO_DMA_BUF_SYNC_RW);
}


static int demo_buf_to_fd(void *ptr) {
    DemoDmabufEntry *e = demo_dmabuf_registry_find(ptr);
    if (e) {
        return e->fd;
    }
    return rpcmem_to_fd(ptr);
}

static void *demo_alloc_shared_buffer(size_t bytes,
                                      const char *label,
                                      int *used_rpcmem) {
    void *ptr = NULL;

    if (used_rpcmem) {
        *used_rpcmem = 0;
    }
    if (bytes == 0) {
        return NULL;
    }

    if (bytes > (size_t)INT_MAX) {

        ptr = demo_dmaheap_alloc(bytes);
        if (ptr) {
            if (used_rpcmem) {
                *used_rpcmem = 1;
            }
            if (label) {
                printf("  %s: %llu bytes via dma_heap (rpcmem int cap exceeded)\n",
                       label, (unsigned long long)bytes);
            }
        }
        return ptr;
    }

    ptr = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (int)bytes);
    if (ptr) {
        if (used_rpcmem) {
            *used_rpcmem = 1;
        }
        return ptr;
    }

    ptr = malloc(bytes);
    if (ptr && label) {
        printf("  WARN: rpcmem_alloc failed for %s (%llu bytes), fallback to malloc\n",
               label,
               (unsigned long long)bytes);
    }
    return ptr;
}

static void demo_free_maybe_shared(void *ptr, int uses_rpcmem) {
    DemoDmabufEntry *e;
    if (!ptr) {
        return;
    }
    e = demo_dmabuf_registry_find(ptr);
    if (e) {
        demo_dmaheap_free(e);
        return;
    }
    if (uses_rpcmem) {
        rpcmem_free(ptr);
    } else {
        free(ptr);
    }
}

static void demo_set_qparams_from_range(float min_val,
                                        float max_val,
                                        int l1_bits,
                                        QCQuantParams *params) {
    float range;
    float levels;

    if (!params) {
        return;
    }

    range = max_val - min_val;
    if (range < 1e-12f) {
        range = 1e-12f;
    }

    levels = (float)((1 << l1_bits) - 1);
    params->scale = range / levels;
    params->zero_point = min_val;
}

static void demo_compute_range(const float *data,
                               size_t count,
                               float *min_val,
                               float *max_val) {
    size_t i;
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;

    if (!data || count == 0) {
        if (min_val) {
            *min_val = 0.0f;
        }
        if (max_val) {
            *max_val = 0.0f;
        }
        return;
    }

    for (i = 0; i < count; i++) {
        if (data[i] < local_min) {
            local_min = data[i];
        }
        if (data[i] > local_max) {
            local_max = data[i];
        }
    }

    if (min_val) {
        *min_val = local_min;
    }
    if (max_val) {
        *max_val = local_max;
    }
}

static int demo_l1_levels(const RetrievalDemoEncodedDataset *dataset) {
    if (!dataset || (dataset->l1_bits != 4 && dataset->l1_bits != 8)) {
        return -1;
    }
    return (1 << dataset->l1_bits) - 1;
}

static uint8_t demo_unpack_l1_code(const RetrievalDemoEncodedDataset *dataset,
                                   const uint8_t *l1_codes,
                                   int dim_idx) {
    if (dataset->l1_bits == 4) {
        uint8_t packed = l1_codes[dim_idx >> 1];
        return (uint8_t)(((dim_idx & 1) == 0)
                             ? (packed & 0x0F)
                             : ((packed >> 4) & 0x0F));
    }
    return l1_codes[dim_idx];
}

static const uint8_t *demo_l1_code_ptr(const RetrievalDemoEncodedDataset *dataset,
                                       int vec_idx) {

    if (dataset->l1_chunk_count <= 0 || dataset->l1_chunk_vectors <= 0) {
        return dataset->l1_codes +
               (size_t)vec_idx * (size_t)dataset->l1_code_bytes;
    }
    {
        int vpc = dataset->l1_chunk_vectors;
        int c   = vec_idx / vpc;
        int off = vec_idx % vpc;
        return dataset->l1_chunks[c] +
               (size_t)off * (size_t)dataset->l1_code_bytes;
    }
}

static int demo_quantize_l1_vector(const float *vec,
                                   const RetrievalDemoEncodedDataset *dataset,
                                   uint8_t *l1_dst,
                                   float *deq_buf) {
    const float inv_scale = 1.0f / dataset->l1_qparams.scale;
    const int levels = demo_l1_levels(dataset);
    int i;

    if (!vec || !dataset || !l1_dst || !deq_buf || levels <= 0) {
        return -1;
    }

    memset(l1_dst, 0, (size_t)dataset->l1_code_bytes);
    for (i = 0; i < dataset->dim; i++) {
        int q = (int)roundf((vec[i] - dataset->l1_qparams.zero_point) * inv_scale);
        if (q < 0) {
            q = 0;
        } else if (q > levels) {
            q = levels;
        }

        if (dataset->l1_bits == 4) {
            size_t byte_idx = (size_t)i >> 1;
            if (i & 1) {
                l1_dst[byte_idx] |= (uint8_t)(q << 4);
            } else {
                l1_dst[byte_idx] = (uint8_t)q;
            }
        } else {
            l1_dst[i] = (uint8_t)q;
        }
        deq_buf[i] = (float)q * dataset->l1_qparams.scale +
                     dataset->l1_qparams.zero_point;
    }

    return 0;
}

static int demo_quantize_query_l1(const float *query,
                                  const RetrievalDemoEncodedDataset *dataset,
                                  uint8_t *query_l1) {
    if (!query || !dataset || !query_l1) {
        return -1;
    }

    if (dataset->l1_bits == 4) {
        return retrieval_quantize_query_u4(query,
                                           dataset->dim,
                                           dataset->l1_qparams.scale,
                                           dataset->l1_qparams.zero_point,
                                           query_l1);
    }

    return retrieval_quantize_query_u8(query,
                                       dataset->dim,
                                       dataset->l1_qparams.scale,
                                       dataset->l1_qparams.zero_point,
                                       query_l1);
}

static float demo_compute_l1_distance(const RetrievalDemoEncodedDataset *dataset,
                                      const uint8_t *query_l1,
                                      const uint8_t *db_codes) {
    if (!dataset || !query_l1 || !db_codes) {
        return FLT_MAX;
    }

    if (dataset->l1_bits == 4) {
        return retrieval_compute_int4_packed_distance_u8(query_l1,
                                                         db_codes,
                                                         dataset->dim,
                                                         dataset->l1_qparams.scale);
    }

    return retrieval_compute_int8_native_distance_u8(query_l1,
                                                     db_codes,
                                                     dataset->dim,
                                                     dataset->l1_qparams.scale);
}

static int demo_compute_l1_distances_batch(const RetrievalDemoEncodedDataset *dataset,
                                           const uint8_t *query_l1,
                                           const uint8_t *flat_db_codes,
                                           int num_vectors,
                                           float *distances) {
    int i;

    if (!dataset || !query_l1 || !flat_db_codes || !distances || num_vectors <= 0) {
        return -1;
    }

    if (dataset->l1_bits == 8) {
        return retrieval_compute_int8_native_distance_batch(query_l1,
                                                            flat_db_codes,
                                                            dataset->dim,
                                                            num_vectors,
                                                            dataset->l1_qparams.scale,
                                                            distances);
    }

    for (i = 0; i < num_vectors; i++) {
        distances[i] = retrieval_compute_int4_packed_distance_u8(
            query_l1,
            flat_db_codes + (size_t)i * (size_t)dataset->l1_code_bytes,
            dataset->dim,
            dataset->l1_qparams.scale);
    }

    return 0;
}

static int demo_alloc_encoded_dataset(RetrievalDemoEncodedDataset *dataset,
                                      int num_vectors,
                                      int dim) {
    size_t l1_flat_size;
    size_t residual_flat_size;

    if (!dataset || num_vectors <= 0 || dim <= 0) {
        return -1;
    }

    memset(dataset, 0, sizeof(*dataset));
    dataset->num_vectors = num_vectors;
    dataset->dim = dim;
    dataset->l1_bits = RETRIEVAL_DEMO_L1_BITS;
    dataset->l1_code_bytes = RETRIEVAL_DEMO_L1_CODE_BYTES(dim);
    dataset->residual_bits = RETRIEVAL_DEMO_RESIDUAL_BITS;
    dataset->residual_code_bytes = RETRIEVAL_DEMO_RESIDUAL_CODE_BYTES(dim);

    l1_flat_size = (size_t)num_vectors * (size_t)dataset->l1_code_bytes;
    residual_flat_size = (size_t)num_vectors * (size_t)dataset->residual_code_bytes;
    dataset->l1_codes = (uint8_t *)malloc(l1_flat_size);
    dataset->residual_qparams = (QCResidualQuantParams *)calloc(
        (size_t)num_vectors, sizeof(QCResidualQuantParams));
    dataset->residual_codes = (uint8_t *)malloc(residual_flat_size);
    if (!dataset->l1_codes || !dataset->residual_qparams || !dataset->residual_codes) {
        retrieval_demo_free_encoded_dataset(dataset);
        return -1;
    }

    return 0;
}

void retrieval_demo_free_encoded_dataset(RetrievalDemoEncodedDataset *dataset) {
    if (!dataset) {
        return;
    }

    if (dataset->l1_chunk_count > 0) {

        int c;
        for (c = 0; c < dataset->l1_chunk_count; c++) {
            demo_free_maybe_shared(dataset->l1_chunks[c],
                                   dataset->l1_chunk_uses_rpcmem[c]);
            dataset->l1_chunks[c] = NULL;
        }
    } else {
        demo_free_maybe_shared(dataset->l1_codes, dataset->l1_codes_uses_rpcmem);
    }
    free(dataset->residual_qparams);
    free(dataset->residual_codes);
    memset(dataset, 0, sizeof(*dataset));
}

static int demo_read_fvec_row(FILE *fp, int dim, float *buf) {
    int d = 0;

    if (fread(&d, sizeof(int), 1, fp) != 1) {
        return -1;
    }
    if (d != dim) {
        return -1;
    }
    if (fread(buf, sizeof(float), (size_t)dim, fp) != (size_t)dim) {
        return -1;
    }
    return 0;
}

static void demo_quantize_residual(const float *residual, int dim,
                                  QCResidualQuantParams *rqp,
                                  uint8_t *codes, int residual_bits) {
    float rmin = FLT_MAX;
    float rmax = -FLT_MAX;
    int i;

    for (i = 0; i < dim; i++) {
        if (residual[i] < rmin) rmin = residual[i];
        if (residual[i] > rmax) rmax = residual[i];
    }
    float range = rmax - rmin;
    if (range < 1e-12f) range = 1e-12f;

    if (residual_bits == 4) {
        int levels = 15;
        rqp->scale = range / (float)levels;
        rqp->zero_point = rmin;
        float inv_scale = 1.0f / rqp->scale;
        memset(codes, 0, (size_t)((dim + 1) / 2));
        for (i = 0; i < dim; i++) {
            float v = (residual[i] - rqp->zero_point) * inv_scale;
            int q = (int)roundf(v);
            if (q < 0) q = 0;
            if (q > levels) q = levels;
            if ((i & 1) == 0) {
                codes[i >> 1] = (uint8_t)(q & 0x0F);
            } else {
                codes[i >> 1] |= (uint8_t)((q & 0x0F) << 4);
            }
        }
    } else {
        rqp->scale = range / 255.0f;
        rqp->zero_point = rmin;
        float inv_scale = 1.0f / rqp->scale;
        for (i = 0; i < dim; i++) {
            float v = (residual[i] - rqp->zero_point) * inv_scale;
            int q = (int)roundf(v);
            if (q < 0) q = 0;
            if (q > 255) q = 255;
            codes[i] = (uint8_t)q;
        }
    }
}

static int demo_encode_vector(const float *vec,
                              RetrievalDemoEncodedDataset *dataset,
                              int vec_idx,
                              float *deq_buf,
                              float *res_buf) {
    uint8_t *l1_dst;
    uint8_t *res_dst;

    if (!vec || !dataset || !deq_buf || !res_buf ||
        vec_idx < 0 || vec_idx >= dataset->num_vectors) {
        return -1;
    }

    l1_dst = dataset->l1_codes + (size_t)vec_idx * (size_t)dataset->l1_code_bytes;
    res_dst = dataset->residual_codes + (size_t)vec_idx * (size_t)dataset->residual_code_bytes;

    if (demo_quantize_l1_vector(vec, dataset, l1_dst, deq_buf) != 0) {
        return -1;
    }

    qc_compute_residual(vec, deq_buf, dataset->dim, res_buf);
    demo_quantize_residual(res_buf, dataset->dim,
                           &dataset->residual_qparams[vec_idx],
                           res_dst, dataset->residual_bits);
    return 0;
}

int retrieval_demo_encode_dataset_from_memory(const float *data,
                                              int num_vectors,
                                              int dim,
                                              RetrievalDemoEncodedDataset *dataset) {
    float *deq_buf = NULL;
    float *res_buf = NULL;
    float gmin;
    float gmax;
    int v = 0;

    if (!data || !dataset || num_vectors <= 0 || dim <= 0) {
        return -1;
    }

    if (demo_alloc_encoded_dataset(dataset, num_vectors, dim) != 0) {
        return -1;
    }

    demo_compute_range(data, (size_t)num_vectors * (size_t)dim, &gmin, &gmax);
    demo_set_qparams_from_range(gmin, gmax, dataset->l1_bits, &dataset->l1_qparams);

    deq_buf = (float *)malloc((size_t)dim * sizeof(float));
    res_buf = (float *)malloc((size_t)dim * sizeof(float));
    if (!deq_buf || !res_buf) {
        free(deq_buf);
        free(res_buf);
        retrieval_demo_free_encoded_dataset(dataset);
        return -1;
    }

    for (v = 0; v < num_vectors; v++) {
        if (demo_encode_vector(data + (size_t)v * (size_t)dim,
                               dataset,
                               v,
                               deq_buf,
                               res_buf) != 0) {
            free(deq_buf);
            free(res_buf);
            retrieval_demo_free_encoded_dataset(dataset);
            return -1;
        }
    }

    free(deq_buf);
    free(res_buf);
    return 0;
}

int retrieval_demo_encode_dataset_from_fvecs_streaming(const char *path,
                                                       int num_vectors,
                                                       int dim,
                                                       RetrievalDemoEncodedDataset *dataset) {
    FILE *fp = NULL;
    float *vec_buf = NULL;
    float *deq_buf = NULL;
    float *res_buf = NULL;
    float gmin = FLT_MAX;
    float gmax = -FLT_MAX;
    int v = 0;
    int i = 0;

    if (!path || !dataset || num_vectors <= 0 || dim <= 0) {
        return -1;
    }

    vec_buf = (float *)malloc((size_t)dim * sizeof(float));
    if (!vec_buf) {
        return -1;
    }

    fp = fopen(path, "rb");
    if (!fp) {
        free(vec_buf);
        return -1;
    }

    for (v = 0; v < num_vectors; v++) {
        if (demo_read_fvec_row(fp, dim, vec_buf) != 0) {
            fclose(fp);
            free(vec_buf);
            return -1;
        }
        for (i = 0; i < dim; i++) {
            if (vec_buf[i] < gmin) {
                gmin = vec_buf[i];
            }
            if (vec_buf[i] > gmax) {
                gmax = vec_buf[i];
            }
        }
    }
    fclose(fp);

    if (demo_alloc_encoded_dataset(dataset, num_vectors, dim) != 0) {
        free(vec_buf);
        return -1;
    }
    demo_set_qparams_from_range(gmin, gmax, dataset->l1_bits, &dataset->l1_qparams);

    deq_buf = (float *)malloc((size_t)dim * sizeof(float));
    res_buf = (float *)malloc((size_t)dim * sizeof(float));
    if (!deq_buf || !res_buf) {
        free(vec_buf);
        free(deq_buf);
        free(res_buf);
        retrieval_demo_free_encoded_dataset(dataset);
        return -1;
    }

    fp = fopen(path, "rb");
    if (!fp) {
        free(vec_buf);
        free(deq_buf);
        free(res_buf);
        retrieval_demo_free_encoded_dataset(dataset);
        return -1;
    }

    for (v = 0; v < num_vectors; v++) {
        if (demo_read_fvec_row(fp, dim, vec_buf) != 0 ||
            demo_encode_vector(vec_buf, dataset, v, deq_buf, res_buf) != 0) {
            fclose(fp);
            free(vec_buf);
            free(deq_buf);
            free(res_buf);
            retrieval_demo_free_encoded_dataset(dataset);
            return -1;
        }
    }

    fclose(fp);
    free(vec_buf);
    free(deq_buf);
    free(res_buf);
    return 0;
}


#define L1_CACHE_MAGIC   0x314C4D53u
#define L1_CACHE_VERSION 1u


typedef struct {
    uint32_t magic;
    uint32_t version;
    int32_t  num_vectors;
    int32_t  dim;
    int32_t  l1_bits;
    float    l1_scale;
    float    l1_zero_point;
    uint8_t  reserved[4];
} L1CacheHeader;

#define QPARAMS_CACHE_MAGIC   0x51504D53u
#define QPARAMS_CACHE_VERSION 1u


typedef struct {
    uint32_t magic;
    uint32_t version;
    int32_t  num_vectors;
    int32_t  residual_bits;
    float    l1_scale;
    float    l1_zero_point;
    uint8_t  reserved[8];
} QParamsCacheHeader;


typedef struct {
    int     fd;
    int     residual_code_bytes;
    int     vectors_per_block;
    int     vpb_pos;
    int     block_cur;
    uint8_t block_buf[RESIDUAL_BLOCK_SIZE];
} ResidualStreamWriter;

static void residual_stream_writer_init(ResidualStreamWriter *w,
                                        int fd,
                                        int residual_code_bytes)
{
    memset(w, 0, sizeof(*w));
    w->fd                 = fd;
    w->residual_code_bytes = residual_code_bytes;
    w->vectors_per_block  = RESIDUAL_BLOCK_SIZE / residual_code_bytes;
}


static int residual_stream_write_one(ResidualStreamWriter *w,
                                     const uint8_t *codes)
{
    memcpy(w->block_buf
               + (size_t)w->vpb_pos * (size_t)w->residual_code_bytes,
           codes,
           (size_t)w->residual_code_bytes);
    w->vpb_pos++;
    if (w->vpb_pos >= w->vectors_per_block) {
        if (pwrite_full(w->fd, w->block_buf, RESIDUAL_BLOCK_SIZE,
                        (off_t)w->block_cur * RESIDUAL_BLOCK_SIZE) != 0) {
            return -1;
        }
        w->block_cur++;
        w->vpb_pos = 0;
        memset(w->block_buf, 0, RESIDUAL_BLOCK_SIZE);
    }
    return 0;
}


static int residual_stream_writer_finish(ResidualStreamWriter *w)
{
    if (w->vpb_pos > 0) {
        size_t used = (size_t)w->vpb_pos * (size_t)w->residual_code_bytes;
        memset(w->block_buf + used, 0, RESIDUAL_BLOCK_SIZE - used);
        if (pwrite_full(w->fd, w->block_buf, RESIDUAL_BLOCK_SIZE,
                        (off_t)w->block_cur * RESIDUAL_BLOCK_SIZE) != 0) {
            return -1;
        }
        w->block_cur++;
        w->vpb_pos = 0;
    }
    return 0;
}


static int encode_compute_chunk_size(int dim, int residual_code_bytes)
{
    size_t cap_bytes;
    size_t bytes_per_vec;
    int chunk;

#if ENCODE_CHUNK_MAX_MB > 0
    cap_bytes = (size_t)ENCODE_CHUNK_MAX_MB * 1024u * 1024u;
#else
    {

        FILE *mf = fopen("/proc/meminfo", "r");
        unsigned long long avail = 0;
        char line[128];
        cap_bytes = 128ULL * 1024 * 1024;
        if (mf) {
            while (fgets(line, sizeof(line), mf)) {
                if (sscanf(line, "MemAvailable: %llu kB", &avail) == 1) {

                    size_t cand = (size_t)(avail / 32ULL) * 1024ULL;
                    if (cand > 4ULL * 1024 * 1024)
                        cap_bytes = cand;
                    break;
                }
            }
            fclose(mf);
        }
    }
#endif


    bytes_per_vec = (size_t)dim * sizeof(float)
                  + (size_t)residual_code_bytes;
    chunk = (int)(cap_bytes / bytes_per_vec);
    if (chunk < 1)     chunk = 1;
    if (chunk > 65536) chunk = 65536;
    return chunk;
}


static int write_qparams_cache(const char *path,
                               const RetrievalDemoEncodedDataset *dataset)
{
    QParamsCacheHeader hdr;
    int fd;
    size_t total;

    if (!path || !dataset || !dataset->residual_qparams) {
        return -1;
    }

    memset(&hdr, 0, sizeof(hdr));
    hdr.magic         = QPARAMS_CACHE_MAGIC;
    hdr.version       = QPARAMS_CACHE_VERSION;
    hdr.num_vectors   = dataset->num_vectors;
    hdr.residual_bits = dataset->residual_bits;
    hdr.l1_scale      = dataset->l1_qparams.scale;
    hdr.l1_zero_point = dataset->l1_qparams.zero_point;

    fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        return -1;
    }

    if (pwrite_full(fd, &hdr, sizeof(hdr), 0) != 0) {
        close(fd);
        unlink(path);
        return -1;
    }

    total = (size_t)dataset->num_vectors * sizeof(QCResidualQuantParams);
    if (pwrite_full(fd, dataset->residual_qparams, total,
                    (off_t)sizeof(hdr)) != 0) {
        close(fd);
        unlink(path);
        return -1;
    }

    if (fsync(fd) != 0) {
        close(fd);
        unlink(path);
        return -1;
    }
    if (close(fd) != 0) {
        unlink(path);
        return -1;
    }
    return 0;
}


int retrieval_demo_encode_dataset_streaming_to_disk(
    const char *fvecs_path,
    int num_vectors,
    int dim,
    const char *residual_path,
    const char *qparams_path,
    const int32_t *reverse_map,
    RetrievalDemoEncodedDataset *dataset)
{
    FILE *fp           = NULL;
    float *vec_buf     = NULL;
    float *chunk_float = NULL;
    uint8_t *chunk_res = NULL;
    float *deq_buf     = NULL;
    float *res_buf     = NULL;
    float gmin         = FLT_MAX;
    float gmax         = -FLT_MAX;
    int chunk_size     = 0;
    int rfd            = -1;
    int v, i, chunk_start, nvec, cv;
    ResidualStreamWriter sw;
    int ret = -1;

    if (!fvecs_path || !residual_path || !dataset ||
        num_vectors <= 0 || dim <= 0) {
        return -1;
    }


    if (reverse_map != NULL) {
        return -2;
    }


    vec_buf = (float *)malloc((size_t)dim * sizeof(float));
    if (!vec_buf) { goto out; }

    fp = fopen(fvecs_path, "rb");
    if (!fp) { goto out; }

    for (v = 0; v < num_vectors; v++) {
        if (demo_read_fvec_row(fp, dim, vec_buf) != 0) { goto out; }
        for (i = 0; i < dim; i++) {
            if (vec_buf[i] < gmin) gmin = vec_buf[i];
            if (vec_buf[i] > gmax) gmax = vec_buf[i];
        }
    }
    fclose(fp); fp = NULL;


    {
        size_t l1_flat = (size_t)num_vectors
                       * (size_t)RETRIEVAL_DEMO_L1_CODE_BYTES(dim);
        int    l1_bits = RETRIEVAL_DEMO_L1_BITS;
        int    l1_cbytes = RETRIEVAL_DEMO_L1_CODE_BYTES(dim);
        int    res_bits  = RETRIEVAL_DEMO_RESIDUAL_BITS;
        int    res_cbytes = RETRIEVAL_DEMO_RESIDUAL_CODE_BYTES(dim);

        memset(dataset, 0, sizeof(*dataset));
        dataset->num_vectors         = num_vectors;
        dataset->dim                 = dim;
        dataset->l1_bits             = l1_bits;
        dataset->l1_code_bytes       = l1_cbytes;
        dataset->residual_bits       = res_bits;
        dataset->residual_code_bytes = res_cbytes;

        demo_set_qparams_from_range(gmin, gmax, l1_bits, &dataset->l1_qparams);

        dataset->l1_codes = (uint8_t *)malloc(l1_flat);
        dataset->residual_qparams = (QCResidualQuantParams *)calloc(
            (size_t)num_vectors, sizeof(QCResidualQuantParams));

        if (!dataset->l1_codes || !dataset->residual_qparams) { goto out; }
    }


    chunk_size = encode_compute_chunk_size(dim, dataset->residual_code_bytes);
    printf("  [P8.2] chunk_size=%d (%.1f MB float+res)\n",
           chunk_size,
           (double)chunk_size
               * ((double)dim * 4.0 + (double)dataset->residual_code_bytes)
               / (1024.0 * 1024.0));

    chunk_float = (float *)malloc((size_t)chunk_size * (size_t)dim * sizeof(float));
    chunk_res   = (uint8_t *)malloc((size_t)chunk_size
                                    * (size_t)dataset->residual_code_bytes);
    deq_buf = (float *)malloc((size_t)dim * sizeof(float));
    res_buf = (float *)malloc((size_t)dim * sizeof(float));
    if (!chunk_float || !chunk_res || !deq_buf || !res_buf) { goto out; }


    rfd = open(residual_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (rfd < 0) { goto out; }

    residual_stream_writer_init(&sw, rfd, dataset->residual_code_bytes);


    fp = fopen(fvecs_path, "rb");
    if (!fp) { goto out; }

    for (chunk_start = 0; chunk_start < num_vectors; chunk_start += chunk_size) {
        nvec = chunk_size;
        if (chunk_start + nvec > num_vectors)
            nvec = num_vectors - chunk_start;


        for (cv = 0; cv < nvec; cv++) {
            if (demo_read_fvec_row(fp, dim,
                                   chunk_float + (size_t)cv * (size_t)dim) != 0) {
                goto out;
            }
        }


        for (cv = 0; cv < nvec; cv++) {
            int gv           = chunk_start + cv;
            const float *src = chunk_float + (size_t)cv * (size_t)dim;
            uint8_t *l1_dst  = dataset->l1_codes
                             + (size_t)gv * (size_t)dataset->l1_code_bytes;
            uint8_t *res_dst = chunk_res
                             + (size_t)cv * (size_t)dataset->residual_code_bytes;

            if (demo_quantize_l1_vector(src, dataset, l1_dst, deq_buf) != 0) {
                goto out;
            }
            qc_compute_residual(src, deq_buf, dim, res_buf);
            demo_quantize_residual(res_buf, dim,
                                   &dataset->residual_qparams[gv],
                                   res_dst, dataset->residual_bits);

            if (residual_stream_write_one(&sw, res_dst) != 0) { goto out; }
        }
    }

    fclose(fp); fp = NULL;

    if (residual_stream_writer_finish(&sw) != 0) { goto out; }
    if (fsync(rfd) != 0) { goto out; }
    close(rfd); rfd = -1;


    if (qparams_path) {
        if (write_qparams_cache(qparams_path, dataset) != 0) {
            printf("  [P8.2] WARN: failed to write qparams cache (%s)\n",
                   qparams_path);

        }
    }

    ret = 0;

out:
    if (fp)  fclose(fp);
    if (rfd >= 0) { close(rfd); unlink(residual_path); }
    free(vec_buf);
    free(chunk_float);
    free(chunk_res);
    free(deq_buf);
    free(res_buf);
    if (ret != 0) {
        retrieval_demo_free_encoded_dataset(dataset);
    }
    return ret;
}


static float retrieval_demo_exact_distance_residual_scalar(
    const float *query,
    const RetrievalDemoEncodedDataset *dataset,
    int vec_idx,
    ResidualBlockCache *cache) {
    const uint8_t *l1_codes;
    const uint8_t *residual_codes = NULL;
    const QCResidualQuantParams *res_qp = NULL;
    float dist = 0.0f;
    int i = 0;

    l1_codes = demo_l1_code_ptr(dataset, vec_idx);
    if (dataset->residual_qparams && dataset->residual_code_bytes > 0) {
        res_qp = &dataset->residual_qparams[vec_idx];
        if (cache) {
            residual_codes = residual_cache_get(cache, vec_idx);
            if (!residual_codes) return FLT_MAX;
        } else if (dataset->residual_codes) {
            residual_codes = dataset->residual_codes +
                             (size_t)vec_idx * (size_t)dataset->residual_code_bytes;
        }
    }

    for (i = 0; i < dataset->dim; i++) {
        float value = (float)demo_unpack_l1_code(dataset, l1_codes, i) *
                      dataset->l1_qparams.scale +
                      dataset->l1_qparams.zero_point;
        float diff;

        if (res_qp && residual_codes) {
            uint8_t res_val;
            if (dataset->residual_bits == 4) {
                res_val = (uint8_t)(((i & 1) == 0)
                          ? (residual_codes[i >> 1] & 0x0F)
                          : ((residual_codes[i >> 1] >> 4) & 0x0F));
            } else {
                res_val = residual_codes[i];
            }
            value += (float)res_val * res_qp->scale + res_qp->zero_point;
        }

        diff = query[i] - value;
        dist += diff * diff;
    }

    return dist;
}

#ifdef __aarch64__
#include <arm_neon.h>


static float retrieval_demo_exact_distance_residual_neon(
    const float *query,
    const RetrievalDemoEncodedDataset *dataset,
    int vec_idx,
    ResidualBlockCache *cache) {
    const uint8_t *l1_codes = demo_l1_code_ptr(dataset, vec_idx);
    const uint8_t *residual_codes = NULL;
    const QCResidualQuantParams *res_qp = NULL;
    const int dim = dataset->dim;
    int i;
    uint8_t l1_unpacked[RETRIEVAL_ASYNC_DEMO_MAX_DIM];
    const uint8_t *l1_src;

    if (dataset->residual_qparams && dataset->residual_code_bytes > 0) {
        res_qp = &dataset->residual_qparams[vec_idx];
        if (cache) {
            residual_codes = residual_cache_get(cache, vec_idx);
            if (!residual_codes) return FLT_MAX;
        } else if (dataset->residual_codes) {
            residual_codes = dataset->residual_codes +
                             (size_t)vec_idx * (size_t)dataset->residual_code_bytes;
        }
    }

    if (dataset->l1_bits == 4) {
        int half = (dim + 1) / 2;
        for (i = 0; i < half; i++) {
            l1_unpacked[i * 2]     = l1_codes[i] & 0x0F;
            l1_unpacked[i * 2 + 1] = (l1_codes[i] >> 4) & 0x0F;
        }
        l1_src = l1_unpacked;
    } else {
        l1_src = l1_codes;
    }


    uint8_t res_unpacked[RETRIEVAL_ASYNC_DEMO_MAX_DIM];
    const uint8_t *res_src = NULL;
    if (res_qp && residual_codes) {
        if (dataset->residual_bits == 4) {
            int half = (dim + 1) / 2;
            for (i = 0; i < half; i++) {
                res_unpacked[i * 2]     = residual_codes[i] & 0x0F;
                res_unpacked[i * 2 + 1] = (residual_codes[i] >> 4) & 0x0F;
            }
            res_src = res_unpacked;
        } else {
            res_src = residual_codes;
        }
    }


    float32x4_t vdist = vdupq_n_f32(0.0f);
    float32x4_t vscale_l1 = vdupq_n_f32(dataset->l1_qparams.scale);
    float32x4_t vzp_l1    = vdupq_n_f32(dataset->l1_qparams.zero_point);
    float32x4_t vscale_res, vzp_res;

    if (res_qp) {
        vscale_res = vdupq_n_f32(res_qp->scale);
        vzp_res   = vdupq_n_f32(res_qp->zero_point);
    }


    int neon_limit = dim & ~3;
    for (i = 0; i < neon_limit; i += 4) {
        uint32_t l1_4;
        float32x4_t value;

        memcpy(&l1_4, &l1_src[i], 4);
        {
            uint8x8_t  u8l1  = vcreate_u8((uint64_t)l1_4);
            uint16x4_t u16l1 = vget_low_u16(vmovl_u8(u8l1));
            uint32x4_t u32l1 = vmovl_u16(u16l1);
            float32x4_t fl1  = vcvtq_f32_u32(u32l1);
            value = vmlaq_f32(vzp_l1, fl1, vscale_l1);
        }

        if (res_src) {

            uint32_t res_4;
            memcpy(&res_4, &res_src[i], 4);
            uint8x8_t  u8res  = vcreate_u8((uint64_t)res_4);
            uint16x4_t u16res = vget_low_u16(vmovl_u8(u8res));
            uint32x4_t u32res = vmovl_u16(u16res);
            float32x4_t fres  = vcvtq_f32_u32(u32res);
            float32x4_t dequant_res = vmlaq_f32(vzp_res, fres, vscale_res);
            value = vaddq_f32(value, dequant_res);
        }

        float32x4_t vquery = vld1q_f32(&query[i]);
        float32x4_t diff = vsubq_f32(vquery, value);
        vdist = vfmaq_f32(vdist, diff, diff);
    }


    float tail_dist = 0.0f;
    for (; i < dim; i++) {
        float val = (float)l1_src[i] * dataset->l1_qparams.scale +
                    dataset->l1_qparams.zero_point;
        if (res_src) {
            val += (float)res_src[i] * res_qp->scale + res_qp->zero_point;
        }
        float d = query[i] - val;
        tail_dist += d * d;
    }

    return vaddvq_f32(vdist) + tail_dist;
}
#endif


float retrieval_demo_exact_distance_residual_cached(
    const float *query,
    const RetrievalDemoEncodedDataset *dataset,
    int vec_idx,
    ResidualBlockCache *cache) {
    if (!query || !dataset || vec_idx < 0 || vec_idx >= dataset->num_vectors) {
        return FLT_MAX;
    }

#ifdef __aarch64__
    return retrieval_demo_exact_distance_residual_neon(query, dataset, vec_idx, cache);
#else
    return retrieval_demo_exact_distance_residual_scalar(query, dataset, vec_idx, cache);
#endif
}


float retrieval_demo_exact_distance_residual(const float *query,
                                             const RetrievalDemoEncodedDataset *dataset,
                                             int vec_idx) {
    return retrieval_demo_exact_distance_residual_cached(
        query, dataset, vec_idx, NULL);
}


static void demo_heap_sift_up(DemoCandidate *pool, int i) {
    DemoCandidate tmp = pool[i];
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (tmp.distance <= pool[parent].distance)
            break;
        pool[i] = pool[parent];
        i = parent;
    }
    pool[i] = tmp;
}

static void demo_heap_sift_down(DemoCandidate *pool, int size, int i) {
    DemoCandidate tmp = pool[i];
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


static int demo_candidate_insert(DemoCandidate *pool,
                                 int *pool_size,
                                 int pool_capacity,
                                 int id,
                                 float distance) {
    int size;

    if (!pool || !pool_size || pool_capacity <= 0) {
        return 0;
    }

    size = *pool_size;
    if (size < pool_capacity) {
        pool[size].id       = id;
        pool[size].distance = distance;
        pool[size].expanded = 0;
        demo_heap_sift_up(pool, size);
        *pool_size = size + 1;
        return 1;
    } else if (distance < pool[0].distance) {
        pool[0].id       = id;
        pool[0].distance = distance;
        pool[0].expanded = 0;
        demo_heap_sift_down(pool, size, 0);
        return 1;
    }
    return 0;
}


static void demo_frontier_sift_up(DemoFrontierEntry *f, int i) {
    DemoFrontierEntry tmp = f[i];
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (tmp.distance >= f[parent].distance)
            break;
        f[i] = f[parent];
        i = parent;
    }
    f[i] = tmp;
}

static void demo_frontier_sift_down(DemoFrontierEntry *f, int size, int i) {
    DemoFrontierEntry tmp = f[i];
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

static void demo_frontier_push(DemoFrontierEntry *f, int *f_size,
                               int id, float distance) {
    int pos = *f_size;
    f[pos].id       = id;
    f[pos].distance = distance;
    demo_frontier_sift_up(f, pos);
    *f_size = pos + 1;
}

static int demo_frontier_pop(DemoFrontierEntry *f, int *f_size) {
    int top_id;

    if (*f_size <= 0)
        return -1;

    top_id = f[0].id;
    *f_size -= 1;
    if (*f_size > 0) {
        f[0] = f[*f_size];
        demo_frontier_sift_down(f, *f_size, 0);
    }
    return top_id;
}


static void demo_heap_sort(DemoCandidate *pool, int size) {
    int i;
    for (i = size - 1; i > 0; i--) {
        DemoCandidate tmp = pool[0];
        pool[0] = pool[i];
        pool[i] = tmp;
        demo_heap_sift_down(pool, i, 0);
    }
}

static int demo_recall_hits_at_k(const int *result_ids,
                                 int result_count,
                                 const int32_t *gt_row,
                                 int gt_k,
                                 int k) {
    int hits = 0;
    int i;
    int j;

    for (i = 0; i < k && i < result_count; i++) {
        for (j = 0; j < gt_k && j < k; j++) {
            if (result_ids[i] == (int)gt_row[j]) {
                hits++;
                break;
            }
        }
    }

    return hits;
}

static int demo_promote_l1_codes_to_shared(RetrievalDemoEncodedDataset *dataset) {
    size_t bytes;
    uint8_t *shared_codes;
    int uses_rpcmem = 0;

    if (!dataset || !dataset->l1_codes ||
        dataset->num_vectors <= 0 || dataset->l1_code_bytes <= 0) {
        return -1;
    }


    if (dataset->l1_codes_uses_rpcmem) {

        if (dataset->l1_chunk_count == 0) {
            dataset->l1_chunks[0]           = dataset->l1_codes;
            dataset->l1_chunk_uses_rpcmem[0] = dataset->l1_codes_uses_rpcmem;
            dataset->l1_chunk_fds[0]        = -1;
            dataset->l1_chunk_count         = 1;
            dataset->l1_chunk_vectors       = dataset->num_vectors;
        }
        return 0;
    }

    bytes = (size_t)dataset->num_vectors * (size_t)dataset->l1_code_bytes;
    shared_codes = (uint8_t *)demo_alloc_shared_buffer(bytes, "l1_codes", &uses_rpcmem);
    if (!shared_codes) {
        return -1;
    }

    memcpy(shared_codes, dataset->l1_codes, bytes);
    demo_shared_buf_flush_after_fill(shared_codes);
    demo_free_maybe_shared(dataset->l1_codes, dataset->l1_codes_uses_rpcmem);
    dataset->l1_codes = shared_codes;
    dataset->l1_codes_uses_rpcmem = uses_rpcmem;


    if (dataset->l1_chunk_count == 0) {
        dataset->l1_chunks[0]           = shared_codes;
        dataset->l1_chunk_uses_rpcmem[0] = uses_rpcmem;
        dataset->l1_chunk_fds[0]        = -1;
        dataset->l1_chunk_count         = 1;
        dataset->l1_chunk_vectors       = dataset->num_vectors;
    }
    return 0;
}

static void demo_sort_candidates_by_id(int32_t *candidate_ids,
                                       float *candidate_dists,
                                       int count) {
    int j;

    for (j = 1; j < count; j++) {
        int32_t tid = candidate_ids[j];
        float tdt = candidate_dists[j];
        int32_t tid_key = g_reorder_map ? g_reorder_map[tid] : tid;
        int k = j - 1;

        while (k >= 0) {
            int32_t kid_key = g_reorder_map
                ? g_reorder_map[candidate_ids[k]] : candidate_ids[k];
            if (kid_key <= tid_key) break;
            candidate_ids[k + 1] = candidate_ids[k];
            candidate_dists[k + 1] = candidate_dists[k];
            k--;
        }
        candidate_ids[k + 1] = tid;
        candidate_dists[k + 1] = tdt;
    }
}

static void demo_sort_candidates_by_distance(int32_t *candidate_ids,
                                             float *candidate_dists,
                                             int count) {
    int j;

    for (j = 1; j < count; j++) {
        int32_t tid = candidate_ids[j];
        float tdt = candidate_dists[j];
        int k = j - 1;

        while (k >= 0 && candidate_dists[k] > tdt) {
            candidate_ids[k + 1] = candidate_ids[k];
            candidate_dists[k + 1] = candidate_dists[k];
            k--;
        }
        candidate_ids[k + 1] = tid;
        candidate_dists[k + 1] = tdt;
    }
}

static void demo_accumulate_query_recall(DemoResult *result,
                                         const int *result_ids,
                                         int result_count,
                                         const int32_t *gt_row) {
    if (!result || !result_ids || !gt_row) {
        return;
    }

    result->recall1_hits += demo_recall_hits_at_k(result_ids, result_count, gt_row, DEMO_GT_K, 1);
    result->recall10_hits += demo_recall_hits_at_k(result_ids, result_count, gt_row, DEMO_GT_K, 10);
    result->recall100_hits += demo_recall_hits_at_k(result_ids, result_count, gt_row, DEMO_GT_K, 100);
}

static int demo_rerank_candidates(const float *query,
                                  const RetrievalDemoConfig *config,
                                  int32_t *candidate_ids,
                                  float *candidate_dists,
                                  int actual_count,
                                  int *result_ids,
                                  ResidualBlockCache *cache) {
    int j;
    int rc;

    if (!query || !config || !candidate_ids || !candidate_dists || actual_count < 0) {
        return -1;
    }

    demo_sort_candidates_by_id(candidate_ids, candidate_dists, actual_count);
    for (j = 0; j < actual_count; j++) {
        candidate_dists[j] = retrieval_demo_exact_distance_residual_cached(
            query,
            &g_dataset,
            (int)candidate_ids[j],
            cache);
    }
    demo_sort_candidates_by_distance(candidate_ids, candidate_dists, actual_count);

    rc = (config->top_k < actual_count) ? config->top_k : actual_count;
    if (result_ids) {
        for (j = 0; j < rc; j++) {
            result_ids[j] = (int)candidate_ids[j];
        }
    }
    return rc;
}

static int load_graph_nndescent(const char *path, int n_base, int k, int32_t *graph) {
    FILE *fp = NULL;
    int32_t header[3];
    int32_t *tmp_ids = NULL;
    float *tmp_dists = NULL;
    int i;

    if (!path || !graph || n_base <= 0 || k <= 0) {
        return -1;
    }

    fp = fopen(path, "rb");
    if (!fp) {
        return -1;
    }
    posix_fadvise(fileno(fp), 0, 0, POSIX_FADV_SEQUENTIAL);
    if (fread(header, sizeof(int32_t), 3, fp) != 3) {
        fclose(fp);
        return -1;
    }
    if (header[0] != n_base || header[1] != k) {
        fclose(fp);
        return -1;
    }

    tmp_ids = (int32_t *)malloc((size_t)k * sizeof(int32_t));
    tmp_dists = (float *)malloc((size_t)k * sizeof(float));
    if (!tmp_ids || !tmp_dists) {
        fclose(fp);
        free(tmp_ids);
        free(tmp_dists);
        return -1;
    }

    for (i = 0; i < n_base; i++) {
        if (fread(tmp_ids, sizeof(int32_t), (size_t)k, fp) != (size_t)k ||
            fread(tmp_dists, sizeof(float), (size_t)k, fp) != (size_t)k) {
            fclose(fp);
            free(tmp_ids);
            free(tmp_dists);
            return -1;
        }
        memcpy(graph + (size_t)i * (size_t)k,
               tmp_ids,
               (size_t)k * sizeof(int32_t));


        if ((i & 0x3FFFF) == 0x3FFFF) {
            posix_fadvise(fileno(fp), 0, ftello(fp), POSIX_FADV_DONTNEED);
        }
    }


    posix_fadvise(fileno(fp), 0, 0, POSIX_FADV_DONTNEED);
    fclose(fp);
    free(tmp_ids);
    free(tmp_dists);
    return 0;
}


int retrieval_demo_load_qparams_from_cache(
    const char *path,
    int num_vectors,
    int residual_bits,
    RetrievalDemoEncodedDataset *dataset)
{
    QParamsCacheHeader hdr;
    int fd;
    size_t total;

    if (!path || !dataset || !dataset->residual_qparams || num_vectors <= 0) {
        return -1;
    }

    fd = open(path, O_RDONLY);
    if (fd < 0) { return -1; }

    if (pread_full(fd, &hdr, sizeof(hdr), 0) != 0) { close(fd); return -1; }

    if (hdr.magic         != QPARAMS_CACHE_MAGIC      ||
        hdr.version       != QPARAMS_CACHE_VERSION     ||
        hdr.num_vectors   != num_vectors               ||
        hdr.residual_bits != residual_bits             ||
        hdr.l1_scale      != dataset->l1_qparams.scale ||
        hdr.l1_zero_point != dataset->l1_qparams.zero_point) {
        close(fd);
        return -1;
    }

    total = (size_t)num_vectors * sizeof(QCResidualQuantParams);
    if (pread_full(fd, dataset->residual_qparams, total,
                   (off_t)sizeof(hdr)) != 0) {
        close(fd);
        return -1;
    }

    close(fd);
    return 0;
}


int retrieval_demo_load_l1_and_qparams_from_disk(
    const char *l1_path,
    const char *qparams_path,
    int num_vectors,
    int dim,
    int residual_bits,
    RetrievalDemoEncodedDataset *dataset)
{
    L1CacheHeader hdr;
    int fd = -1;
    size_t l1_bytes;
    int l1_bits, l1_cbytes;

    if (!l1_path || !qparams_path || !dataset || num_vectors <= 0 || dim <= 0) {
        return -1;
    }

    fd = open(l1_path, O_RDONLY);
    if (fd < 0) { return -1; }
    if (pread_full(fd, &hdr, sizeof(hdr), 0) != 0) { close(fd); return -1; }

    if (hdr.magic       != L1_CACHE_MAGIC   ||
        hdr.version     != L1_CACHE_VERSION ||
        hdr.num_vectors != num_vectors      ||
        hdr.dim         != dim              ||
        (hdr.l1_bits != 4 && hdr.l1_bits != 8) || hdr.l1_bits != RETRIEVAL_DEMO_L1_BITS) {
        close(fd);
        return -1;
    }

    l1_bits   = hdr.l1_bits;
    l1_cbytes = RETRIEVAL_DEMO_L1_CODE_BYTES(dim);

    memset(dataset, 0, sizeof(*dataset));
    dataset->num_vectors         = num_vectors;
    dataset->dim                 = dim;
    dataset->l1_bits             = l1_bits;
    dataset->l1_code_bytes       = l1_cbytes;
    dataset->l1_qparams.scale      = hdr.l1_scale;
    dataset->l1_qparams.zero_point = hdr.l1_zero_point;
    dataset->residual_bits       = residual_bits;
    dataset->residual_code_bytes = RETRIEVAL_DEMO_RESIDUAL_CODE_BYTES(dim);

    l1_bytes = (size_t)num_vectors * (size_t)l1_cbytes;


    {
        int vpc = (int)(RETRIEVAL_DEMO_L1_CHUNK_BYTES / (size_t)l1_cbytes);
        int chunk_count, c;
        const off_t base_off = (off_t)sizeof(hdr);
        if (vpc < 1) { vpc = 1; }
        chunk_count = (num_vectors + vpc - 1) / vpc;
        if (chunk_count > MAX_L1_CHUNKS) {
            printf("  ERROR: L1 needs %d chunks (vpc=%d) but MAX_L1_CHUNKS=%d; "
                   "raise RETRIEVAL_DEMO_L1_CHUNK_BYTES or MAX_L1_CHUNKS\n",
                   chunk_count, vpc, MAX_L1_CHUNKS);
            close(fd);
            retrieval_demo_free_encoded_dataset(dataset);
            return -1;
        }

        dataset->residual_qparams = (QCResidualQuantParams *)calloc(
            (size_t)num_vectors, sizeof(QCResidualQuantParams));
        if (!dataset->residual_qparams) {
            close(fd);
            retrieval_demo_free_encoded_dataset(dataset);
            return -1;
        }

        for (c = 0; c < chunk_count; c++) {
            int    chunk_vecs  = (c == chunk_count - 1)
                                     ? (num_vectors - c * vpc) : vpc;
            size_t chunk_bytes = (size_t)chunk_vecs * (size_t)l1_cbytes;
            off_t  chunk_off   = base_off + (off_t)c * (off_t)vpc *
                                            (off_t)l1_cbytes;
            int    uses_rpcmem = 0;
            uint8_t *buf = (uint8_t *)demo_alloc_shared_buffer(
                chunk_bytes, "l1_chunk", &uses_rpcmem);
            if (!buf) {
                printf("  ERROR: alloc L1 chunk %d/%d (%llu bytes) failed\n",
                       c, chunk_count, (unsigned long long)chunk_bytes);
                close(fd);
                retrieval_demo_free_encoded_dataset(dataset);
                return -1;
            }
            dataset->l1_chunks[c]           = buf;
            dataset->l1_chunk_uses_rpcmem[c] = uses_rpcmem;
            dataset->l1_chunk_fds[c]        = -1;
            dataset->l1_chunk_count         = c + 1;


            {
                const size_t SUB = (size_t)256 * 1024 * 1024;
                size_t done = 0;
                while (done < chunk_bytes) {
                    size_t n = chunk_bytes - done;
                    if (n > SUB) { n = SUB; }
                    if (pread_full(fd, buf + done, n,
                                   chunk_off + (off_t)done) != 0) {
                        close(fd);
                        retrieval_demo_free_encoded_dataset(dataset);
                        return -1;
                    }
                    posix_fadvise(fd, chunk_off + (off_t)done, (off_t)n,
                                  POSIX_FADV_DONTNEED);
                    done += n;
                }
            }
            demo_shared_buf_flush_after_fill(buf);
        }

        dataset->l1_chunk_vectors     = vpc;
        dataset->l1_codes             = dataset->l1_chunks[0];
        dataset->l1_codes_uses_rpcmem = dataset->l1_chunk_uses_rpcmem[0];
    }
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);
    fd = -1;


    if (retrieval_demo_load_qparams_from_cache(
            qparams_path, num_vectors, residual_bits, dataset) != 0) {
        retrieval_demo_free_encoded_dataset(dataset);
        return -1;
    }


    return 0;
}

int retrieval_demo_load_residual_codes_from_block_file(
    const char *path,
    RetrievalDemoEncodedDataset *dataset) {
    enum { RESIDUAL_READ_BATCH_BLOCKS = 1024 };
    struct stat st;
    uint8_t *flat = NULL;
    uint8_t *batch = NULL;
    size_t flat_bytes;
    size_t batch_capacity;
    off_t expected_file_bytes;
    int fd = -1;
    int vectors_per_block;
    int total_blocks;
    int first_block;

    if (!path || !dataset || dataset->num_vectors <= 0 ||
        dataset->residual_code_bytes <= 0 ||
        dataset->residual_code_bytes > RESIDUAL_BLOCK_SIZE ||
        dataset->residual_codes) {
        return -1;
    }

    vectors_per_block = RESIDUAL_BLOCK_SIZE /
                        dataset->residual_code_bytes;
    if (vectors_per_block <= 0) {
        return -1;
    }
    total_blocks = 1 + (dataset->num_vectors - 1) / vectors_per_block;

    if ((size_t)dataset->num_vectors >
        SIZE_MAX / (size_t)dataset->residual_code_bytes) {
        return -1;
    }
    flat_bytes = (size_t)dataset->num_vectors *
                 (size_t)dataset->residual_code_bytes;
    expected_file_bytes = (off_t)total_blocks *
                          (off_t)RESIDUAL_BLOCK_SIZE;

    fd = open(path, O_RDONLY);
    if (fd < 0 || fstat(fd, &st) != 0 ||
        st.st_size != expected_file_bytes) {
        if (fd >= 0) close(fd);
        return -1;
    }

    flat = (uint8_t *)malloc(flat_bytes);
    batch_capacity = (size_t)RESIDUAL_READ_BATCH_BLOCKS *
                     (size_t)RESIDUAL_BLOCK_SIZE;
    batch = (uint8_t *)malloc(batch_capacity);
    if (!flat || !batch) {
        free(flat);
        free(batch);
        close(fd);
        return -1;
    }

    for (first_block = 0; first_block < total_blocks;
         first_block += RESIDUAL_READ_BATCH_BLOCKS) {
        int block_count = total_blocks - first_block;
        int b;
        size_t read_bytes;
        off_t read_offset;

        if (block_count > RESIDUAL_READ_BATCH_BLOCKS) {
            block_count = RESIDUAL_READ_BATCH_BLOCKS;
        }
        read_bytes = (size_t)block_count * (size_t)RESIDUAL_BLOCK_SIZE;
        read_offset = (off_t)first_block * (off_t)RESIDUAL_BLOCK_SIZE;
        if (pread_full(fd, batch, read_bytes, read_offset) != 0) {
            free(flat);
            free(batch);
            close(fd);
            return -1;
        }

        for (b = 0; b < block_count; b++) {
            int block_idx = first_block + b;
            int first_vec = block_idx * vectors_per_block;
            int vector_count = dataset->num_vectors - first_vec;
            size_t used_bytes;

            if (vector_count > vectors_per_block) {
                vector_count = vectors_per_block;
            }
            used_bytes = (size_t)vector_count *
                         (size_t)dataset->residual_code_bytes;
            memcpy(flat + (size_t)first_vec *
                          (size_t)dataset->residual_code_bytes,
                   batch + (size_t)b * (size_t)RESIDUAL_BLOCK_SIZE,
                   used_bytes);
        }

        posix_fadvise(fd, read_offset, (off_t)read_bytes,
                      POSIX_FADV_DONTNEED);
    }

    free(batch);
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);
    dataset->residual_codes = flat;
    return 0;
}

static int load_fvecs_n(const char *path, int n, int dim, float **out) {
    FILE *fp = NULL;
    float *data = NULL;
    int i;

    if (!path || !out || n <= 0 || dim <= 0) {
        return -1;
    }

    fp = fopen(path, "rb");
    if (!fp) {
        return -1;
    }

    data = (float *)malloc((size_t)n * (size_t)dim * sizeof(float));
    if (!data) {
        fclose(fp);
        return -1;
    }

    for (i = 0; i < n; i++) {
        if (demo_read_fvec_row(fp, dim, data + (size_t)i * (size_t)dim) != 0) {
            fclose(fp);
            free(data);
            return -1;
        }
    }

    fclose(fp);
    *out = data;
    return 0;
}

static int load_ivecs_n(const char *path, int n, int dim, int32_t **out) {
    if (path && !*path && out && n>0 && dim>0) {
        *out=malloc((size_t)n*dim*sizeof(int32_t));
        if (!*out) return -1;
        memset(*out,0xff,(size_t)n*dim*sizeof(int32_t)); return 0;
    }
    FILE *fp = NULL;
    int32_t *data = NULL;
    int i;

    if (!path || !out || n <= 0 || dim <= 0) {
        return -1;
    }

    fp = fopen(path, "rb");
    if (!fp) {
        return -1;
    }

    data = (int32_t *)malloc((size_t)n * (size_t)dim * sizeof(int32_t));
    if (!data) {
        fclose(fp);
        return -1;
    }

    for (i = 0; i < n; i++) {
        int d = 0;
        if (fread(&d, sizeof(int), 1, fp) != 1 || d != dim ||
            fread(data + (size_t)i * (size_t)dim,
                  sizeof(int32_t),
                  (size_t)dim,
                  fp) != (size_t)dim) {
            fclose(fp);
            free(data);
            return -1;
        }
    }

    fclose(fp);
    for (size_t j=0;j<(size_t)n*dim;j++) data[j]=rover_original_to_index(data[j]);
    *out = data;
    return 0;
}

static void demo_free_loaded_data(void) {
    retrieval_demo_free_encoded_dataset(&g_dataset);
    demo_free_maybe_shared(g_graph, g_graph_uses_rpcmem);
    free(g_queries);
    free(g_gt);
    free(g_reorder_map);
    free(g_reverse_map);
    g_graph = NULL;
    g_queries = NULL;
    g_gt = NULL;
    g_graph_uses_rpcmem = 0;
    g_reorder_map = NULL;
    g_reverse_map = NULL;
}

static int rover_load_residual_map(void) {
    if(g_reorder_map || !*rover_config.residual_map)return 0;
    FILE *f=fopen(rover_config.residual_map,"r");
    uint8_t *seen=calloc((size_t)DATA_N_BASE,1);
    int32_t *map=malloc((size_t)DATA_N_BASE*sizeof(*map));int rc=-1;
    if(!f || !seen || !map)goto done;
    for(int i=0;i<DATA_N_BASE;i++) {
        int id;if(fscanf(f,"%d",&id)!=1 || id<0 || id>=DATA_N_BASE || seen[id])goto done;
        seen[id]=1;map[i]=id;
    }
    int extra;if(fscanf(f,"%d",&extra)!=EOF)goto done;
    g_reorder_map=map;map=NULL;rc=0;
done:if(f)fclose(f);free(seen);free(map);return rc;
}
