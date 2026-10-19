static int reorder_validate(const int32_t *map, const int32_t *rev, int N) {
    int i;
    for (i = 0; i < N; i++) {
        if (map[i] < 0 || map[i] >= N) return -1;
        if (rev[i] < 0 || rev[i] >= N) return -1;
        if (rev[map[i]] != i) return -1;
    }
    return 0;
}


static int sort_by_degree_desc(int N, int K, const int *in_degree, int *order) {
    int *bucket_count = NULL;
    int *bucket_start = NULL;
    int max_deg = 0;
    int i;

    for (i = 0; i < N; i++) {
        int d = K + in_degree[i];
        if (d > max_deg) max_deg = d;
    }
    bucket_count = (int *)calloc((size_t)(max_deg + 1), sizeof(int));
    bucket_start = (int *)malloc((size_t)(max_deg + 1) * sizeof(int));
    if (!bucket_count || !bucket_start) {
        free(bucket_count); free(bucket_start);
        return -1;
    }
    for (i = 0; i < N; i++)
        bucket_count[K + in_degree[i]]++;


    bucket_start[max_deg] = 0;
    for (i = max_deg - 1; i >= 0; i--)
        bucket_start[i] = bucket_start[i + 1] + bucket_count[i + 1];

    for (i = 0; i < N; i++) {
        int d = K + in_degree[i];
        order[bucket_start[d]++] = i;
    }
    free(bucket_count); free(bucket_start);
    return 0;
}


static int compute_in_degree(const int32_t *graph, int N, int K, int *in_degree) {
    int i, j;
    memset(in_degree, 0, (size_t)N * sizeof(int));
    for (i = 0; i < N; i++) {
        const int32_t *nb = graph + (size_t)i * (size_t)K;
        for (j = 0; j < K; j++) {
            int v = nb[j];
            if (v >= 0 && v < N && v != i) in_degree[v]++;
        }
    }
    return 0;
}


static double reorder_now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}


#include "retrieval_rpc_demo_reorder_scheme_b.c"


#include "retrieval_rpc_demo_reorder_scheme_c.c"


typedef struct ReorderBucketNode {
    int prev;
    int next;
} ReorderBucketNode;

static void reorder_bucket_remove(int v, const int *priority, int *heads,
                                  ReorderBucketNode *nodes) {
    int prev = nodes[v].prev, next = nodes[v].next;
    if (prev >= 0) nodes[prev].next = next;
    else heads[priority[v]] = next;
    if (next >= 0) nodes[next].prev = prev;
    nodes[v].prev = nodes[v].next = -1;
}

static void reorder_bucket_move(int v, int score, int *priority, int *heads,
                                ReorderBucketNode *nodes) {
    reorder_bucket_remove(v, priority, heads, nodes);
    priority[v] = score;
    nodes[v].next = heads[score];
    if (heads[score] >= 0) nodes[heads[score]].prev = v;
    heads[score] = v;
}

static int reorder_scheme_d(const int32_t *graph, int N, int K, int vpb,
                            int32_t *reorder_map, int32_t *reverse_map) {
    int *placed = NULL;
    int *priority = NULL;
    int *bucket_head = NULL;
    ReorderBucketNode *nodes = NULL;
    int *window = NULL;
    int max_prio, pos, win_cap;
    int i;

    placed      = (int *)calloc((size_t)N, sizeof(int));
    priority    = (int *)calloc((size_t)N, sizeof(int));
    nodes       = (ReorderBucketNode *)malloc((size_t)N * sizeof(ReorderBucketNode));
    win_cap     = vpb > 0 ? vpb : 64;
    if (win_cap > N) win_cap = N;
    bucket_head = (int *)malloc(((size_t)win_cap + 1) * sizeof(int));
    window      = (int *)malloc((size_t)win_cap * sizeof(int));
    if (!placed || !priority || !bucket_head || !nodes || !window)
        goto fail_d;


    for (i = 0; i <= win_cap; i++) bucket_head[i] = -1;
    for (i = 0; i < N; i++) {
        nodes[i].prev = -1;
        nodes[i].next = bucket_head[0];
        if (bucket_head[0] >= 0) nodes[bucket_head[0]].prev = i;
        bucket_head[0] = i;
    }
    max_prio = 0;

    pos = 0;
    while (pos < N) {
        while (max_prio >= 0 && bucket_head[max_prio] < 0) max_prio--;
        if (max_prio < 0) goto fail_d;
        int v = bucket_head[max_prio];
        reorder_bucket_remove(v, priority, bucket_head, nodes);
        reorder_map[v] = pos;
        reverse_map[pos] = v;
        placed[v] = 1;


        if (pos >= win_cap) {
            int old_v = window[pos % win_cap];
            {
                int j;
                const int32_t *nb = graph + (size_t)old_v * (size_t)K;
                for (j = 0; j < K; j++) {
                    int w = nb[j];
                    if (w >= 0 && w < N && !placed[w] && priority[w] > 0)
                        reorder_bucket_move(w, priority[w] - 1, priority, bucket_head, nodes);
                }
            }
        }
        window[pos % win_cap] = v;


        {
            int j;
            const int32_t *nb = graph + (size_t)v * (size_t)K;
            for (j = 0; j < K; j++) {
                int w = nb[j];
                if (w >= 0 && w < N && !placed[w]) {
                    if (priority[w] >= win_cap) goto fail_d;
                    reorder_bucket_move(w, priority[w] + 1, priority, bucket_head, nodes);
                    if (priority[w] > max_prio)
                        max_prio = priority[w];
                }
            }
        }

        pos++;
    }

    free(placed); free(priority); free(bucket_head);
    free(nodes); free(window);
    return 0;

fail_d:
    free(placed); free(priority); free(bucket_head);
    free(nodes); free(window);
    return -1;
}


static int reorder_compute(const int32_t *graph, int N, int K, int vpb,
                           int scheme,
                           int32_t **reorder_map_out,
                           int32_t **reverse_map_out) {
    int32_t *rmap = NULL;
    int32_t *rev  = NULL;
    int rc;

    if (!graph || N <= 0 || K <= 0 || vpb <= 0 ||
        !reorder_map_out || !reverse_map_out)
        return -1;

    rmap = (int32_t *)malloc((size_t)N * sizeof(int32_t));
    rev  = (int32_t *)malloc((size_t)N * sizeof(int32_t));
    if (!rmap || !rev) { free(rmap); free(rev); return -1; }

    switch (scheme) {
    case 0: {
        int i;
        for (i = 0; i < N; i++) { rmap[i] = i; rev[i] = i; }
        rc = 0;
        break;
    }
    case 1: rc = reorder_scheme_b(graph, N, K, vpb, rmap, rev); break;
    case 2: rc = reorder_scheme_c(graph, N, K, vpb, rmap, rev); break;
    case 3: rc = reorder_scheme_d(graph, N, K, vpb, rmap, rev); break;
    default:
        printf("  ERROR: unknown reorder scheme %d\n", scheme);
        free(rmap); free(rev);
        return -1;
    }

    if (rc != 0) { free(rmap); free(rev); return -1; }

    if (reorder_validate(rmap, rev, N) != 0) {
        printf("  ERROR: reorder map validation failed\n");
        free(rmap); free(rev);
        return -1;
    }

    *reorder_map_out = rmap;
    *reverse_map_out = rev;
    return 0;
}
