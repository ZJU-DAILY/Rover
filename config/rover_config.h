#ifndef ROVER_CONFIG_H
#define ROVER_CONFIG_H
#include <stdint.h>

typedef struct RoverConfig {
    int graph_entry, expand_count, rnn_s, rnn_t1;
    const char *build_distance;
    int gt_k;
    int parent_n;
    const char *parent_graph;
    int n, dim, nq, degree, sq_bits, residual_bits, pq_bits, pq_m, pq_compression;
    int disk, top_k, pool, entry_points, lanes, source_window, first_factor;
    int cache_pages, refine_cache_pages, refine_mode, refine_factor, refine_all;
    int third_residual, correction, train_samples, train_iters, seed, domain, unsigned_pd;
    int build_pool, build_candidates, build_iterations, build_threads, build_refine_candidates;
    int backlink_threads, checkpoint_every, reverse_candidates, split_distance;
    int cooldown_every, cooldown_ms;
    const char *command, *method, *search_mode, *build_method, *graph_format;
    const char *index_dir, *base, *query, *gt, *graph, *order_map, *output;
    const char *l1, *residual, *qparams, *error4, *bias, *residual_map;
    const char *pq_prefix, *pq_book, *pq_codes, *pq_residual, *pq_qparams;
    const char *result_dir, *metrics, *trace_path, *checkpoint, *identity;
    const char *layout, *layout_hash, *dataset;
    unsigned char error4_hash[32];
} RoverConfig;
extern RoverConfig rover_config;
int rover_config_parse(int argc, char **argv);
int rover_results_open(void);
int rover_write_result(int query, const int *ids, const float *distances, int count);
int rover_results_close(void);
int rover_original_to_index(int id);
const int32_t *rover_graph_to_original(void);
#endif
