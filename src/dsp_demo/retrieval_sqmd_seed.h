#ifndef RETRIEVAL_SQMD_SEED_H
#define RETRIEVAL_SQMD_SEED_H

#ifdef __cplusplus
extern "C" {
#endif

int retrieval_sqmd_make_seed_ids(int num_vectors,
                                 int entry_points,
                                 int graph_entry,
                                 unsigned int seed,
                                 int *seed_ids,
                                 int seed_capacity);

#ifdef __cplusplus
}
#endif

#endif
