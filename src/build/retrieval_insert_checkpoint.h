#ifndef RETRIEVAL_INSERT_CHECKPOINT_H
#define RETRIEVAL_INSERT_CHECKPOINT_H
#include "build/rover_insert_build_core.h"

typedef struct RetrievalInsertCheckpoint {
    const char *path;
    const char *identity;
    uint64_t interval;
    uint64_t resumed_vectors, save_us, load_us, saves;
} RetrievalInsertCheckpoint;

int retrieval_insert_checkpoint_load(RetrievalInsertCheckpoint *, int total, int dim,
    const RoverInsertBuildConfig *, RetrievalRnnGraph *, RoverInsertBuildStats *);
int retrieval_insert_checkpoint_save(RetrievalInsertCheckpoint *, int total,
    const RoverInsertBuildConfig *, const RetrievalRnnGraph *, const RoverInsertBuildStats *);
int rover_insert_build_insert_checkpointed(const float *, int, int,
    const RoverInsertBuildConfig *, const RoverInsertBuildRuntime *,
    RetrievalInsertCheckpoint *, RetrievalRnnGraph *, RoverInsertBuildStats *);
#endif
