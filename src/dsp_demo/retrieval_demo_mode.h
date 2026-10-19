#ifndef RETRIEVAL_DEMO_MODE_H
#define RETRIEVAL_DEMO_MODE_H

#include "dsp_demo/retrieval_demo_params.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum RetrievalDemoDbMode {
    RETRIEVAL_DEMO_DB_MODE_DB_PAIR = 0,
    RETRIEVAL_DEMO_DB_MODE_SQMD = 1,
    RETRIEVAL_DEMO_DB_MODE_PURE_PQ = 2
} RetrievalDemoDbMode;

typedef struct RetrievalDemoDbModeConfig {
    RetrievalDemoDbMode mode;
    const char *mode_name;
    int pipeline_slot_count;
    int db_pair_worker_count;
    int sqmd_lane_count;
    int uses_db_pair_worker;
    int uses_sqmd_worker;
} RetrievalDemoDbModeConfig;

int retrieval_demo_db_mode_resolve(const char *env_value,
                                   RetrievalDemoDbModeConfig *cfg);

#ifdef __cplusplus
}
#endif

#endif
