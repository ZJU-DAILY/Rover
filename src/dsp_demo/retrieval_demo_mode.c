#include "dsp_demo/retrieval_demo_mode.h"

#include <string.h>

static void retrieval_demo_db_mode_set_db_pair(RetrievalDemoDbModeConfig *cfg) {
    cfg->mode = RETRIEVAL_DEMO_DB_MODE_DB_PAIR;
    cfg->mode_name = "db_pair";
    cfg->pipeline_slot_count = DEMO_DB_PAIR_MAX_CONCURRENT * 2;
    cfg->db_pair_worker_count = DEMO_DB_PAIR_MAX_CONCURRENT;
    cfg->sqmd_lane_count = 0;
    cfg->uses_db_pair_worker = 1;
    cfg->uses_sqmd_worker = 0;
}

static void retrieval_demo_db_mode_set_sqmd(RetrievalDemoDbModeConfig *cfg) {
    cfg->mode = RETRIEVAL_DEMO_DB_MODE_SQMD;
    cfg->mode_name = "sqmd";
    cfg->pipeline_slot_count = 1;
    cfg->db_pair_worker_count = 0;
    cfg->sqmd_lane_count = rover_config.lanes;
    cfg->uses_db_pair_worker = 0;
    cfg->uses_sqmd_worker = 1;
}

static void retrieval_demo_db_mode_set_pure_pq(
        RetrievalDemoDbModeConfig *cfg) {
    cfg->mode = RETRIEVAL_DEMO_DB_MODE_PURE_PQ;
    cfg->mode_name = "pure_pq";
    cfg->pipeline_slot_count = 1;
    cfg->db_pair_worker_count = 0;
    cfg->sqmd_lane_count = PURE_PQ_SQMD_LANES;
    cfg->uses_db_pair_worker = 0;
    cfg->uses_sqmd_worker = 1;
}

int retrieval_demo_db_mode_resolve(const char *env_value,
                                   RetrievalDemoDbModeConfig *cfg) {
    if (!cfg)
        return -1;

    if (!env_value || env_value[0] == '\0' ||
        strcmp(env_value, "db") == 0 ||
        strcmp(env_value, "db_pair") == 0) {
        retrieval_demo_db_mode_set_db_pair(cfg);
        return 0;
    }

    if (strcmp(env_value, "sqmd") == 0) {
#if SQMD_ENABLE
        retrieval_demo_db_mode_set_sqmd(cfg);
        return 0;
#else
        return -1;
#endif
    }

    if (strcmp(env_value, "pure_pq") == 0) {
#if SQMD_ENABLE
        retrieval_demo_db_mode_set_pure_pq(cfg);
        return 0;
#else
        return -1;
#endif
    }

    return -1;
}
