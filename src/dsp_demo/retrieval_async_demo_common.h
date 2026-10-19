#ifndef RETRIEVAL_ASYNC_DEMO_COMMON_H
#define RETRIEVAL_ASYNC_DEMO_COMMON_H

#include <stdint.h>
#include "dsp_demo/retrieval_demo_params.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum RetrievalAsyncDemoStatus {
    RETRIEVAL_ASYNC_DEMO_STATUS_PENDING   = 0,
    RETRIEVAL_ASYNC_DEMO_STATUS_RUNNING   = 1,
    RETRIEVAL_ASYNC_DEMO_STATUS_DONE      = 2,
    RETRIEVAL_ASYNC_DEMO_STATUS_FAILED    = 3,
    RETRIEVAL_ASYNC_DEMO_STATUS_NOT_FOUND = 4
} RetrievalAsyncDemoStatus;

#ifdef __cplusplus
}
#endif

#endif
