#ifndef RETRIEVAL_BUILD_PAUSE_H
#define RETRIEVAL_BUILD_PAUSE_H
#include <stdint.h>

#define RETRIEVAL_BUILD_PAUSE_SUPERVISOR 8
#define RETRIEVAL_BUILD_PAUSE_PARTICIPANTS 9
#define RETRIEVAL_BUILD_PAUSE_BYTES (10u * 128u)
typedef struct __attribute__((aligned(128))) {
    volatile int32_t value;
    uint8_t reserved[124];
} RetrievalBuildPauseSlot;
typedef struct {
    RetrievalBuildPauseSlot request;
    RetrievalBuildPauseSlot ack[RETRIEVAL_BUILD_PAUSE_PARTICIPANTS];
} RetrievalBuildPause;
_Static_assert(sizeof(RetrievalBuildPause)==RETRIEVAL_BUILD_PAUSE_BYTES,"pause ABI size");
#endif
