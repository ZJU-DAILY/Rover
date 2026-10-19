#ifndef RETRIEVAL_DEMO_INTERNAL_TYPES_H
#define RETRIEVAL_DEMO_INTERNAL_TYPES_H

typedef struct DemoCandidate {
    int   id;
    float distance;
    int   expanded;
} DemoCandidate;

typedef struct DemoFrontierEntry {
    int   id;
    float distance;
} DemoFrontierEntry;

#endif
