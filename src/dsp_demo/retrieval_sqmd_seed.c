#include "dsp_demo/retrieval_sqmd_seed.h"

#include <stdint.h>

static int sqmd_seed_contains(const int *seed_ids, int count, int id) {
    int index;

    for (index = 0; index < count; index++) {
        if (seed_ids[index] == id)
            return 1;
    }
    return 0;
}

int retrieval_sqmd_make_seed_ids(int num_vectors,
                                 int entry_points,
                                 int graph_entry,
                                 unsigned int seed,
                                 int *seed_ids,
                                 int seed_capacity) {
    uint64_t attempt_limit;
    uint64_t attempts = 0;
    int target;
    int count = 0;

    if (num_vectors <= 0 || entry_points <= 0 || !seed_ids ||
        seed_capacity <= 0)
        return -1;

    target = entry_points;
    if (target > seed_capacity)
        target = seed_capacity;
    if (target > num_vectors)
        target = num_vectors;

    if (graph_entry >= 0 && graph_entry < num_vectors)
        seed_ids[count++] = graph_entry;

    attempt_limit = (uint64_t)target * 64u + 1024u;
    while (count < target && attempts < attempt_limit) {
        int entry_id;

        seed = seed * 1103515245u + 12345u;
        entry_id = (int)((seed / 65536u) % (unsigned int)num_vectors);
        attempts++;
        if (sqmd_seed_contains(seed_ids, count, entry_id))
            continue;
        seed_ids[count++] = entry_id;
    }


    if (count < target) {
        int entry_id;

        for (entry_id = 0; entry_id < num_vectors && count < target;
             entry_id++) {
            if (!sqmd_seed_contains(seed_ids, count, entry_id))
                seed_ids[count++] = entry_id;
        }
    }

    return count;
}
