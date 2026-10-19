#include <stdlib.h>
#include "calculator.h"
#include "HAP_farf.h"
int rover_dsp_l1_bits=4;
#include "dsp_demo/retrieval_l1_int8_core.c"
#include "dsp_demo/retrieval_async_dsp.c"

AEEResult calculator_rover_configure(remote_handle64 h,int32 version,int32 bits) {
    (void)h;
#ifdef RETRIEVAL_ENABLE_FULL_BUILD
    const int expected_abi=2;
#else
    const int expected_abi=1;
#endif
    if(version!=expected_abi || (bits!=4 && bits!=8))return AEE_EBADPARM;
    rover_dsp_l1_bits=bits;return 0;
}
int calculator_open(const char *uri,remote_handle64 *handle) {
    (void)uri;if(!handle)return AEE_EBADPARM;
    void *state=calloc(1,1);if(!state)return AEE_ENOMEMORY;
    *handle=(remote_handle64)(uintptr_t)state;
    ++g_retrieval_async_demo_open_count;return 0;
}
int calculator_close(remote_handle64 handle) {
    if(handle) {
        if(g_retrieval_async_demo_open_count>0)--g_retrieval_async_demo_open_count;
        if(!g_retrieval_async_demo_open_count)calculator_retrieval_async_reset(handle);
        free((void*)(uintptr_t)handle);
    }
    return 0;
}
