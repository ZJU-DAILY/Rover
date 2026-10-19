#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "calculator.h"
#include "remote.h"
#include "rpcmem.h"
#include "rover_config.h"
#include "dsp_demo/retrieval_rpc_demo.h"
int main(int argc,char **argv) {
    int rc=rover_config_parse(argc,argv);
    if(rc)return rc>0?0:2;
    remote_handle64 handle=0;
    char uri[1024];
    const char *domains[]={"adsp","mdsp","sdsp","cdsp"};
    if(rover_config.domain<0 || rover_config.domain>3)return 2;
    struct remote_rpc_control_unsigned_module pd={0};
    pd.domain=rover_config.domain;pd.enable=rover_config.unsigned_pd;
    rc=remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,&pd,sizeof(pd));
    if(rc){fprintf(stderr,"FastRPC PD configuration failed: %d\n",rc);return 1;}
    snprintf(uri,sizeof(uri),"%s&_dom=%s",calculator_URI,domains[rover_config.domain]);
    rpcmem_init();rc=calculator_open(uri,&handle);
    if(rc)goto done;
#ifdef ROVER_ENABLE_BUILD
    const int abi=2;
#else
    const int abi=1;
#endif
    rc=calculator_rover_configure(handle,abi,rover_config.sq_bits);
    if(rc)goto done;
    if(!strcmp(rover_config.command,"search")) {
        printf("ROVER_EVALUATION enabled=%d\n",*rover_config.gt!=0);
        rc=rover_results_open();
        if(!rc)rc=run_retrieval_search_rpc_demo_selected(handle,rover_config.domain,rover_config.search_mode);
        if(rover_results_close())rc=-1;
    } else {
#ifdef ROVER_ENABLE_BUILD
        rc=rover_run_build(handle,rover_config.domain);
#else
        fprintf(stderr,"Construction is available on graph-build-insert.\n");rc=-1;
#endif
    }
done:
    if(handle){int close_rc=calculator_close(handle);if(!rc)rc=close_rc;}
    rpcmem_deinit();if(rc)fprintf(stderr,"Rover failed: %d\n",rc);return rc?1:0;
}
