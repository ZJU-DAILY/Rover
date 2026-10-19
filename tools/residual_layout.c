#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <sys/time.h>
#include "rover_limits.h"
#include "dsp_demo/retrieval_rpc_demo_reorder.c"
int main(int argc,char **argv) {
    if(argc!=5){fprintf(stderr,"Usage: rover-residual-layout graph.rover scheme(1..3) vectors_per_block out_map.txt\n");return 2;}
    int scheme=atoi(argv[2]),vpb=atoi(argv[3]),h[4],*graph=NULL,*map=NULL,*reverse=NULL,rc=1;
    FILE *f=fopen(argv[1],"rb"),*out=NULL;
    if(!f)return 1;
    if(fread(h,sizeof(h),1,f)!=1 || h[0]!=0x35314252 || h[1]<1 || h[3]<1 || scheme<1 || scheme>3 || vpb<1)goto done;
    graph=malloc((size_t)h[1]*h[3]*sizeof(int));
    if(!graph || fread(graph,sizeof(int),(size_t)h[1]*h[3],f)!=(size_t)h[1]*h[3] || fgetc(f)!=EOF)goto done;
    for(size_t i=0;i<(size_t)h[1]*h[3];i++)if(graph[i]<-1 || graph[i]>=h[1])goto done;
    if(reorder_compute(graph,h[1],h[3],vpb,scheme,&map,&reverse))goto done;
    out=fopen(argv[4],"wx");if(!out)goto done;
    for(int i=0;i<h[1];i++)if(fprintf(out,"%d\n",map[i])<0)goto done;
    rc=0;
done:if(f)fclose(f);if(out && fclose(out))rc=1;free(graph);free(map);free(reverse);return rc;
}
