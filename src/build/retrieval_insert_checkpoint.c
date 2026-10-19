#include "build/retrieval_insert_checkpoint.h"
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef struct InsertCheckpointHeader {
    char magic[8];
    uint32_t version, endian, header_bytes, stats_bytes;
    uint32_t total, dim, degree, inserted;
    RoverInsertBuildConfig config;
    char identity[65];
} InsertCheckpointHeader;

static uint64_t ic_now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return (uint64_t)t.tv_sec*1000000+(uint64_t)t.tv_nsec/1000;
}
static uint64_t ic_checksum(uint64_t h,const void *data,size_t n) {
    const unsigned char *p=data;
    for(size_t i=0;i<n;i++) { h^=p[i]; h*=UINT64_C(1099511628211); }
    return h;
}
static int ic_write(FILE *f,const void *data,size_t n,uint64_t *hash) {
    *hash=ic_checksum(*hash,data,n);return fwrite(data,1,n,f)==n?0:-1;
}
static int ic_read(FILE *f,void *data,size_t n,uint64_t *hash) {
    if(fread(data,1,n,f)!=n)return -1;
    *hash=ic_checksum(*hash,data,n);return 0;
}
static int ic_identity(const char *s) {
    if(!s || strlen(s)!=64)return -1;
    for(int i=0;i<64;i++)if(!((s[i]>='0' && s[i]<='9') || (s[i]>='a' && s[i]<='f')))return -1;
    return 0;
}
static int ic_valid(const RetrievalRnnGraph *g) {
    int expected=g->num_vectors<=g->max_degree?g->num_vectors-1:g->max_degree;
    for(int i=0;i<g->num_vectors;i++) {
        if(g->degrees[i]!=expected)return -1;
        for(int j=0;j<expected;j++) {
            size_t s=(size_t)i*g->max_degree+j;int v=g->neighbors[s];
            if(v<0 || v>=g->num_vectors || v==i || !isfinite(g->neighbor_dists[s]) || g->neighbor_dists[s]<0)return -1;
            for(int k=0;k<j;k++)if(g->neighbors[(size_t)i*g->max_degree+k]==v)return -1;
        }
    }
    return 0;
}
int retrieval_insert_checkpoint_save(RetrievalInsertCheckpoint *cp,int total,
    const RoverInsertBuildConfig *cfg,const RetrievalRnnGraph *g,const RoverInsertBuildStats *stats) {
    uint64_t begin=ic_now(),hash=UINT64_C(14695981039346656037);
    InsertCheckpointHeader h={0};char *tmp=NULL,*parent=NULL;FILE *f=NULL;int fd=-1,rc=-1;
    if(!cp || !cp->path || ic_identity(cp->identity) || !cfg || !g || !stats ||
       sizeof(int)!=4 || sizeof(float)!=4 || g->num_vectors<1 || g->num_vectors>total ||
       stats->inserted_vectors!=(uint64_t)g->num_vectors || stats->provider_failures || ic_valid(g))goto done;
    memcpy(h.magic,"RVICP001",8);h.version=1;h.endian=0x01020304;h.header_bytes=sizeof(h);h.stats_bytes=sizeof(*stats);
    h.total=total;h.dim=g->dim;h.degree=g->max_degree;h.inserted=g->num_vectors;h.config=*cfg;
    memcpy(h.identity,cp->identity,65);
    tmp=malloc(strlen(cp->path)+5);parent=strdup(cp->path);if(!tmp || !parent)goto done;
    sprintf(tmp,"%s.tmp",cp->path);
    char *slash=strrchr(parent,'/');if(slash) {if(slash==parent)slash[1]=0;else *slash=0;}else strcpy(parent,".");
    fd=open(parent,O_RDONLY|O_DIRECTORY);if(fd<0)goto done;
    f=fopen(tmp,"wb");if(!f)goto done;
    size_t rows=g->num_vectors,edges=rows*(size_t)g->max_degree;
    if(ic_write(f,&h,sizeof(h),&hash) || ic_write(f,stats,sizeof(*stats),&hash) ||
       ic_write(f,g->degrees,rows*sizeof(int),&hash) ||
       ic_write(f,g->neighbors,edges*sizeof(int),&hash) ||
       ic_write(f,g->neighbor_dists,edges*sizeof(float),&hash) ||
       fwrite(&hash,1,sizeof(hash),f)!=sizeof(hash) || fflush(f) || fsync(fileno(f)))goto done;
    if(fclose(f)) {f=NULL;goto done;}f=NULL;
    if(rename(tmp,cp->path) || fsync(fd))goto done;
    cp->saves++;rc=0;
done:
    if(f)fclose(f);if(fd>=0)close(fd);
    if(rc && tmp)unlink(tmp);
    free(tmp);free(parent);
    if(cp)cp->save_us+=ic_now()-begin;
    if(rc)fprintf(stderr,"INSERT_CHECKPOINT save failed\n");
    return rc;
}
int retrieval_insert_checkpoint_load(RetrievalInsertCheckpoint *cp,int total,int dim,
    const RoverInsertBuildConfig *cfg,RetrievalRnnGraph *g,RoverInsertBuildStats *stats) {
    uint64_t begin=ic_now(),hash=UINT64_C(14695981039346656037),stored;
    InsertCheckpointHeader h;struct stat st;FILE *f=NULL;int rc=-1;
    if(!cp || !cp->path || ic_identity(cp->identity) || !cfg || !g || !stats || total<=0 || dim<=0 || sizeof(int)!=4 || sizeof(float)!=4)return -1;
    f=fopen(cp->path,"rb");if(!f)return errno==ENOENT?1:-1;
    if(ic_read(f,&h,sizeof(h),&hash) || memcmp(h.magic,"RVICP001",8) || h.version!=1 || h.endian!=0x01020304 ||
       h.header_bytes!=sizeof(h) || h.stats_bytes!=sizeof(*stats) || h.total!=(uint32_t)total || h.dim!=(uint32_t)dim ||
       h.degree!=(uint32_t)cfg->r || !h.inserted || h.inserted>(uint32_t)total ||
       memcmp(&h.config,cfg,sizeof(*cfg)) || memcmp(h.identity,cp->identity,65))goto done;
    uint64_t bytes=sizeof(h)+sizeof(*stats)+(uint64_t)h.inserted*(4+8*(uint64_t)h.degree)+8;
    if(fstat(fileno(f),&st) || (uint64_t)st.st_size!=bytes ||
       retrieval_rnn_graph_alloc_without_flags(g,total,dim,cfg->r))goto done;
    g->num_vectors=h.inserted;
    size_t rows=h.inserted,edges=rows*(size_t)h.degree;
    if(ic_read(f,stats,sizeof(*stats),&hash) || ic_read(f,g->degrees,rows*sizeof(int),&hash) ||
       ic_read(f,g->neighbors,edges*sizeof(int),&hash) || ic_read(f,g->neighbor_dists,edges*sizeof(float),&hash) ||
       fread(&stored,1,sizeof(stored),f)!=sizeof(stored) || stored!=hash ||
       stats->inserted_vectors!=rows || stats->provider_failures || ic_valid(g))goto done;
    cp->resumed_vectors=rows;rc=0;
done:
    fclose(f);cp->load_us+=ic_now()-begin;
    if(rc) {retrieval_rnn_graph_free(g);memset(stats,0,sizeof(*stats));fprintf(stderr,"INSERT_CHECKPOINT invalid or incompatible; refusing fresh restart\n");}
    return rc;
}
