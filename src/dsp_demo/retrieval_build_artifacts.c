#define _FILE_OFFSET_BITS 64
#include "dsp_demo/retrieval_build_artifacts.h"
#include "dsp_demo/retrieval_pq_artifact.h"
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
static int ba_write(int fd,const void *buf,size_t n,off_t offset) {
    const unsigned char *p=buf;while(n){ssize_t w=pwrite(fd,p,n,offset);
        if(w<0&&errno==EINTR)continue;if(w<=0)return -1;p+=w;n-=w;offset+=w;}return 0;
}
int retrieval_build_hash(const char *p,unsigned char hash[32]) {
    struct stat st;if(stat(p,&st)||st.st_size<0)return -1;
    return retrieval_pq_artifact_hash_file_prefix(p,st.st_size,hash);
}
static void ba_u32(unsigned char *p,uint32_t v){memcpy(p,&v,4);}
static void ba_u64(unsigned char *p,uint64_t v){memcpy(p,&v,8);}

static float ba_sub(float a,float b){volatile float x=a-b;return x;}
static float ba_mul(float a,float b){volatile float x=a*b;return x;}
int retrieval_build_error4(const char *output,RetrievalRawReader *raw,int n,int dim,
    RetrievalRefineDecode decode,void *provider,const char *paths[3],const char *identity,int pq) {
    if(!output||!raw||!decode||n<1||dim<1||dim>8192)return -1;
    int fd=-1,rc=-1,cb=(dim+1)/2,vpb=4096/cb;
    uint64_t qb=(uint64_t)n*8,blocks=((uint64_t)n+vpb-1)/vpb;
    unsigned char header[256]={0},block[4096];
    float *vec=NULL,*recon=NULL,*params=NULL;
    if(!output||!raw||!decode||n<1||dim<1||dim>8192||!vpb)return -1;
    fd=open(output,O_RDWR|O_CREAT|O_EXCL,0644);if(fd<0)return -1;
    vec=malloc((size_t)dim*4);recon=malloc((size_t)dim*4);params=malloc((size_t)vpb*8);
    if(!vec||!recon||!params||ftruncate(fd,256+qb+blocks*4096))goto done;
    for(int first=0;first<n;first+=vpb){
        int count=n-first;if(count>vpb)count=vpb;memset(block,0,4096);
        for(int row=0;row<count;row++){
            float lo=INFINITY,hi=-INFINITY,scale,inv;
            if(retrieval_raw_read(raw,first+row,vec)||decode(provider,first+row,recon))goto done;
            for(int d=0;d<dim;d++){vec[d]=ba_sub(vec[d],recon[d]);if(!isfinite(vec[d]))goto done;
                if(vec[d]<lo)lo=vec[d];if(vec[d]>hi)hi=vec[d];}
            scale=fmaxf(ba_sub(hi,lo),1e-12f)/15.0f;inv=1.0f/scale;
            params[2*row]=scale;params[2*row+1]=lo;
            for(int d=0;d<dim;d++){
                volatile float x=ba_mul(ba_sub(vec[d],lo),inv)+0.5f;
                int code=(int)floorf(x);if(code<0)code=0;if(code>15)code=15;
                block[row*cb+d/2]|=(unsigned char)(code<<((d%2)*4));
            }
        }
        if(ba_write(fd,params,(size_t)count*8,256+(uint64_t)first*8)||
           ba_write(fd,block,4096,256+qb+(uint64_t)(first/vpb)*4096))goto done;
        if(first%100000< vpb){printf("BUILD15_ARTIFACT error4_rows=%d/%d\n",first+count,n);fflush(stdout);}
    }
    memcpy(header,pq?"PQERR4V1":"SQERR4V1",8);
    ba_u32(header+8,1);ba_u32(header+12,n);ba_u32(header+16,dim);ba_u32(header+20,pq?8:4);
    ba_u32(header+24,4);ba_u32(header+28,4096);ba_u32(header+32,1);ba_u32(header+36,1);
    ba_u64(header+40,qb);ba_u64(header+48,blocks*4096);
    if(fsync(fd)||retrieval_pq_artifact_hash_file_range(output,256,qb+blocks*4096,header+56))goto done;
    for(int i=0;i<3;i++)if(retrieval_build_hash(paths[i],header+88+i*32))goto done;
    if(retrieval_build_hash(identity,header+184)||retrieval_pq_artifact_hash_bytes(NULL,0,header+216)||
       ba_write(fd,header,256,0)||fsync(fd))goto done;
    rc=0;
done:
    if(fd>=0&&close(fd))rc=-1;free(vec);free(recon);free(params);return rc;
}
