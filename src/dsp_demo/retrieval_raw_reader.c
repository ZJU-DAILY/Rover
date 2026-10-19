#define _FILE_OFFSET_BITS 64
#include "dsp_demo/retrieval_raw_reader.h"
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
struct RetrievalRawReader { int fd,n,dim; size_t slots; pthread_key_t key; };
typedef struct { int32_t *ids; float *vectors; } RawCache;
static void raw_free(void *ptr) {
    RawCache *c=ptr; if(c){free(c->ids);free(c->vectors);free(c);}
}
int retrieval_raw_open(const char *path,int n,int dim,size_t budget,RetrievalRawReader **out) {
    struct stat st; RetrievalRawReader *r;
    if(!out||!path||n<1||dim<1||dim>8192||budget<(size_t)dim*4+4)return -1;
    *out=NULL; r=calloc(1,sizeof(*r)); if(!r)return -1;
    r->fd=open(path,O_RDONLY);
    if(r->fd<0||fstat(r->fd,&st)||st.st_size!=(int64_t)n*(dim+1)*4||pthread_key_create(&r->key,raw_free)) {
        if(r->fd>=0)close(r->fd);free(r);return -1;
    }
    r->n=n;r->dim=dim;r->slots=budget/((size_t)dim*4+4);*out=r;return 0;
}
int retrieval_raw_read(RetrievalRawReader *r,int id,float *out) {
    RawCache *c; size_t slot,bytes; unsigned char *dst; off_t offset; int32_t dim;
    if(!r||!out||id<0||id>=r->n)return -1;
    c=pthread_getspecific(r->key);
    if(!c){
        c=calloc(1,sizeof(*c));if(!c)return -1;
        c->ids=malloc(r->slots*sizeof(*c->ids));c->vectors=malloc(r->slots*r->dim*4);
        if(!c->ids||!c->vectors||pthread_setspecific(r->key,c)){raw_free(c);return -1;}
        memset(c->ids,0xff,r->slots*sizeof(*c->ids));
    }
    slot=(size_t)(uint32_t)id%r->slots;
    if(c->ids[slot]!=id){
        offset=(off_t)id*(r->dim+1)*4;
        if(pread(r->fd,&dim,4,offset)!=4||dim!=r->dim)return -1;
        bytes=(size_t)r->dim*4;dst=(unsigned char *)out;offset+=4;
        while(bytes){ssize_t got=pread(r->fd,dst,bytes,offset);if(got<0&&errno==EINTR)continue;
            if(got<=0)return -1;bytes-=got;dst+=got;offset+=got;}
        for(int d=0;d<r->dim;d++)if(!isfinite(out[d]))return -1;
        memcpy(c->vectors+slot*r->dim,out,(size_t)r->dim*4);c->ids[slot]=id;
    }else memcpy(out,c->vectors+slot*r->dim,(size_t)r->dim*4);
    return 0;
}
int retrieval_raw_distances(void *ptr,const int32_t *a,const int32_t *b,int count,float *out) {
    RetrievalRawReader *r=ptr;if(!r||count<0||!a||!b||!out)return -1;
    float va[r->dim],vb[r->dim];
    for(int i=0;i<count;i++){
        float sum=0;
        if(retrieval_raw_read(r,a[i],va)||retrieval_raw_read(r,b[i],vb))return -1;

        for(int d=0;d<r->dim;d++){float diff=va[d]-vb[d];sum+=diff*diff;}
        out[i]=sum;
    }return 0;
}
void retrieval_raw_close(RetrievalRawReader *r) {
    if(!r)return;raw_free(pthread_getspecific(r->key));pthread_setspecific(r->key,NULL);
    pthread_key_delete(r->key);close(r->fd);free(r);
}
