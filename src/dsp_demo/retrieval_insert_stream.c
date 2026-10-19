#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64
#include "retrieval_insert_stream.h"
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

struct RetrievalInsertStream {
    int fd,n,dim,current;
    float *query;
    RetrievalFinalRefineContext *encoded;
    RetrievalInsertStreamStats stats;
};
static int read_exact(int fd,void *out,size_t bytes,off_t offset) {
    unsigned char *p=out;
    while(bytes) {
        ssize_t got=pread(fd,p,bytes,offset);
        if(got<0 && errno==EINTR)continue;
        if(got<=0)return -1;
        p+=got;bytes-=(size_t)got;offset+=got;
    }
    return 0;
}
int retrieval_insert_stream_open(const char *path,int n,int dim,
    RetrievalFinalRefineContext *encoded,RetrievalInsertStream **out) {
    if(!out)return -1;
    *out=NULL;
    if(!path || !encoded || n<1 || dim<1 || dim>8192)return -1;
    RetrievalInsertStream *s=calloc(1,sizeof(*s));
    if(!s)return -1;
    s->fd=open(path,O_RDONLY);s->current=-1;
    struct stat st;
    if(s->fd<0 || fstat(s->fd,&st) || st.st_size!=(int64_t)n*(dim+1)*4)goto fail;
    s->query=malloc((size_t)dim*sizeof(float));
    if(!s->query)goto fail;
    s->n=n;s->dim=dim;s->encoded=encoded;*out=s;return 0;
fail:
    retrieval_insert_stream_close(s);return -1;
}
int retrieval_insert_stream_begin(RetrievalInsertStream *s,int id) {
    if(!s || id<0 || id>=s->n || id<=s->current)return -1;
    int32_t dim;
    off_t offset=(off_t)id*(s->dim+1)*4;
    float query[s->dim];
    if(read_exact(s->fd,&dim,sizeof(dim),offset) || dim!=s->dim ||
       read_exact(s->fd,query,(size_t)s->dim*sizeof(float),offset+4))return -1;
    for(int d=0;d<s->dim;d++)if(!isfinite(query[d]))return -1;
    memcpy(s->query,query,(size_t)s->dim*sizeof(float));
    s->current=id;s->stats.raw_rows++;return 0;
}
const float *retrieval_insert_stream_query(const RetrievalInsertStream *s) {
    return s && s->current>=0?s->query:NULL;
}
int retrieval_insert_stream_read(void *ctx,int id,float *out) {
    RetrievalInsertStream *s=ctx;
    if(!s || !out || id<0 || id>s->current)return -1;
    if(id==s->current) {
        memcpy(out,s->query,(size_t)s->dim*sizeof(float));
        s->stats.current_vector_reads++;return 0;
    }
    int rc=retrieval_final_refine_read_vector(s->encoded,id,out);
    if(!rc)s->stats.old_vector_decodes++;
    return rc;
}
void retrieval_insert_stream_stats(const RetrievalInsertStream *s,
                                   RetrievalInsertStreamStats *out) {
    if(out)*out=s?s->stats:(RetrievalInsertStreamStats){0};
}
void retrieval_insert_stream_close(RetrievalInsertStream *s) {
    if(!s)return;
    if(s->fd>=0)close(s->fd);
    free(s->query);free(s);
}
