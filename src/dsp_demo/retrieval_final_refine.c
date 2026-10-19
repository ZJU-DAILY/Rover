#define _POSIX_C_SOURCE 200809L
#include "retrieval_final_refine.h"
#include "retrieval_pq_artifact.h"
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define REFINE_BLOCK 4096
#define REFINE_HEADER 256
typedef struct { int32_t graph, original, row, position; double distance; } RefineItem;
typedef struct {
    unsigned references;
    int fd;
    float *qparams;
} RefineBacking;
struct RetrievalFinalRefineContext {
    RetrievalFinalRefineConfig c;
    int fd, code_bytes, vpb;
    uint64_t code_offset, artifact_bytes;
    float *qparams, *vector;
    int64_t *tags;
    uint8_t *blocks;
    RefineItem *items;
    RefineBacking *backing;
};
static uint32_t u32(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24;
}
static uint64_t u64(const uint8_t *p) { return u32(p) | (uint64_t)u32(p+4)<<32; }
static int read_at(int fd, void *buf, size_t bytes, uint64_t offset) {
    uint8_t *p = buf;
    while (bytes) {
        ssize_t n = pread(fd, p, bytes, (off_t)offset);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return RETRIEVAL_REFINE_IO;
        p += n; bytes -= (size_t)n; offset += (uint64_t)n;
    }
    return 0;
}

static float decode_value(unsigned code, float scale, float zero) {
    volatile float product = (float)code * scale;
    volatile float value = product + zero;
    return value;
}
static float add_f32(float a, float b) { volatile float v = a+b; return v; }
static unsigned code_at(const uint8_t *p, int bits, int d) {
    return bits == 4 ? (p[d/2] >> (4*(d&1))) & 15u : p[d];
}
int retrieval_final_refine_decode_sq(const uint8_t *l1, int lb, float ls, float lz,
    const uint8_t *r, int rb, float rs, float rz, int dim, float *out) {
    int d;
    if (!l1 || !r || !out || dim <= 0 || (lb != 4 && lb != 8) ||
        (rb != 4 && rb != 8) || !isfinite(ls) || ls <= 0 || !isfinite(lz) ||
        !isfinite(rs) || rs < 0 || !isfinite(rz)) return RETRIEVAL_REFINE_INVALID;
    for (d=0; d<dim; ++d) {
        out[d] = add_f32(decode_value(code_at(l1,lb,d),ls,lz),
                         decode_value(code_at(r,rb,d),rs,rz));
        if (!isfinite(out[d])) return RETRIEVAL_REFINE_NUMERIC;
    }
    return 0;
}
static int permutation(const int32_t *map, int n) {
    uint8_t *seen; int i;
    if (!map) return 0;
    seen = calloc(((size_t)n+7)/8,1);
    if (!seen) return -1;
    for (i=0; i<n; ++i) {
        int v=map[i];
        if (v<0 || v>=n || (seen[v/8] & (1u<<(v&7)))) { free(seen); return -1; }
        seen[v/8] |= (uint8_t)(1u<<(v&7));
    }
    free(seen); return 0;
}
void retrieval_final_refine_close(RetrievalFinalRefineContext *ctx) {
    if (!ctx) return;
    if (ctx->backing) {
        if (__atomic_sub_fetch(&ctx->backing->references, 1, __ATOMIC_ACQ_REL) == 0) {
            if (ctx->backing->fd >= 0) close(ctx->backing->fd);
            free(ctx->backing->qparams); free(ctx->backing);
        }
    } else {
        if (ctx->fd >= 0) close(ctx->fd);
        free(ctx->qparams);
    }
    free(ctx->vector); free(ctx->tags);
    free(ctx->blocks); free(ctx->items); free(ctx);
}
void retrieval_final_refine_reset(RetrievalFinalRefineContext *ctx) {
    int i;
    if (ctx && ctx->tags) for (i=0; i<ctx->c.cache_slots; ++i) ctx->tags[i]=-1;
}
static int refine_open(const RetrievalFinalRefineConfig *c,
                       const char *pq_codebook,
                       RetrievalFinalRefineContext **out) {
    RetrievalFinalRefineContext *ctx; uint8_t h[REFINE_HEADER], hash[32];
    uint64_t qb,cb; struct stat st; int i, rc=RETRIEVAL_REFINE_ARTIFACT;
    const uint32_t endian=1;
    if (!out) return RETRIEVAL_REFINE_INVALID;
    *out=NULL;
    if (!c || !c->decode || c->num_vectors<=0 || c->dim<=0 || c->dim>8192 ||
        c->capacity<=0 || c->capacity>1000000 || c->cache_slots<=0 || c->cache_slots>65536 ||
        (c->mode!=RETRIEVAL_REFINE_CURRENT && c->mode!=RETRIEVAL_REFINE_ERROR4) ||
        *(const uint8_t *)&endian!=1 || sizeof(float)!=4 ||
        permutation(c->graph_to_original,c->num_vectors) ||
        permutation(c->graph_to_storage,c->num_vectors)) return RETRIEVAL_REFINE_INVALID;
    ctx=calloc(1,sizeof(*ctx)); if (!ctx) return RETRIEVAL_REFINE_INVALID;
    ctx->c=*c; ctx->fd=-1; ctx->code_bytes=(c->dim+1)/2;
    ctx->vpb=REFINE_BLOCK/ctx->code_bytes;
    ctx->vector=malloc((size_t)c->dim*sizeof(float));
    ctx->items=malloc((size_t)c->capacity*sizeof(RefineItem));
    if (!ctx->vector || !ctx->items) goto fail;
    if (c->mode==RETRIEVAL_REFINE_ERROR4) {
        if (!c->artifact_path || !c->artifact_sha256) goto fail;
        ctx->fd=open(c->artifact_path,O_RDONLY);
        if (ctx->fd<0 || fstat(ctx->fd,&st) || st.st_size<REFINE_HEADER ||
            read_at(ctx->fd,h,sizeof(h),0)) goto fail;
        qb=(uint64_t)c->num_vectors*8;
        cb=((uint64_t)c->num_vectors+ctx->vpb-1)/ctx->vpb*REFINE_BLOCK;
        if (memcmp(h,pq_codebook ? "PQERR4V1" : "SQERR4V1",8) || u32(h+8)!=1 || u32(h+12)!=(uint32_t)c->num_vectors ||
            u32(h+16)!=(uint32_t)c->dim || (u32(h+20)!=4 && u32(h+20)!=8) ||
            (u32(h+24)!=4 && u32(h+24)!=8) || u32(h+28)!=REFINE_BLOCK ||
            u32(h+32)!=1 || u32(h+36)!=1 || u64(h+40)!=qb || u64(h+48)!=cb ||
            (uint64_t)st.st_size!=REFINE_HEADER+qb+cb) goto fail;
        if (retrieval_pq_artifact_hash_file_prefix(c->artifact_path,(uint64_t)st.st_size,hash) ||
            memcmp(hash,c->artifact_sha256,32) ||
            retrieval_pq_artifact_hash_file_range(c->artifact_path,REFINE_HEADER,qb+cb,hash) ||
            memcmp(hash,h+56,32)) goto fail;
        for (i=0;i<3;++i) {
            if (!c->base_paths[i] || stat(c->base_paths[i],&st) ||
                retrieval_pq_artifact_hash_file_prefix(c->base_paths[i],(uint64_t)st.st_size,hash) ||
                memcmp(hash,h+88+32*i,32)) goto fail;
        }
        if (pq_codebook && (stat(pq_codebook,&st) ||
            retrieval_pq_artifact_hash_file_prefix(pq_codebook,(uint64_t)st.st_size,hash) ||
            memcmp(hash,h+184,32))) goto fail;
        if (c->graph_to_original || c->graph_to_storage) {
            int32_t *maps=malloc((size_t)c->num_vectors*8);
            if (!maps) goto fail;
            for (i=0;i<c->num_vectors;++i) {
                maps[i]=c->graph_to_original ? c->graph_to_original[i]:i;
                maps[(size_t)c->num_vectors+i]=c->graph_to_storage ? c->graph_to_storage[i]:i;
            }
            rc=retrieval_pq_artifact_hash_bytes(maps,(size_t)c->num_vectors*8,hash);
            free(maps);
            if (rc || memcmp(hash,h+216,32)) { rc=RETRIEVAL_REFINE_ARTIFACT; goto fail; }
        } else {
            retrieval_pq_artifact_hash_bytes(NULL,0,hash);
            if (memcmp(hash,h+216,32)) goto fail;
        }
        rc=RETRIEVAL_REFINE_ARTIFACT;
        ctx->qparams=malloc((size_t)qb);
        ctx->tags=malloc((size_t)c->cache_slots*sizeof(int64_t));
        ctx->blocks=malloc((size_t)c->cache_slots*REFINE_BLOCK);
        if (!ctx->qparams || !ctx->tags || !ctx->blocks || read_at(ctx->fd,ctx->qparams,(size_t)qb,REFINE_HEADER)) goto fail;
        for (i=0;i<c->num_vectors;++i)
            if (!isfinite(ctx->qparams[2*(size_t)i]) || ctx->qparams[2*(size_t)i]<0 ||
                !isfinite(ctx->qparams[2*(size_t)i+1])) goto fail;
        ctx->code_offset=REFINE_HEADER+qb; ctx->artifact_bytes=REFINE_HEADER+qb+cb;
        retrieval_final_refine_reset(ctx);
    }
    ctx->backing=malloc(sizeof(*ctx->backing));
    if (!ctx->backing) goto fail;
    ctx->backing->references=1; ctx->backing->fd=ctx->fd;
    ctx->backing->qparams=ctx->qparams;
    *out=ctx; return 0;
fail:
    retrieval_final_refine_close(ctx); return rc;
}
int retrieval_final_refine_open(const RetrievalFinalRefineConfig *c,
                               RetrievalFinalRefineContext **out) {
    return refine_open(c,NULL,out);
}

int retrieval_final_refine_clone(const RetrievalFinalRefineContext *source,
    RetrievalRefineDecode decode, void *provider, int cache_slots,
    RetrievalFinalRefineContext **out) {
    RetrievalFinalRefineContext *ctx;
    if (!out) return RETRIEVAL_REFINE_INVALID;
    *out=NULL;
    if (!source || !source->backing || !decode || cache_slots<1 || cache_slots>65536)
        return RETRIEVAL_REFINE_INVALID;
    ctx=malloc(sizeof(*ctx)); if (!ctx) return RETRIEVAL_REFINE_INVALID;
    *ctx=*source; ctx->c.decode=decode; ctx->c.provider=provider;
    ctx->c.cache_slots=cache_slots;
    ctx->vector=NULL; ctx->items=NULL; ctx->tags=NULL; ctx->blocks=NULL;
    __atomic_add_fetch(&ctx->backing->references, 1, __ATOMIC_RELAXED);
    ctx->vector=malloc((size_t)ctx->c.dim*sizeof(float));
    ctx->items=malloc((size_t)ctx->c.capacity*sizeof(RefineItem));
    if (ctx->c.mode==RETRIEVAL_REFINE_ERROR4) {
        ctx->tags=malloc((size_t)cache_slots*sizeof(int64_t));
        ctx->blocks=malloc((size_t)cache_slots*REFINE_BLOCK);
        if (!ctx->tags || !ctx->blocks) goto fail_clone;
        retrieval_final_refine_reset(ctx);
    }
    if (!ctx->vector || !ctx->items) goto fail_clone;
    *out=ctx; return 0;
fail_clone:
    retrieval_final_refine_close(ctx); return RETRIEVAL_REFINE_INVALID;
}
int retrieval_final_refine_open_pq(const RetrievalFinalRefineConfig *c,
                                 const char *codebook_path,
                                 RetrievalFinalRefineContext **out) {
    if (!codebook_path) return RETRIEVAL_REFINE_INVALID;
    return refine_open(c,codebook_path,out);
}
static int by_row(const void *a,const void *b) {
    const RefineItem *x=a,*y=b; return (x->row>y->row)-(x->row<y->row);
}
static int by_distance(const void *a,const void *b) {
    const RefineItem *x=a,*y=b;
    if (x->distance!=y->distance) return x->distance<y->distance ? -1:1;
    return (x->original>y->original)-(x->original<y->original);
}

static int decode_vector(RetrievalFinalRefineContext *ctx,int id,int row,
                         RetrievalFinalRefineStats *stats) {
    const uint8_t *codes=NULL;
    int rc=ctx->c.decode(ctx->c.provider,id,ctx->vector);
    if(rc)return rc<0?rc:RETRIEVAL_REFINE_IO;
    if(ctx->c.mode==RETRIEVAL_REFINE_ERROR4) {
        int block=row/ctx->vpb,slot=block%ctx->c.cache_slots;
        if(ctx->tags[slot]!=block) {
            ctx->tags[slot]=-1;
            rc=read_at(ctx->fd,ctx->blocks+(size_t)slot*REFINE_BLOCK,REFINE_BLOCK,
                       ctx->code_offset+(uint64_t)block*REFINE_BLOCK);
            if(rc)return rc;
            ctx->tags[slot]=block;
            if(stats){++stats->code_blocks;stats->code_bytes+=REFINE_BLOCK;}
        } else if(stats)++stats->cache_hits;
        codes=ctx->blocks+(size_t)slot*REFINE_BLOCK+(row%ctx->vpb)*ctx->code_bytes;
    }
    for(int d=0;d<ctx->c.dim;++d) {
        float value=ctx->vector[d];
        if(codes)value=add_f32(value,decode_value(code_at(codes,4,d),
            ctx->qparams[2*(size_t)row],ctx->qparams[2*(size_t)row+1]));
        if(!isfinite(value))return RETRIEVAL_REFINE_NUMERIC;
        ctx->vector[d]=value;
    }
    return 0;
}
int retrieval_final_refine_read_vector(RetrievalFinalRefineContext *ctx,
                                      int id,float *out) {
    if(!ctx || !out || id<0 || id>=ctx->c.num_vectors)return RETRIEVAL_REFINE_INVALID;
    int row=ctx->c.graph_to_storage?ctx->c.graph_to_storage[id]:id;
    int rc=decode_vector(ctx,id,row,NULL);
    if(rc)return rc;
    memcpy(out,ctx->vector,(size_t)ctx->c.dim*sizeof(float));
    return 0;
}
static int refine_impl(RetrievalFinalRefineContext *ctx,const float *query,
    const int32_t *ids,int32_t *out_ids,float *distances,int count,int output_k,RetrievalFinalRefineStats *stats) {
    RetrievalFinalRefineStats s={0}; int i,d,rc;
    if (!ctx || !query || !ids || !distances || count<0 || count>ctx->c.capacity || output_k<=0)
        return RETRIEVAL_REFINE_INVALID;
    for (d=0;d<ctx->c.dim;++d) if (!isfinite(query[d])) return RETRIEVAL_REFINE_NUMERIC;
    for (i=0;i<count;++i) {
        RefineItem *v=&ctx->items[i]; int id=ids[i];
        if (id<0 || id>=ctx->c.num_vectors) return RETRIEVAL_REFINE_INVALID;
        v->position=i; v->graph=id; v->original=ctx->c.graph_to_original ? ctx->c.graph_to_original[id]:id;
        v->row=ctx->c.graph_to_storage ? ctx->c.graph_to_storage[id]:id;
    }
    qsort(ctx->items,(size_t)count,sizeof(RefineItem),by_row);
    for (i=1;i<count;++i) if (ctx->items[i].row==ctx->items[i-1].row) return RETRIEVAL_REFINE_INVALID;
    for (i=0;i<count;++i) {
        RefineItem *v=&ctx->items[i];
        rc=decode_vector(ctx,v->graph,v->row,&s); if(rc)return rc;
        v->distance=0;
        for (d=0;d<ctx->c.dim;++d) {
            float value=ctx->vector[d]; double delta;
            delta=(double)query[d]-(double)value; v->distance+=delta*delta;
        }
        if (!isfinite(v->distance) || v->distance>=FLT_MAX) return RETRIEVAL_REFINE_NUMERIC;
    }
    if(out_ids) qsort(ctx->items,(size_t)count,sizeof(RefineItem),by_distance);
    rc=count<output_k ? count:output_k;
    for (i=0;i<rc;++i) {
        if(out_ids){out_ids[i]=ctx->items[i].graph;distances[i]=(float)ctx->items[i].distance;}
        else distances[ctx->items[i].position]=(float)ctx->items[i].distance;
    }
    s.candidates=(uint64_t)count; s.artifact_bytes=ctx->artifact_bytes;
    s.qparams_bytes=ctx->qparams ? (uint64_t)ctx->c.num_vectors*8:0;
    s.resident_bytes=s.qparams_bytes+(ctx->blocks ? (uint64_t)ctx->c.cache_slots*(REFINE_BLOCK+8):0)
        +(uint64_t)ctx->c.capacity*sizeof(RefineItem)+(uint64_t)ctx->c.dim*4+sizeof(*ctx);
    if (stats) *stats=s;
    return rc;
}

int retrieval_final_refine(RetrievalFinalRefineContext *ctx,const float *query,
    int32_t *ids,float *distances,int count,int output_k,RetrievalFinalRefineStats *stats) {
    return refine_impl(ctx,query,ids,ids,distances,count,output_k,stats);
}
int retrieval_final_refine_score_ids(RetrievalFinalRefineContext *ctx,const float *query,
    const int32_t *ids,float *distances,int count) {
    return refine_impl(ctx,query,ids,NULL,distances,count,count?count:1,NULL);
}
