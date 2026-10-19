#include "dsp_demo/retrieval_search_build_core.h"
#include "dsp_demo/retrieval_raw_reader.h"
#include "dsp_demo/retrieval_build_artifacts.h"
#include "dsp_demo/retrieval_build_live_graph.h"
#include <time.h>
#include <dlfcn.h>
static void b15_phase(const char *name){
    void (*snapshot)(const char *)=dlsym(RTLD_DEFAULT,"rover_mem_snapshot");
    void (*mark)(const char *)=dlsym(RTLD_DEFAULT,"rover_mem_mark");
    if(snapshot)snapshot(name);if(mark)mark(name);
    printf("BUILD15_PHASE %s\n",name);fflush(stdout);
}

typedef struct {
    ResidualBlockCache *sq;
    RetrievalFinalRefineContext *refine;
} Build15RefineWorker;
typedef struct {
    int pq,n,dim,session_open,forward_refine_limit;
    RetrievalBuildLiveGraph live_graph;
    RetrievalGraphReader graph_reader;
    uint64_t refined_candidates,returned_candidates;
    Build15RefineWorker refine_workers[4];
    uint64_t pq_hits,pq_misses,pq_corrected;
    uint64_t worker_refines[4],worker_scores[4];
    unsigned active_refines,peak_refines;
    pthread_mutex_t mu;
    pthread_t thread;
    RetrievalRawReader *raw;
    ResidualBlockCache *sq_cache,*decode_cache;
    RetrievalRpcSqmdSessionArgs sq_session;
    PurePqIndex *index;
    RetrievalPqResidualIndex *residual;
    RetrievalPqResidualQuery *query;
    PurePqSharedRuntime shared;
    PurePqSessionArgs pq_session;
    RetrievalFinalRefineContext *refine;
    uint64_t calls;
    uint64_t search_us,refine_us,raw_query_us,lock_wait_worker_us;
    int profile_limit;
    int profile_counts[4];
    uint64_t profile_result_xor;
    int pause_cursor[8];
    int pause_acks,resume_acks;
    uint64_t pause_wait_us,resume_wait_us;
} Build15Provider;
static uint64_t b15_now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000+t.tv_nsec/1000;}
static int b15_sync(const char *path){int fd=open(path,O_RDONLY);if(fd<0)return -1;int r=fsync(fd);close(fd);return r;}
static int b15_search_candidates(Build15Provider *p,const int32_t *graph,int r,const float *q,
                     const RetrievalSearchBuildConfig *cfg,int k,int *ids,float *dist){
    RetrievalDemoConfig c={0};int count;uint64_t search_begin=b15_now();
    c.entry_points=cfg->entry_points;c.expand_count=cfg->expand_count;c.pool_size=cfg->pool_size;
    c.top_k=cfg->pool_size;c.rerank_factor=1;
    if(p->pq){
        uint64_t work=0;RetrievalPqResidualStats rs={0};PurePqSourcePipeline pipe;
        count=pure_pq_run_one_query_with_reader(p->index,graph,r,0,q,&c,&p->shared,&p->pq_session,
                                  rover_config.correction?p->query:NULL,ids,dist,c.pool_size,&work,&rs,rover_config.source_window,&pipe,&p->graph_reader);
        p->pq_hits+=rs.cache_hits;p->pq_misses+=rs.cache_misses;p->pq_corrected+=rs.corrected_candidates;
        if(rs.cache_errors)count=-1;
    }else{
        DemoResult metrics={0};
        count=sqmd_search_one_query_session_with_reader(&g_dataset,graph,r,0,q,&c,p->sq_cache,&g_sqmd_shared,
                          &p->sq_session,rover_config.source_window,rover_config.split_distance,&metrics,ids,dist,c.pool_size,&p->graph_reader);
        if(metrics.sqmd_abort_count||__atomic_load_n(&g_residual_stats.io_error,__ATOMIC_ACQUIRE))count=-1;
    }
    p->search_us+=b15_now()-search_begin;
    if(count<0||count>cfg->pool_size)return -1;
    int limit=p->forward_refine_limit?p->forward_refine_limit:2*k;
    if(count>limit)count=limit;
    return count;
}
static int b15_refine_search(Build15Provider *p,RetrievalFinalRefineContext *refine,
    const float *q,int *ids,float *dist,int count,int k) {
    uint64_t begin=b15_now();
    unsigned active=__atomic_add_fetch(&p->active_refines,1,__ATOMIC_RELAXED);
    unsigned peak=__atomic_load_n(&p->peak_refines,__ATOMIC_RELAXED);
    while(active>peak&&!__atomic_compare_exchange_n(&p->peak_refines,&peak,active,0,
        __ATOMIC_RELAXED,__ATOMIC_RELAXED)){}
    int result=retrieval_final_refine(refine,q,ids,dist,count,k,NULL);
    if(result>=0) {
        __atomic_fetch_add(&p->refined_candidates,(uint64_t)count,__ATOMIC_RELAXED);
        __atomic_fetch_add(&p->returned_candidates,(uint64_t)result,__ATOMIC_RELAXED);
    }
    __atomic_sub_fetch(&p->active_refines,1,__ATOMIC_RELAXED);
    __atomic_fetch_add(&p->refine_us,b15_now()-begin,__ATOMIC_RELAXED);
    return result;
}
static int b15_search(Build15Provider *p,const int32_t *graph,int r,const float *q,
    const RetrievalSearchBuildConfig *cfg,int k,int *ids,float *dist) {
    int count=b15_search_candidates(p,graph,r,q,cfg,k,ids,dist);
    return count<0?count:b15_refine_search(p,p->refine,q,ids,dist,count,k);
}

static int b15_row_worker(const Build15Provider *p,int id) {
    for(int w=0;w<rover_config.build_threads;w++) if(id<(int)((int64_t)p->n*(w+1)/rover_config.build_threads)) return w;
    return rover_config.build_threads-1;
}
static int b15_provider(void *ptr,const float *base,int n,int dim,const RetrievalSearchBuildConfig *cfg,
    const RetrievalRnnGraph *graph,int id,int cap,RetrievalSearchBuildCandidate *out,int *out_count){
    Build15Provider *p=ptr;float query[dim],dist[cfg->pool_size];int ids[cfg->pool_size],count;
    uint64_t raw_begin=b15_now(),lock_begin;int profile_worker=-1;
    if(p->profile_limit) {

        for(int w=0;w<4;w++)if(id>=(int)((int64_t)n*w/4)&&id<(int)((int64_t)n*(w+1)/4))profile_worker=w;
        if(profile_worker<0 || id-(int)((int64_t)n*profile_worker/4)>=p->profile_limit/4)return -1;
    }
    (void)base;(void)n;if(retrieval_raw_read(p->raw,id,query))return -1;
    lock_begin=b15_now();pthread_mutex_lock(&p->mu);
    if(p->profile_limit && p->calls>=(unsigned)p->profile_limit){pthread_mutex_unlock(&p->mu);return -1;}
    p->lock_wait_worker_us+=b15_now()-lock_begin;p->raw_query_us+=lock_begin-raw_begin;
    if(!p->pq) {
        count=b15_search_candidates(p,graph->neighbors,graph->max_degree,query,cfg,cap,ids,dist);
        pthread_mutex_unlock(&p->mu);
        if(count>=0) count=b15_refine_search(p,p->refine_workers[b15_row_worker(p,id)].refine,
            query,ids,dist,count,cap);
        pthread_mutex_lock(&p->mu);
    } else count=b15_search(p,graph->neighbors,graph->max_degree,query,cfg,cap,ids,dist);
    if(count>=0&&p->profile_limit) {
        ++p->profile_counts[profile_worker];
        p->profile_result_xor^=sqmd_fnv1a64(ids,(size_t)count*sizeof(*ids)) +
            3u*sqmd_fnv1a64(dist,(size_t)count*sizeof(*dist)) +
            (uint64_t)(uint32_t)id*UINT64_C(0x9e3779b97f4a7c15);
    }
    if(count>=0){++p->worker_refines[b15_row_worker(p,id)];++p->calls;if(p->calls%10000==0 || (p->profile_limit && p->calls==(unsigned)p->profile_limit)){printf("BUILD15_PROGRESS queries=%llu total=%llu\n",
        (unsigned long long)p->calls,(unsigned long long)p->n*cfg->iterations);
        printf("BUILD15_COST queries=%llu search_us=%llu refine_us=%llu raw_query_worker_us=%llu lock_wait_worker_us=%llu",
            (unsigned long long)p->calls,(unsigned long long)p->search_us,(unsigned long long)__atomic_load_n(&p->refine_us,__ATOMIC_RELAXED),
            (unsigned long long)p->raw_query_us,(unsigned long long)p->lock_wait_worker_us);
        if(!p->pq)printf(" dsp_wait_us=%llu correction_us=%llu work_items=%llu residual_hits=%llu residual_misses=%llu",
            (unsigned long long)g_sqmd_shared.ctrl->stats.sqmd_cpu_wait_dsp_us,
            (unsigned long long)g_sqmd_shared.ctrl->stats.sqmd_cpu_correct_us,
            (unsigned long long)g_sqmd_shared.ctrl->stats.sqmd_work_items,
            (unsigned long long)p->sq_cache->hits,(unsigned long long)p->sq_cache->misses);
        else printf(" pq_hits=%llu pq_misses=%llu pq_corrected=%llu",
            (unsigned long long)p->pq_hits,(unsigned long long)p->pq_misses,(unsigned long long)p->pq_corrected);
        printf("\n");fflush(stdout);}}
    else {
        RetrievalBuildPause *pause=p->pq?&p->shared.control->build_pause:&g_sqmd_shared.ctrl->build_pause;
        fprintf(stderr,"BUILD15_SEARCH_FAILED completed=%llu row=%d method=%s pause_request=%d pause_acks=",
                (unsigned long long)p->calls,id,p->pq?"pq":"sq",
                __atomic_load_n(&pause->request.value,__ATOMIC_ACQUIRE));
        for(int i=0;i<RETRIEVAL_BUILD_PAUSE_PARTICIPANTS;i++)
            fprintf(stderr,"%s%d",i?",":"",__atomic_load_n(&pause->ack[i].value,__ATOMIC_ACQUIRE));
        if(!p->pq)fprintf(stderr," query_gen=%d rpc_done=%d rpc_ret=%d",
                         p->sq_session.query_gen,p->sq_session.done,p->sq_session.ret);
        fprintf(stderr,"\n");fflush(stderr);
    }
    pthread_mutex_unlock(&p->mu);if(count<0)return -1;
    for(int i=0;i<count;i++){out[i].id=ids[i];out[i].search_dist=dist[i];out[i].exact_dist=0;}
    *out_count=count;return 0;
}

static int b15_prepare_iteration(void *ctx,const float *base,int n,int dim,
    const RetrievalSearchBuildConfig *cfg,const RetrievalRnnGraph *graph,int iteration) {
    Build15Provider *p=ctx;
    (void)base;(void)n;(void)dim;(void)cfg;(void)iteration;
    if(retrieval_live_graph_bind(&p->live_graph,graph))return -1;
    p->graph_reader=(RetrievalGraphReader){retrieval_live_graph_read,&p->live_graph};
    return 0;
}
static int b15_read_vector(void *ctx,int id,float *out) {
    return retrieval_raw_read((RetrievalRawReader *)ctx,id,out);
}
static int b15_score_candidates(void *ctx,int query_id,const int32_t *ids,int count,float *scores) {
    Build15Provider *p=ctx;float query[p->dim];
    if(retrieval_raw_read(p->raw,query_id,query))return -1;

    ++p->worker_scores[b15_row_worker(p,query_id)];
    int result;
    if(!p->pq) {
        result=retrieval_final_refine_score_ids(p->refine_workers[b15_row_worker(p,query_id)].refine,
            query,ids,scores,count);
    } else {
        pthread_mutex_lock(&p->mu);
        result=retrieval_final_refine_score_ids(p->refine,query,ids,scores,count);
        pthread_mutex_unlock(&p->mu);
    }
    return result==count?0:-1;
}

static int b15_quiesce(void *ctx,int pause_requested) {
    Build15Provider *p=ctx;
    RetrievalBuildPause *shared=p->pq?&p->shared.control->build_pause:&g_sqmd_shared.ctrl->build_pause;
    uint64_t begin=b15_now(),deadline=begin+5000000;
    if(pause_requested)for(int i=0;i<rover_config.lanes;i++)p->pause_cursor[i]=p->pq?p->shared.control->lanes[i].cpu_fill_buffer:g_sqmd_shared.ctrl->lanes[i].cpu_fill_buf;
    __atomic_store_n(&shared->request.value,pause_requested,__ATOMIC_RELEASE);
    for(;;) {
        int count=0,error=0;
        for(int i=0;i<RETRIEVAL_BUILD_PAUSE_PARTICIPANTS;i++) {
            if(i>=rover_config.lanes && i!=RETRIEVAL_BUILD_PAUSE_SUPERVISOR)continue;
            int v=__atomic_load_n(&shared->ack[i].value,__ATOMIC_ACQUIRE);
            if(v<0)error=1;
            if(v==pause_requested)++count;
        }
        if(!error&&count==rover_config.lanes+1) {
            for(int i=0;i<rover_config.lanes;i++) {
                int cursor=p->pq?p->shared.control->lanes[i].cpu_fill_buffer:g_sqmd_shared.ctrl->lanes[i].cpu_fill_buf;
                if(cursor!=p->pause_cursor[i]) {
                    __atomic_store_n(&shared->request.value,0,__ATOMIC_RELEASE);
                    fprintf(stderr,"BUILD15_PAUSE_FAILED FIFO cursor changed while drained lane=%d before=%d after=%d request=%d\n",i,p->pause_cursor[i],cursor,pause_requested);
                    return -1;
                }
            }
            if(pause_requested){p->pause_acks=count;p->pause_wait_us=b15_now()-begin;}
            else{p->resume_acks=count;p->resume_wait_us=b15_now()-begin;}
            return 0;
        }
        if(error||b15_now()>=deadline) {
            __atomic_store_n(&shared->request.value,0,__ATOMIC_RELEASE);
            fprintf(stderr,"BUILD15_PAUSE_FAILED request=%d ack_count=%d dsp_error=%d\n",pause_requested,count,error);
            return -1;
        }
        struct timespec t={0,1000000};nanosleep(&t,NULL);
    }
}
typedef struct { FILE *file; int n; uint64_t requested_us; Build15Provider *provider; } Build15Cooling;
static int b15_sensor(const char *path){int v=-1;FILE *f=fopen(path,"r");if(f){if(fscanf(f,"%d",&v)!=1)v=-1;fclose(f);}return v;}
static void b15_cooling(void *ctx,uint64_t queries,uint64_t begin,uint64_t end){
    Build15Cooling *c=ctx;FILE *f=c->file;fprintf(f,"{\"queries\":%llu,\"start_us\":%llu,\"end_us\":%llu,\"actual_us\":%llu,\"inflight\":0,\"dsp_pause_acks\":%d,\"dsp_resume_acks\":%d,\"dsp_pause_wait_us\":%llu,\"dsp_resume_wait_us\":%llu,\"iteration\":%llu,\"requested_us\":%llu,\"battery_percent\":%d,\"battery_temp_tenths_c\":%d}\n",
        (unsigned long long)queries,(unsigned long long)begin,(unsigned long long)end,(unsigned long long)(end-begin),c->provider->pause_acks,c->provider->resume_acks,(unsigned long long)c->provider->pause_wait_us,(unsigned long long)c->provider->resume_wait_us,(unsigned long long)((queries-1)/c->n+1),(unsigned long long)c->requested_us,
        b15_sensor("/sys/class/power_supply/battery/capacity"),b15_sensor("/sys/class/power_supply/battery/temp"));
    fflush(f);printf("BUILD15_COOLDOWN queries=%llu rest_us=%llu\n",(unsigned long long)queries,(unsigned long long)(end-begin));fflush(stdout);
}
static int b15_graph_write(const char *path,const RetrievalRnnGraph *g){
    uint32_t h[4]={0x35314252,(uint32_t)g->num_vectors,(uint32_t)g->dim,(uint32_t)g->max_degree};
    FILE *f=fopen(path,"wbx");int ret=-1;if(!f)return -1;
    for(int i=0;i<g->num_vectors;i++) {
        if(g->degrees[i]<0 || g->degrees[i]>g->max_degree)goto done;
        for(int j=g->degrees[i];j<g->max_degree;j++)
            if(g->neighbors[(size_t)i*g->max_degree+j]!=-1)goto done;
    }
    if(fwrite(h,sizeof(h),1,f)!=1||fwrite(g->neighbors,(size_t)g->num_vectors*g->max_degree*4,1,f)!=1)goto done;
    ret=0;
done:if(fclose(f))ret=-1;if(!ret&&b15_sync(path))ret=-1;return ret;
}
static int b15_graph_read(const char *path,int n,int dim,int r,int32_t **out){
    FILE *f=fopen(path,"rb");uint32_t h[4];int32_t *g=NULL;int ret=-1;struct stat st;
    if(!f)return -1;
    if(fread(h,sizeof(h),1,f)!=1||h[0]!=0x35314252||h[1]!=(unsigned)n||h[2]!=(unsigned)dim||h[3]!=(unsigned)r||
       fstat(fileno(f),&st)||st.st_size!=16+(int64_t)n*r*4)goto done;
    g=malloc((size_t)n*r*4);if(!g||fread(g,(size_t)n*r*4,1,f)!=1)goto done;
    for(int i=0;i<n;i++)for(int j=0;j<r;j++){
        int v=g[(size_t)i*r+j];
        if(v==-1)continue;
        if(v<0||v>=n||v==i||(j && g[(size_t)i*r+j-1]==-1))goto done;
        for(int k=0;k<j;k++)if(g[(size_t)i*r+k]==v)goto done;
    }*out=g;g=NULL;ret=0;
done:free(g);fclose(f);return ret;
}

#include "build/rover_build_driver.c"
