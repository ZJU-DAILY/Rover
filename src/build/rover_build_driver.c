#include "build/rover_insert_build_core.h"
#include "build/retrieval_insert_checkpoint.h"
#include "dsp_demo/retrieval_insert_stream.h"
#include <sys/resource.h>

static void rover_build_report(uint64_t begin,uint64_t ready,uint64_t built,uint64_t saved,uint64_t cooldown) {
    struct rusage usage={0};getrusage(RUSAGE_SELF,&usage);
    printf("ROVER_BUILD_METRICS {\"method\":\"%s\",\"load_setup_us\":%llu,\"build_wall_us\":%llu,"
        "\"cooldown_us\":%llu,\"build_net_us\":%llu,\"save_us\":%llu,\"device_total_us\":%llu,"
        "\"cpu_peak_rss_kib\":%ld,\"user_cpu_s\":%.6f,\"system_cpu_s\":%.6f}\n",
        rover_config.build_method,(unsigned long long)(ready-begin),(unsigned long long)(built-ready),
        (unsigned long long)cooldown,(unsigned long long)(built-ready-cooldown),
        (unsigned long long)(saved-built),(unsigned long long)(saved-begin),usage.ru_maxrss,
        usage.ru_utime.tv_sec+usage.ru_utime.tv_usec/1e6,usage.ru_stime.tv_sec+usage.ru_stime.tv_usec/1e6);
    fflush(stdout);
}

typedef struct RoverStreamProvider {
    Build15Provider *engine;
    RetrievalInsertStream *vectors;
    PurePqCpuSearchState *workspace;
} RoverStreamProvider;

static int rover_insert_begin(void *ctx,const float *base,int n,int dim,
    const RoverInsertBuildConfig *cfg,const RetrievalRnnGraph *graph,int id) {
    (void)base;(void)dim;(void)cfg;(void)graph;(void)id;
    RoverStreamProvider *p=ctx;
    return retrieval_insert_stream_begin(p->vectors,n-1);
}
static int rover_insert_search(void *ctx,const float *base,int n,int dim,
    const RoverInsertBuildConfig *cfg,const RetrievalRnnGraph *graph,int id,
    int cap,RoverInsertBuildCandidate *out,int *count) {
    (void)base;(void)dim;
    RoverStreamProvider *s=ctx;Build15Provider *p=s->engine;
    const float *query=retrieval_insert_stream_query(s->vectors);
    if(!query || id!=n-1 || cap<cfg->candidate_count)return -1;
    int ids[cap];float distances[cap];int found;
    RetrievalDemoConfig search={.pool_size=cfg->pool_size,.entry_points=cfg->entry_points,
        .expand_count=cfg->expand_count,.top_k=cfg->candidate_count<n-1?cfg->candidate_count:n-1,.rerank_factor=1};
    if(p->pq) {
        RetrievalPqResidualStats stats={0};
        found=pure_pq_run_one_query_limited(p->index,graph->neighbors,graph->max_degree,0,
            query,&search,&p->shared,&p->pq_session,rover_config.correction?p->query:NULL,ids,distances,cap,
            NULL,&stats,rover_config.source_window,NULL,NULL,n-1,&s->workspace);
        if(stats.cache_errors)return -1;
    } else {
        RetrievalDemoEncodedDataset prefix=g_dataset;
        prefix.num_vectors=n-1;
        DemoResult metrics={0};
        found=sqmd_search_one_query_session(&prefix,graph->neighbors,graph->max_degree,0,
            query,&search,p->sq_cache,&g_sqmd_shared,&p->sq_session,
            rover_config.source_window,rover_config.split_distance,&metrics,ids,distances,cap);
        if(metrics.sqmd_abort_count || g_residual_stats.io_error)return -1;
    }
    if(found<cfg->r)return -1;
    if(found>cfg->candidate_count)found=cfg->candidate_count;
    for(int i=0;i<found;i++) {
        if(ids[i]<0 || ids[i]>=id || !isfinite(distances[i]))return -1;
        out[i].id=ids[i];out[i].search_dist=distances[i];out[i].exact_dist=FLT_MAX;
    }
    *count=found;return 0;
}

static int rover_save_graph(const char *path,const RetrievalRnnGraph *g) {
    if(access(path,F_OK)==0)return -1;
    char temporary[4096];
    if(snprintf(temporary,sizeof(temporary),"%s.tmp",path)>=(int)sizeof(temporary))return -1;
    if(unlink(temporary) && errno!=ENOENT)return -1;
    if(b15_graph_write(temporary,g) || rename(temporary,path))return -1;
    return b15_sync(rover_config.index_dir);
}

static int rover_rnn_build(remote_handle64 handle,int domain_id) {
    RoverConfig *c=&rover_config;float *base=NULL;RetrievalRnnGraph graph={0};
    RetrievalRnnBuildConfig cfg={.s=c->rnn_s,.r=c->degree,.t1=c->rnn_t1,
        .t2=c->build_iterations,.seed=c->seed};
    RetrievalRnnBuildRuntime rt={.threads=c->build_threads,.batch_size=8192,.min_batch=512,.progress_every_t2=1};
    RetrievalRnnBuildStats stats={0};int rc=-1;
    uint64_t begin=b15_now(),ready=0,built=0;
    int shared=0,fd=-1,mapped=0,loaded=0;DemoRnnDspDistanceCtx dsp={0};
    size_t bytes=(size_t)c->n*c->dim*sizeof(float);
    if(!strcmp(c->build_distance,"dsp")) {
        base=demo_alloc_shared_buffer(bytes,"rnn_base",&shared);
        if(!base || !shared || !fastrpc_mmap || demo_load_fvecs_n_into(c->base,c->n,c->dim,base))goto done;
        fd=rpcmem_to_fd(base);if(fd<0 || fastrpc_mmap(domain_id,fd,base,0,bytes,FASTRPC_MAP_FD))goto done;
        mapped=1;if(calculator_retrieval_rnn_l2_load_dma(handle,fd,c->n,c->dim))goto done;loaded=1;
        if(pthread_mutex_init(&dsp.mu,NULL))goto done;dsp.mu_ready=1;
        dsp.handle=handle;dsp.base=base;dsp.dim=c->dim;dsp.fallback_on_error=0;
        rt.distance_batch=demo_rnn_dsp_distance_batch;rt.distance_ctx=&dsp;
    } else if(strcmp(c->build_distance,"cpu") || load_fvecs_n(c->base,c->n,c->dim,&base))goto done;
    ready=b15_now();
    if(retrieval_rnn_build_graph_ex(base,c->n,c->dim,&cfg,&rt,&graph,&stats))goto done;
    built=b15_now();if(rover_save_graph(c->graph,&graph))goto done;
    printf("ROVER_BUILD method=rnn-descent n=%d dim=%d degree=%d distance_pairs=%llu\n",
        c->n,c->dim,c->degree,(unsigned long long)stats.distance_calls);
    rover_build_report(begin,ready,built,b15_now(),0);rc=0;
done:demo_rnn_l2_provider_cleanup(handle,domain_id,base,shared,fd,mapped,loaded,bytes,&dsp);
    retrieval_rnn_graph_free(&graph);return rc;
}

int rover_run_build(remote_handle64 handle,int domain_id) {
    RoverConfig *c=&rover_config;int rc=-1,stats_ready=0,mapped=0;
    Build15Provider p={0};RetrievalRnnGraph graph={0},parent={0};
    RoverStreamProvider stream={.engine=&p};
    RetrievalFinalRefineConfig fc={0};
    const char *paths[3];uint64_t begin=b15_now(),ready=0,built=0,cooldown=0;
    FILE *cool=NULL;Build15Cooling cooling={0};
    if(!*c->base || !*c->graph || !*c->index_dir || c->build_threads<1 || c->build_threads>64 ||
        c->build_iterations<1 || c->backlink_threads<1 || c->backlink_threads>4)return -1;
    if(c->cooldown_every && (strcmp(c->build_method,"search-build") || c->cooldown_ms<=0))return -1;
    if((!strcmp(c->command,"insert") && (!c->parent_n || !*c->parent_graph)) ||
       (strcmp(c->command,"insert") && c->parent_n))return -1;
    if((!strcmp(c->command,"insert") || !strcmp(c->command,"resume")) && strcmp(c->build_method,"stream-insert"))return -1;
    if(!strcmp(c->build_method,"rnn-descent")) {
        if(c->rnn_s>c->degree)return -1;
        return rover_rnn_build(handle,domain_id);
    }
    if(c->build_threads>4)return -1;
    if(c->build_candidates<c->degree || c->build_candidates>c->build_pool || c->build_pool>2000 ||
       c->reverse_candidates>c->build_candidates)return -1;
    int streaming=!strcmp(c->build_method,"stream-insert");
    if(!streaming && strcmp(c->build_method,"search-build"))return -1;
    if(streaming && c->build_threads!=1)return -1;
    if(!strcmp(c->command,"resume") && (!streaming || !*c->checkpoint || access(c->checkpoint,F_OK)))return -1;
    if(c->parent_n && (!streaming || c->parent_n<=c->degree || c->parent_n>=c->n))return -1;
#if !RETRIEVAL_ENABLE_BUILD_PAUSE
    if(c->cooldown_every)return -1;
#endif
    p.pq=!strcmp(c->method,"pq");p.n=c->n;p.dim=c->dim;
    p.shared.control_fd=p.shared.lut_fd=p.shared.domain_id=-1;
    pthread_mutex_init(&p.mu,NULL);
    if(retrieval_live_graph_init(&p.live_graph))goto done;
    p.forward_refine_limit=c->build_refine_candidates?c->build_refine_candidates:c->build_pool;
    if(p.forward_refine_limit<c->build_candidates || p.forward_refine_limit>c->build_pool)goto done;
    if(!streaming && retrieval_raw_open(c->base,c->n,c->dim,16u*1024*1024,&p.raw))goto done;
    if(p.pq) {
        PurePqPrepareRequest req={.mode=PURE_PQ_LOAD_ONLY,.base_fvecs_path=c->base,
            .artifact_prefix=c->pq_prefix,.expected_num_vectors=c->n,.expected_dim=c->dim,
            .m=c->pq_m,.ks=1<<c->pq_bits,.train_max=c->train_samples,.kmeans_iters=c->train_iters,
            .encode_chunk_vectors=8192,.residual_bits=c->residual_bits,
            .memory_budget_bytes=256u*1024*1024,.layout_id="original"};
        if(pure_pq_index_prepare(&req,&p.index) ||
           pure_pq_index_map_dsp(p.index,handle,domain_id))goto done;
        mapped=1;
        if(pure_pq_index_open_residual(p.index,c->pq_prefix,c->residual_bits,c->cache_pages,&p.residual) ||
           retrieval_pq_residual_query_create(p.residual,&p.query))goto done;
        paths[0]=c->pq_codes;paths[1]=c->pq_residual;paths[2]=c->pq_qparams;
        fc.decode=pure_pq_refine_decode;fc.provider=p.query;
    } else {
        if(rover_load_residual_map())goto done;
        if(retrieval_demo_load_l1_and_qparams_from_disk(c->l1,c->qparams,c->n,c->dim,c->residual_bits,&g_dataset))goto done;
        g_residual_vpb=4096/g_dataset.residual_code_bytes;
        g_residual_total_blocks=(c->n+g_residual_vpb-1)/g_residual_vpb;
        if(c->disk) {g_residual_fd=open(c->residual,O_RDONLY);if(g_residual_fd<0)goto done;}
        else if(retrieval_demo_load_residual_codes_from_block_file(c->residual,&g_dataset))goto done;
        memset(&g_residual_stats,0,sizeof(g_residual_stats));
        if(pthread_mutex_init(&g_residual_stats.mu,NULL))goto done;stats_ready=1;
        p.sq_cache=residual_cache_create(g_residual_fd,g_residual_vpb,g_residual_total_blocks,
            c->cache_pages,g_dataset.residual_codes,g_dataset.residual_code_bytes,&g_residual_stats);
        if(!p.sq_cache)goto done;
        for(int w=0;w<c->build_threads;w++) {
            p.refine_workers[w].sq=residual_cache_create(g_residual_fd,g_residual_vpb,g_residual_total_blocks,
                c->cache_pages,g_dataset.residual_codes,g_dataset.residual_code_bytes,&g_residual_stats);
            if(!p.refine_workers[w].sq)goto done;
        }
        if(c->split_distance && sqmd_load_correction_bias(c->bias,&g_dataset,&g_sqmd_correction_bias))goto done;
        paths[0]=c->l1;paths[1]=c->residual;paths[2]=c->qparams;
        fc.decode=sqmd_final_decode;fc.provider=&p.refine_workers[0].sq;
    }

    fc.mode=c->third_residual?RETRIEVAL_REFINE_ERROR4:RETRIEVAL_REFINE_CURRENT;
    fc.graph_to_storage=g_reorder_map;
    fc.num_vectors=c->n;fc.dim=c->dim;fc.capacity=c->build_pool;
    fc.cache_slots=c->third_residual?c->refine_cache_pages:1;
    fc.artifact_path=c->third_residual?c->error4:NULL;
    fc.artifact_sha256=c->third_residual?c->error4_hash:NULL;
    for(int i=0;i<3;i++)fc.base_paths[i]=paths[i];
    if(p.pq?retrieval_final_refine_open_pq(&fc,c->pq_book,&p.refine):retrieval_final_refine_open(&fc,&p.refine))goto done;
    p.refine_workers[0].refine=p.refine;
    if(!p.pq)for(int w=1;w<c->build_threads;w++)
        if(retrieval_final_refine_clone(p.refine,fc.decode,&p.refine_workers[w].sq,
            fc.cache_slots,&p.refine_workers[w].refine))goto done;
    if(p.pq) {
        if(pure_pq_shared_init(&p.shared,domain_id,pure_pq_index_metadata(p.index)) ||
           pure_pq_session_open(handle,&p.shared,c->lanes,&p.thread,&p.pq_session))goto done;
    } else {
        int fds[MAX_L1_CHUNKS];g_dma_domain_id=domain_id;
        if(demo_promote_l1_codes_to_shared(&g_dataset) || demo_map_l1_chunks(domain_id,fds)<0)goto done;
        if(calculator_retrieval_load_dma(handle,RETRIEVAL_DMA_NO_GRAPH_FD,fds,g_dataset.l1_chunk_count,
            g_dataset.l1_chunk_vectors,c->n,c->dim,c->degree,0,g_dataset.l1_qparams.scale,g_dataset.l1_qparams.zero_point))goto done;
        mapped=1;
        if(demo_sqmd_buffer_init(&g_sqmd_shared,domain_id,c->lanes) ||
           sqmd_session_open(handle,&g_sqmd_shared,&p.thread,&p.sq_session))goto done;
    }
    p.session_open=1;
    ready=b15_now();b15_phase("BUILD_BEGIN");
    printf("ROVER_BUILD method=%s codec=%s bits=%d residual=%d third=%d degree=%d pool=%d candidates=%d threads=%d\n",
        c->build_method,c->method,p.pq?c->pq_bits:c->sq_bits,c->residual_bits,c->third_residual,c->degree,
        c->build_pool,c->build_candidates,c->build_threads);fflush(stdout);
    if(streaming) {
        RoverInsertBuildConfig cfg={c->degree,1,c->build_candidates,c->entry_points,c->expand_count,c->build_pool,c->seed,1,1};
        RoverInsertBuildRuntime rt={.threads=1,.backlink_threads=c->backlink_threads,.search=rover_insert_search,
            .search_ctx=&stream,.prepare_iteration=rover_insert_begin,.iteration_ctx=&stream,
            .read_vector=retrieval_insert_stream_read};
        RoverInsertBuildStats stats={0};
        RetrievalInsertCheckpoint checkpoint={.path=c->checkpoint,.identity=c->identity,.interval=c->checkpoint_every};
        if(retrieval_insert_stream_open(c->base,c->n,c->dim,p.refine,&stream.vectors))goto done;
        rt.vector_ctx=stream.vectors;
        if(c->parent_n) {
            int32_t *neighbors=NULL;
            if(b15_graph_read(c->parent_graph,c->parent_n,c->dim,c->degree,&neighbors))goto done;
            if(retrieval_rnn_graph_alloc_without_flags(&parent,c->parent_n,c->dim,c->degree)){free(neighbors);goto done;}
            memcpy(parent.neighbors,neighbors,(size_t)c->parent_n*c->degree*sizeof(int));free(neighbors);
            for(int i=0;i<c->parent_n;i++)
                while(parent.degrees[i]<c->degree && parent.neighbors[(size_t)i*c->degree+parent.degrees[i]]>=0)
                    parent.degrees[i]++;

            for(int i=0;i<c->parent_n;i++) {
                float q[c->dim],v[c->dim];
                if(retrieval_final_refine_read_vector(p.refine,i,q))goto done;
                for(int j=0;j<parent.degrees[i];j++) {
                    int target=parent.neighbors[(size_t)i*c->degree+j];
                    if(retrieval_final_refine_read_vector(p.refine,target,v))goto done;
                    double distance=0;for(int d=0;d<c->dim;d++){double e=(double)q[d]-v[d];distance+=e*e;}
                    parent.neighbor_dists[(size_t)i*c->degree+j]=(float)distance;
                }
            }
            if(rover_insert_build_insert_batch_consume_parent(NULL,c->parent_n,c->n-c->parent_n,c->dim,
                &cfg,&rt,&parent,&graph,&stats))goto done;
        } else if(rover_insert_build_insert_checkpointed(NULL,c->n,c->dim,&cfg,&rt,&checkpoint,&graph,&stats))goto done;
        printf("ROVER_INSERT inserted=%llu resumed=%llu checkpoint_saves=%llu decode_hits=%llu "
            "compute_us=%llu checkpoint_load_us=%llu checkpoint_save_us=%llu search_us=%llu prune_us=%llu backlink_us=%llu\n",
            (unsigned long long)stats.inserted_vectors,(unsigned long long)checkpoint.resumed_vectors,
            (unsigned long long)checkpoint.saves,(unsigned long long)stats.backlink_old_hits,
            (unsigned long long)stats.insert_us,(unsigned long long)checkpoint.load_us,
            (unsigned long long)checkpoint.save_us,(unsigned long long)stats.search_us,
            (unsigned long long)stats.prune_us,(unsigned long long)stats.backlink_us);
    } else {
        RetrievalSearchBuildConfig cfg={c->degree,c->build_iterations,c->build_candidates,c->entry_points,c->expand_count,c->build_pool,c->seed,1,1};
        RetrievalSearchBuildRuntime rt={.threads=c->build_threads,.progress_every_iter=1,.search=b15_provider,.search_ctx=&p,
            .distance_batch=retrieval_raw_distances,.distance_ctx=p.raw,.score_candidates=b15_score_candidates,.score_ctx=&p,
            .read_vector=b15_read_vector,.read_vector_ctx=p.raw,.row_vector_buffer=1,.reverse_candidate_count=c->reverse_candidates,
            .prepare_iteration=b15_prepare_iteration,.iteration_ctx=&p,.publish_row=retrieval_live_graph_publish,.publish_ctx=&p.live_graph};
        RetrievalSearchBuildStats stats={0};
        if(c->cooldown_every) {
            char journal[4096];
            if(snprintf(journal,sizeof(journal),"%s/cooldown.jsonl",c->result_dir)>=(int)sizeof(journal))goto done;
            cool=fopen(journal,"wx");if(!cool)goto done;
            cooling=(Build15Cooling){.file=cool,.n=c->n,.requested_us=(uint64_t)c->cooldown_ms*1000,.provider=&p};
            rt.cooldown_every_queries=c->cooldown_every;rt.cooldown_us=cooling.requested_us;
            rt.cooldown_event=b15_cooling;rt.cooldown_ctx=&cooling;
            rt.cooldown_quiesce=b15_quiesce;rt.cooldown_quiesce_ctx=&p;
        }
        if(retrieval_search_build_graph_ex(NULL,c->n,c->dim,&cfg,&rt,&graph,&stats))goto done;
        cooldown=stats.cooldown_us;
        printf("ROVER_SEARCH_BUILD searched_queries=%llu scores_reused=%llu scores_computed=%llu "
            "row_vector_loads=%llu reverse_rows=%llu reverse_edges=%llu reverse_us=%llu reverse_index_peak_bytes=%llu cooldown_events=%llu\n",
            (unsigned long long)stats.searched_queries,(unsigned long long)stats.scores_reused,
            (unsigned long long)stats.scores_computed,(unsigned long long)stats.row_vector_loads,
            (unsigned long long)stats.reverse_rows,(unsigned long long)stats.reverse_edges_seen,
            (unsigned long long)stats.reverse_us,(unsigned long long)stats.reverse_index_peak_bytes,
            (unsigned long long)stats.cooldown_events);
    }
    built=b15_now();
    if(rover_save_graph(c->graph,&graph))goto done;
    b15_phase("SAVED");rover_build_report(begin,ready,built,b15_now(),cooldown);
    printf("ROVER_BUILD_COMPLETE graph=%s\n",c->graph);rc=0;
done:
    if(cool && fclose(cool))rc=-1;
    if(p.session_open) {
        int err=p.pq?pure_pq_session_close(&p.shared,p.thread,&p.pq_session):sqmd_session_close(&g_sqmd_shared,p.thread,&p.sq_session);
        if(err)rc=-1;
    }
    retrieval_insert_stream_close(stream.vectors);pure_pq_cpu_search_destroy(stream.workspace);
    for(int w=1;w<4;w++)retrieval_final_refine_close(p.refine_workers[w].refine);
    retrieval_final_refine_close(p.refine);
    for(int w=0;w<4;w++)residual_cache_destroy(p.refine_workers[w].sq);
    if(p.pq) {
        pure_pq_shared_destroy(&p.shared);
        if(mapped && pure_pq_index_unmap_dsp(p.index))rc=-1;
        retrieval_pq_residual_query_destroy(p.query);retrieval_pq_residual_index_close(p.residual);pure_pq_index_release(p.index);
    } else {
        demo_sqmd_buffer_destroy(&g_sqmd_shared,domain_id);
        if(mapped)calculator_retrieval_unload_dma(handle);
        demo_dma_unmap();residual_cache_destroy(p.sq_cache);
        if(g_residual_fd>=0){close(g_residual_fd);g_residual_fd=-1;}
        free(g_sqmd_correction_bias);g_sqmd_correction_bias=NULL;
        retrieval_demo_free_encoded_dataset(&g_dataset);
        if(stats_ready)pthread_mutex_destroy(&g_residual_stats.mu);
    }
    free(g_reorder_map);g_reorder_map=NULL;
    retrieval_live_graph_destroy(&p.live_graph);retrieval_rnn_graph_free(&parent);retrieval_rnn_graph_free(&graph);
    retrieval_raw_close(p.raw);pthread_mutex_destroy(&p.mu);
    if(rc)fprintf(stderr,"Rover construction failed (errno=%d)\n",errno);return rc;
}
