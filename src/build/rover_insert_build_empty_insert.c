static int sb_empty_bootstrap_candidates(void *ctx, const float *base, int n,
    int dim, const RoverInsertBuildConfig *cfg, const RetrievalRnnGraph *graph,
    int query_id, int cap, RoverInsertBuildCandidate *items, int *count) {
    (void)ctx; (void)base; (void)dim; (void)cfg; (void)graph;
    if (query_id != n-1 || n-1 > cap) return -1;
    for (int i=0; i<n-1; ++i) {
        items[i].id=i; items[i].search_dist=0; items[i].exact_dist=FLT_MAX;
    }
    *count=n-1;
    return 0;
}

int rover_insert_build_insert_checkpointed(const float *base, int n, int dim,
    const RoverInsertBuildConfig *cfg, const RoverInsertBuildRuntime *runtime,
    RetrievalInsertCheckpoint *checkpoint, RetrievalRnnGraph *out, RoverInsertBuildStats *stats) {
    RetrievalRnnGraph graph={0};
    SbBacklinkPool *backlinks=NULL;
    RoverInsertBuildCandidate *items=NULL, *pair=NULL;
    RoverInsertBuildStats local_stats;
    int *forward=NULL, *old=NULL, rc=-1;
    uint64_t begin=sb_now_us(),prior_us=0,save_start=checkpoint?checkpoint->save_us:0;
    if (!out || n<0 || dim<=0 || !cfg || !runtime || (n && !base && !runtime->read_vector) ||
        cfg->r<=0 || cfg->candidate_count<cfg->r || cfg->pool_size<cfg->candidate_count ||
        cfg->entry_points<=0 || cfg->expand_count<=0 || cfg->prune_mode!=1 ||
        cfg->batch_align!=1 || runtime->threads!=1 || !runtime->search) return -1;
    memset(out,0,sizeof(*out));
    if (!stats) stats=&local_stats;
    rover_insert_build_stats_reset(stats);
    stats->threads_used=1;
    if (!n) {out->dim=dim; out->max_degree=cfg->r; return 0;}
    int restored=checkpoint ? retrieval_insert_checkpoint_load(checkpoint,n,dim,cfg,&graph,stats) : 1;
    if(restored<0)goto done;
    if(restored==1 && retrieval_rnn_graph_alloc_without_flags(&graph,n,dim,cfg->r))goto done;
    if(!restored) {
        prior_us=stats->insert_us;
        printf("INSERT_CHECKPOINT_RESUME inserted=%d total=%d compute_s=%.6f load_s=%.6f\n",
            graph.num_vectors,n,prior_us/1e6,checkpoint->load_us/1e6);fflush(stdout);
    }
    items=malloc((size_t)cfg->candidate_count*sizeof(*items));
    forward=malloc((size_t)cfg->r*sizeof(*forward));
    old=malloc((size_t)cfg->r*sizeof(*old));
    if (!items || !forward || !old) goto done;
    if(sb_backlink_pool_open(cfg,runtime,dim,&backlinks))goto done;
    if(backlinks)stats->threads_used=backlinks->threads;
    stats->init_us+=sb_now_us()-begin;
    begin=sb_now_us();

    if(restored==1) {graph.num_vectors=1;stats->inserted_vectors=1;}
    for (int logical=graph.num_vectors+1;logical<=n;++logical) {
        RoverInsertBuildConfig small=*cfg;
        RoverInsertBuildRuntime active=*runtime;
        if (logical<=cfg->r+1) {
            small.r=logical-1;
            active.search=sb_empty_bootstrap_candidates;
            active.search_pair=NULL;
        }
        if (sb_insert_one_inplace(base,logical,logical-1,dim,&small,&active,backlinks,&graph,
                                 items,pair,forward,old,stats)) goto done;
        if(checkpoint && checkpoint->interval && ((uint64_t)logical%checkpoint->interval==0 || logical==n)) {
            stats->insert_us=prior_us+sb_now_us()-begin-(checkpoint->save_us-save_start);
            printf("INSERT_CHECKPOINT_BEGIN inserted=%d total=%d\n",logical,n);fflush(stdout);
            uint64_t save_before=checkpoint->save_us;
            if(retrieval_insert_checkpoint_save(checkpoint,n,cfg,&graph,stats))goto done;
            printf("INSERT_CHECKPOINT_SAVED inserted=%d total=%d compute_s=%.6f save_s=%.6f bytes=%llu\n",
                logical,n,stats->insert_us/1e6,(checkpoint->save_us-save_before)/1e6,
                (unsigned long long)(sizeof(InsertCheckpointHeader)+sizeof(*stats)+(uint64_t)logical*(4+8*(uint64_t)cfg->r)+8));fflush(stdout);
        }
        if (logical%10000==0 || logical==n) {
            printf("INSERT_EMPTY_PROGRESS inserted=%d total=%d wall_s=%.6f search_s=%.6f prune_s=%.6f backlink_s=%.6f backlinks=%llu\n",
                logical,n,(prior_us+sb_now_us()-begin-(checkpoint?checkpoint->save_us-save_start:0))/1e6,stats->search_us/1e6,
                stats->prune_us/1e6,stats->backlink_us/1e6,
                (unsigned long long)stats->backlink_rows_considered);
            fflush(stdout);
        }
    }
    sb_fill_graph_stats(&graph,stats);
    stats->iterations_done=1;
    *out=graph; memset(&graph,0,sizeof(graph)); rc=0;
done:
    sb_backlink_pool_close(backlinks);
    stats->insert_us=prior_us+sb_now_us()-begin-(checkpoint?checkpoint->save_us-save_start:0);
    free(items); free(forward); free(old); retrieval_rnn_graph_free(&graph);
    return rc;
}


int rover_insert_build_insert_empty(const float *base,int n,int dim,
    const RoverInsertBuildConfig *cfg,const RoverInsertBuildRuntime *runtime,
    RetrievalRnnGraph *out,RoverInsertBuildStats *stats) {
    return rover_insert_build_insert_checkpointed(base,n,dim,cfg,runtime,NULL,out,stats);
}
