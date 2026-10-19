typedef struct {
    int cap, dim;
    RetrievalSearchBuildCandidate *items, *best;
    int32_t *ids;
    float *scores, *vectors;
    uint8_t *valid, *loaded;
    int *selected;
} SbRow;
static void sb_row_free(SbRow *s) {
    free(s->items);free(s->best);free(s->ids);free(s->scores);
    free(s->vectors);free(s->valid);free(s->loaded);free(s->selected);
    memset(s,0,sizeof(*s));
}
static int sb_row_init(SbRow *s,int cap,int dim,int buffered) {
    memset(s,0,sizeof(*s));s->cap=cap;s->dim=dim;
    if ((size_t)cap>SIZE_MAX/sizeof(*s->items) || (size_t)cap>SIZE_MAX/sizeof(float)/(size_t)dim) return -1;
    s->items=malloc((size_t)cap*sizeof(*s->items));s->best=malloc((size_t)cap*sizeof(*s->best));
    s->ids=malloc((size_t)cap*sizeof(*s->ids));s->scores=malloc((size_t)cap*sizeof(*s->scores));
    s->valid=malloc(cap);s->loaded=malloc(cap);s->selected=malloc((size_t)cap*sizeof(int));
    if(buffered)s->vectors=malloc((size_t)cap*dim*sizeof(float));
    if(!s->items||!s->best||!s->ids||!s->scores||!s->valid||!s->loaded||!s->selected||(buffered&&!s->vectors)) {
        sb_row_free(s);return -1;
    }return 0;
}
static int sb_score_valid(float d) { return isfinite(d)&&d>=0; }
static int sb_score_ids(const RetrievalSearchBuildRuntime *rt,int u,const int32_t *ids,int n,float *scores,RetrievalSearchBuildStats *st) {
    if(!n)return 0;
    uint64_t t=sb_now_us();int rc=rt->score_candidates(rt->score_ctx,u,ids,n,scores);
    st->score_us+=sb_now_us()-t;
    if(rc)return -1;
    for(int i=0;i<n;i++)if(!sb_score_valid(scores[i]))return -1;
    st->scores_computed+=n;return 0;
}

static int sb_row_add(SbRow *s,int *n,int id,float score,int valid,RetrievalSearchBuildStats *st) {
    if(valid&&!sb_score_valid(score))return -1;
    for(int i=0;i<*n;i++)if(s->items[i].id==id) {
        ++st->duplicate_rejects;
        if(valid&&!s->valid[i]){s->items[i].exact_dist=score;s->valid[i]=1;++st->scores_reused;}
        return 0;
    }
    if(*n>=s->cap)return -1;
    s->items[*n].id=id;s->items[*n].search_dist=score;s->items[*n].exact_dist=score;
    s->valid[*n]=valid;if(valid)++st->scores_reused;++*n;return 0;
}
static int sb_merge_scored(SbRow *s,const RetrievalSearchBuildRuntime *rt,const RetrievalRnnGraph *old,
    int u,int nvec,const RetrievalSearchBuildCandidate *provider,int count,int old_valid,RetrievalSearchBuildStats *st) {
    int n=0,missing=0;
    for(int i=0;i<count;i++) {
        int id=provider[i].id;
        if(id==u){++st->self_rejects;continue;}
        if(id<0||id>=nvec)continue;
        if(sb_row_add(s,&n,id,provider[i].search_dist,1,st))return -1;
    }
    if(old)for(int i=0;i<old->degrees[u];i++) {
        size_t k=sb_slot(old,u,i);int id=old->neighbors[k];
        if(id<0||id>=nvec||id==u)return -1;
        if(sb_row_add(s,&n,id,old->neighbor_dists[k],old_valid,st))return -1;
    }
    for(int i=0;i<n;i++)if(!s->valid[i])s->ids[missing++]=s->items[i].id;
    if(sb_score_ids(rt,u,s->ids,missing,s->scores,st))return -1;
    missing=0;for(int i=0;i<n;i++)if(!s->valid[i])s->items[i].exact_dist=s->scores[missing++];
    qsort(s->items,n,sizeof(*s->items),sb_candidate_compare);
    memset(s->loaded,0,n);st->candidates_seen+=n;return n;
}
static const float *sb_row_vector(SbRow *s,int k,const float *base,const RetrievalSearchBuildRuntime *rt,RetrievalSearchBuildStats *st) {
    float *v=s->vectors+(size_t)k*s->dim;
    if(!s->loaded[k]) {
        uint64_t t=sb_now_us();int rc=0;
        if(rt->read_vector)rc=rt->read_vector(rt->read_vector_ctx,s->items[k].id,v);
        else memcpy(v,base+(size_t)s->items[k].id*s->dim,(size_t)s->dim*sizeof(float));
        st->row_read_us+=sb_now_us()-t;
        if(rc)return NULL;
        for(int d=0;d<s->dim;d++)if(!isfinite(v[d]))return NULL;
        s->loaded[k]=1;++st->row_vector_loads;
    }return v;
}
static float sb_row_pair(SbRow *s,int a,int b,const float *base,const RetrievalSearchBuildRuntime *rt,RetrievalSearchBuildStats *st) {
    float sum=0;uint64_t t;
    if(s->vectors) {
        const float *va=sb_row_vector(s,a,base,rt,st),*vb=sb_row_vector(s,b,base,rt,st);
        if(!va||!vb)return NAN;
        t=sb_now_us();
        for(int d=0;d<s->dim;d++){float delta=va[d]-vb[d];sum+=delta*delta;}
    }else{t=sb_now_us();sum=sb_distance(rt,base,s->dim,s->items[a].id,s->items[b].id);}
    st->pair_distance_us+=sb_now_us()-t;++st->exact_distance_pairs;return sum;
}
static int sb_prune_scored(SbRow *s,int n,int u,const float *base,const RetrievalSearchBuildConfig *cfg,
    const RetrievalSearchBuildRuntime *rt,RetrievalRnnGraph *out,RetrievalSearchBuildStats *st) {

    if(n<cfg->r)return -1;
    int selected=0;
    for(int i=0;i<n&&selected<cfg->r;i++) {
        int occluded=0;
        if(cfg->prune_mode)for(int j=0;j<selected;j++) {
            float d=sb_row_pair(s,s->selected[j],i,base,rt,st);
            if(!sb_score_valid(d))return -1;
            if(d<s->items[i].exact_dist){occluded=1;break;}
        }
        if(!occluded)s->selected[selected++]=i;
    }
    if(selected<cfg->r) {
        ++st->rows_filled;
        for(int i=0;i<n&&selected<cfg->r;i++) {
            int found=0;for(int j=0;j<selected;j++)if(s->selected[j]==i){found=1;break;}
            if(!found){s->selected[selected++]=i;++st->fill_edges;}
        }
    }
    out->degrees[u]=0;
    for(int j=0;j<selected;j++) {
        int i=s->selected[j];
        if(sb_set_row_edge(out,u,j,s->items[i].id,s->items[i].exact_dist))return -1;
    }
    sb_sort_row(out,u);st->candidates_kept+=selected;return selected==cfg->r?0:-1;
}
static int sb_revised_rows(SearchBuildWorker *w) {
    SbRow s;int ret=-1,cap=w->cfg->candidate_count;
    RetrievalSearchBuildCandidate *c=malloc((size_t)cap*sizeof(*c));
    if(!c)return -1;
    if(sb_row_init(&s,cap+w->cfg->r,w->dim,w->runtime->row_vector_buffer)){free(c);return -1;}
    for(int u=w->start;u<w->end;u++) {
        if(w->runtime->diagnostic_rows_per_worker && u-w->start>=w->runtime->diagnostic_rows_per_worker){ret=1;goto done;}
        int count=0;
        if(__atomic_load_n(w->failed,__ATOMIC_ACQUIRE)||sb_gate_enter(w->gate))goto done;
        uint64_t t=sb_now_us();
        if(w->runtime->search(w->runtime->search_ctx,w->base,w->num_vectors,w->dim,w->cfg,w->old_graph,u,cap,c,&count))goto done;
        w->stats.search_us+=sb_now_us()-t;++w->stats.searched_queries;
        if(count<0||count>cap)goto done;
        t=sb_now_us();int n=sb_merge_scored(&s,w->runtime,w->old_graph,u,w->num_vectors,c,count,w->old_scores_valid,&w->stats);
        if(n<0||sb_prune_scored(&s,n,u,w->base,w->cfg,w->runtime,w->new_graph,&w->stats))goto done;
        if(w->runtime->publish_row && w->runtime->publish_row(w->runtime->publish_ctx,
            w->old_graph,w->new_graph,u))goto done;
        w->stats.prune_us+=sb_now_us()-t;
        if(sb_gate_finish(w->gate))goto done;
        if(w->runtime->progress_every_iter&&(u-w->start+1)%10000==0) {
            printf("  search-build revised rows: [%d,%d) %d/%d score_reused=%llu score_computed=%llu raw_loads=%llu\n",w->start,w->end,u-w->start+1,w->end-w->start,
                (unsigned long long)w->stats.scores_reused,(unsigned long long)w->stats.scores_computed,(unsigned long long)w->stats.row_vector_loads);fflush(stdout);
        }
    }ret=0;
done:sb_row_free(&s);free(c);return ret;
}
typedef struct {uint32_t *offsets;int32_t *ids;} SbReverse;
typedef struct {SearchBuildWorker w;const SbReverse *reverse;} SbReverseWorker;

static void sb_heap_down(RetrievalSearchBuildCandidate *a,int n,int i) {
    for(;;){int c=2*i+1;if(c>=n)break;
        if(c+1<n&&sb_candidate_compare(a+c,a+c+1)<0)++c;
        if(sb_candidate_compare(a+i,a+c)>=0)break;
        RetrievalSearchBuildCandidate tmp=a[i];a[i]=a[c];a[c]=tmp;i=c;}
}
static void sb_heap_offer(RetrievalSearchBuildCandidate *a,int *n,int cap,int id,float d) {
    RetrievalSearchBuildCandidate v={id,d,d};
    if(*n<cap){int i=(*n)++;a[i]=v;while(i>0){int p=(i-1)/2;if(sb_candidate_compare(a+p,a+i)>=0)break;
        RetrievalSearchBuildCandidate tmp=a[p];a[p]=a[i];a[i]=tmp;i=p;}}
    else if(sb_candidate_compare(&v,a)<0){a[0]=v;sb_heap_down(a,*n,0);}
}
static void *sb_reverse_worker(void *arg) {
    SbReverseWorker *rw=arg;SearchBuildWorker *w=&rw->w;SbRow s;int cap=w->runtime->reverse_candidate_count;
    w->ret=-1;
    if(sb_row_init(&s,w->cfg->candidate_count+w->cfg->r,w->dim,w->runtime->row_vector_buffer))goto failure;
    for(int u=w->start;u<w->end;u++) {
        if(__atomic_load_n(w->failed,__ATOMIC_ACQUIRE))goto done;
        uint32_t begin=rw->reverse->offsets[u],end=rw->reverse->offsets[u+1];int kept=0;
        for(uint32_t pos=begin;pos<end;) {
            int batch=(end-pos>(uint32_t)cap)?cap:(int)(end-pos);
            if(sb_score_ids(w->runtime,u,rw->reverse->ids+pos,batch,s.scores,&w->stats))goto done;
            for(int j=0;j<batch;j++)sb_heap_offer(s.best,&kept,cap,rw->reverse->ids[pos+j],s.scores[j]);
            pos+=batch;
        }
        w->stats.reverse_edges_seen+=end-begin;
        w->stats.reverse_candidates_dropped+=(uint64_t)(end-begin)-kept;
        int n=sb_merge_scored(&s,w->runtime,w->new_graph,u,w->num_vectors,s.best,kept,1,&w->stats);
        if(n<0||sb_prune_scored(&s,n,u,w->base,w->cfg,w->runtime,w->new_graph,&w->stats))goto done;
        ++w->stats.reverse_rows;
        if(w->runtime->progress_every_iter&&(u-w->start+1)%10000==0) {
            printf("  search-build reverse rows: [%d,%d) %d/%d\n",w->start,w->end,u-w->start+1,w->end-w->start);fflush(stdout);
        }
    }w->ret=0;
done:sb_row_free(&s);
failure:if(w->ret)__atomic_store_n(w->failed,1,__ATOMIC_RELEASE);return NULL;
}
static int sb_reverse_phase(const float *base,int n,int dim,const RetrievalSearchBuildConfig *cfg,
    const RetrievalSearchBuildRuntime *rt,RetrievalRnnGraph *g,int threads,RetrievalSearchBuildStats *st) {
    SbReverse rev={0};uint32_t *cursor=NULL;SbReverseWorker *w=NULL;pthread_t *tids=NULL;
    uint64_t t=sb_now_us(),total=0;int rc=-1,launched=0,failed=0;
    if((size_t)n+1>SIZE_MAX/sizeof(uint32_t))return -1;
    rev.offsets=calloc((size_t)n+1,sizeof(uint32_t));if(!rev.offsets)goto done;
    for(int u=0;u<n;u++)for(int j=0;j<g->degrees[u];j++) {
        int v=g->neighbors[sb_slot(g,u,j)];
        if(v<0||v>=n||v==u||++total>UINT32_MAX)goto done;
        ++rev.offsets[v+1];
    }
    for(int i=0;i<n;i++) {
        if(st&&rev.offsets[i+1]>st->reverse_max_indegree)st->reverse_max_indegree=rev.offsets[i+1];
        rev.offsets[i+1]+=rev.offsets[i];
    }
    if(total>SIZE_MAX/sizeof(int32_t))goto done;
    rev.ids=malloc((size_t)total*sizeof(int32_t));cursor=malloc((size_t)n*sizeof(uint32_t));
    if(!rev.ids||!cursor)goto done;
    memcpy(cursor,rev.offsets,(size_t)n*sizeof(uint32_t));
    for(int u=0;u<n;u++)for(int j=0;j<g->degrees[u];j++)rev.ids[cursor[g->neighbors[sb_slot(g,u,j)]]++]=u;
    if(st){uint64_t bytes=4*(total+2*(uint64_t)n+1);if(bytes>st->reverse_index_peak_bytes)st->reverse_index_peak_bytes=bytes;st->reverse_index_us+=sb_now_us()-t;}
    free(cursor);cursor=NULL;
    w=calloc(threads,sizeof(*w));tids=calloc(threads,sizeof(*tids));if(!w||!tids)goto done;
    for(int i=0;i<threads;i++) {
        w[i].w.base=base;w[i].w.num_vectors=n;w[i].w.dim=dim;w[i].w.cfg=cfg;w[i].w.runtime=rt;w[i].w.new_graph=g;
        w[i].w.start=(int)((int64_t)n*i/threads);w[i].w.end=(int)((int64_t)n*(i+1)/threads);
        w[i].w.failed=&failed;w[i].reverse=&rev;
        if(pthread_create(&tids[i],NULL,sb_reverse_worker,&w[i])){__atomic_store_n(&failed,1,__ATOMIC_RELEASE);break;}
        ++launched;
    }
    for(int i=0;i<launched;i++){pthread_join(tids[i],NULL);sb_stats_accumulate(st,&w[i].w.stats);}
    if(launched==threads&&!__atomic_load_n(&failed,__ATOMIC_ACQUIRE))rc=0;
done:
    free(tids);free(w);free(cursor);free(rev.ids);free(rev.offsets);
    if(st)st->reverse_us+=sb_now_us()-t;return rc;
}
