#ifndef RETRIEVAL_BUILD_LIVE_GRAPH_H
#define RETRIEVAL_BUILD_LIVE_GRAPH_H
#include <pthread.h>
#include <string.h>
#include <math.h>
#include "dsp_demo/retrieval_rnn_descent_core.h"
#include "dsp_demo/retrieval_graph_reader.h"


enum { RETRIEVAL_LIVE_GRAPH_LOCKS = 64 };
typedef struct RetrievalBuildLiveGraph {
    const RetrievalRnnGraph *graph;
    pthread_mutex_t locks[RETRIEVAL_LIVE_GRAPH_LOCKS];
    int initialized;
    uint64_t published_rows, row_reads, reads_after_publish;
} RetrievalBuildLiveGraph;

static void retrieval_live_graph_destroy(RetrievalBuildLiveGraph *view) {
    while(view->initialized) pthread_mutex_destroy(&view->locks[--view->initialized]);
    view->graph=NULL;
}
static int retrieval_live_graph_init(RetrievalBuildLiveGraph *view) {
    memset(view,0,sizeof(*view));
    for(int i=0;i<RETRIEVAL_LIVE_GRAPH_LOCKS;i++) {
        if(pthread_mutex_init(&view->locks[i],NULL)) {
            retrieval_live_graph_destroy(view);return -1;
        }
        view->initialized++;
    }
    return 0;
}

static int retrieval_live_graph_bind(RetrievalBuildLiveGraph *view,const RetrievalRnnGraph *graph) {
    if(view->initialized!=RETRIEVAL_LIVE_GRAPH_LOCKS || !graph ||
       !graph->neighbors || !graph->neighbor_dists || !graph->degrees) return -1;
    view->graph=graph;return 0;
}
static int retrieval_live_graph_read(void *ctx,int source,int32_t *neighbors,int count) {
    RetrievalBuildLiveGraph *view=ctx;const RetrievalRnnGraph *g=view->graph;
    if(!g || !neighbors || source<0 || source>=g->num_vectors || count<0 || count>g->max_degree)return -1;
    pthread_mutex_t *lock=&view->locks[source%RETRIEVAL_LIVE_GRAPH_LOCKS];
    pthread_mutex_lock(lock);
    memcpy(neighbors,g->neighbors+(size_t)source*g->max_degree,(size_t)count*sizeof(*neighbors));
    pthread_mutex_unlock(lock);
    __atomic_fetch_add(&view->row_reads,1,__ATOMIC_RELAXED);
    if(__atomic_load_n(&view->published_rows,__ATOMIC_RELAXED))
        __atomic_fetch_add(&view->reads_after_publish,1,__ATOMIC_RELAXED);
    return 0;
}

static int retrieval_live_graph_publish(void *ctx,const RetrievalRnnGraph *graph,
                                       const RetrievalRnnGraph *completed,int row) {
    RetrievalBuildLiveGraph *view=ctx;
    if(!graph || view->graph!=graph || !completed || completed->num_vectors!=graph->num_vectors ||
       completed->max_degree!=graph->max_degree || row<0 || row>=graph->num_vectors ||
       completed->degrees[row]!=graph->max_degree)return -1;
    int r=graph->max_degree;size_t first=(size_t)row*r;
    for(int i=0;i<r;i++) {
        int id=completed->neighbors[first+i];float d=completed->neighbor_dists[first+i];
        if(id<0 || id>=graph->num_vectors || id==row || !isfinite(d) || d<0)return -1;
        for(int j=0;j<i;j++)if(completed->neighbors[first+j]==id)return -1;
    }
    pthread_mutex_t *lock=&view->locks[row%RETRIEVAL_LIVE_GRAPH_LOCKS];
    pthread_mutex_lock(lock);
    memcpy(graph->neighbors+first,completed->neighbors+first,(size_t)r*sizeof(int32_t));
    memcpy(graph->neighbor_dists+first,completed->neighbor_dists+first,(size_t)r*sizeof(float));
    graph->degrees[row]=r;
    pthread_mutex_unlock(lock);
    __atomic_fetch_add(&view->published_rows,1,__ATOMIC_RELAXED);
    return 0;
}
#endif
