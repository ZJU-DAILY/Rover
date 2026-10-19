#include "rover_config.h"
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "dsp_demo/retrieval_build_artifacts.h"

RoverConfig rover_config = {
    .graph_entry=-1,.expand_count=32,.rnn_s=16,.rnn_t1=4,.build_distance="cpu",
    .gt_k=100, .sq_bits=4, .residual_bits=4, .pq_bits=8, .disk=1,
    .top_k=10, .pool=128, .entry_points=32, .lanes=4, .source_window=1,
    .first_factor=3, .cache_pages=2048, .refine_cache_pages=2048,
    .refine_factor=2, .correction=1, .train_samples=100000, .train_iters=20,
    .seed=42, .domain=3, .unsigned_pd=1, .degree=16, .build_pool=128,
    .build_candidates=16, .build_iterations=1, .build_threads=1,
    .backlink_threads=1, .checkpoint_every=1000000, .cooldown_ms=10000,
    .parent_graph="", .method="sq", .search_mode="sqmd", .build_method="stream-insert", .graph_format="nndescent",
    .base="", .query="", .gt="", .graph="", .order_map="", .index_dir="",
    .residual_map="", .l1="", .residual="", .qparams="", .error4="", .bias="", .output="results.jsonl",
    .pq_prefix="", .pq_book="", .pq_codes="", .pq_residual="", .pq_qparams="",
    .result_dir=".", .metrics="metrics.csv", .trace_path="trace.bin", .checkpoint="", .identity="",
    .layout="original", .layout_hash="0000000000000000000000000000000000000000000000000000000000000000", .dataset="dataset"
};

typedef struct { const char *name; size_t offset; int integer; } Option;
#define INT(name, field) {name, offsetof(RoverConfig, field), 1}
#define STR(name, field) {name, offsetof(RoverConfig, field), 0}
static const Option options[] = {
    INT("parent-n",parent_n), STR("parent-graph",parent_graph),
    STR("build-distance",build_distance),INT("expand-count",expand_count),INT("rnn-s",rnn_s),INT("rnn-t1",rnn_t1),
    INT("gt-k",gt_k), INT("n",n), INT("dim",dim), INT("nq",nq), INT("graph-degree",degree),
    INT("sq-bits",sq_bits), INT("residual-bits",residual_bits), INT("pq-bits",pq_bits),
    INT("pq-m",pq_m), INT("pq-compression",pq_compression), INT("top-k",top_k),
    INT("pool",pool), INT("entry-points",entry_points), INT("lanes",lanes), INT("source-window",source_window),
    INT("first-stage-factor",first_factor), INT("cache-pages",cache_pages), INT("refine-cache-pages",refine_cache_pages),
    INT("refine-factor",refine_factor), INT("pq-train-samples",train_samples), INT("pq-train-iters",train_iters),
    INT("seed",seed), INT("domain",domain), INT("unsigned-pd",unsigned_pd), INT("build-pool",build_pool),
    INT("build-candidates",build_candidates), INT("build-iterations",build_iterations), INT("build-threads",build_threads),
    INT("build-refine-candidates",build_refine_candidates), INT("backlink-threads",backlink_threads),
    INT("checkpoint-every",checkpoint_every), INT("reverse-candidates",reverse_candidates),
    INT("cooldown-every",cooldown_every), INT("cooldown-ms",cooldown_ms),
    STR("method",method), STR("search-mode",search_mode), STR("build-method",build_method), STR("graph-format",graph_format),
    STR("index-dir",index_dir), STR("base",base), STR("query",query), STR("gt",gt), STR("graph",graph),
    STR("order-map",order_map), STR("output",output), STR("l1",l1), STR("residual",residual), STR("qparams",qparams),
    STR("residual-map",residual_map), STR("error4",error4), STR("bias",bias), STR("pq-prefix",pq_prefix), STR("pq-book",pq_book), STR("pq-codes",pq_codes),
    STR("pq-residual",pq_residual), STR("pq-qparams",pq_qparams), STR("result-dir",result_dir), STR("metrics",metrics),
    STR("checkpoint",checkpoint), STR("identity",identity), STR("layout",layout), STR("layout-hash",layout_hash), STR("dataset",dataset)
};
static int number(const char *s, int *out) {
    char *end; errno=0; long n=strtol(s,&end,10);
    if(errno || !*s || *end || n<0 || n>INT_MAX)return -1;
    *out=(int)n; return 0;
}
static int choice(const char *s,const char *a,const char *b,int *v) {
    if(!strcmp(s,a)){*v=0;return 0;} if(!strcmp(s,b)){*v=1;return 0;} return -1;
}
int rover_config_parse(int argc,char **argv) {
    RoverConfig *c=&rover_config; int refine=0,subdim=0;
    if(argc<2 || !strcmp(argv[1],"--help")) {
        puts("Rover search|build-index|insert|resume [options]\nUse Rover_walkthrough.py on the host for metadata validation, preparation and orchestration.");
        for(size_t i=0;i<sizeof(options)/sizeof(*options);i++)printf("  --%s VALUE\n",options[i].name);
        puts("  --third-residual off|on  --refine off|on  --refine-all\n  --correction off|online  --storage memory|disk  --split-distance off|on  --pq-subdim N");
        return 1;
    }
    c->command=argv[1];
    if(strcmp(c->command,"search") && strcmp(c->command,"build-index") && strcmp(c->command,"insert") && strcmp(c->command,"resume"))return -1;
    for(int i=2;i<argc;i++) {
        const char *key=argv[i];int found=0;
        if(!strcmp(key,"--refine-all")){c->refine_all=1;continue;}
        if(strncmp(key,"--",2) || ++i>=argc){fprintf(stderr,"Missing option value: %s\n",key);return -1;}
        key+=2; const char *v=argv[i];
        if(!strcmp(key,"graph-entry")){if(!strcmp(v,"-1"))c->graph_entry=-1;else if(number(v,&c->graph_entry))return -1;continue;}
        if(!strcmp(key,"third-residual")){if(choice(v,"off","on",&c->third_residual))return -1;continue;}
        if(!strcmp(key,"refine")){if(choice(v,"off","on",&refine))return -1;continue;}
        if(!strcmp(key,"correction")){if(choice(v,"off","online",&c->correction))return -1;continue;}
        if(!strcmp(key,"storage")){if(choice(v,"memory","disk",&c->disk))return -1;continue;}
        if(!strcmp(key,"split-distance")){if(choice(v,"off","on",&c->split_distance))return -1;continue;}
        if(!strcmp(key,"pq-subdim")){if(number(v,&subdim))return -1;continue;}
        for(size_t j=0;j<sizeof(options)/sizeof(*options);j++)if(!strcmp(key,options[j].name)) {
            char *field=(char*)c+options[j].offset;
            if(options[j].integer){if(number(v,(int*)field))return -1;}
            else *(const char**)field=v;
            found=1;break;
        }
        if(!found){fprintf(stderr,"Unknown option: --%s\n",key);return -1;}
    }
#define REQUIRE(test, message) do { if(!(test)){fprintf(stderr,"Rover: %s\n",message);return -1;} } while(0)
    REQUIRE(c->n>0 && c->dim>0 && c->dim<=1024,"n > 0 and dim in 1..1024 required");
    REQUIRE(c->domain<=3 && c->unsigned_pd<=1,"domain 0..3 and unsigned-pd 0/1 required");
    REQUIRE(c->degree>0 && c->degree<c->n,"graph-degree must be in 1..n-1");
    REQUIRE((c->sq_bits==4 || c->sq_bits==8) && (c->pq_bits==4 || c->pq_bits==8) && (c->residual_bits==4 || c->residual_bits==8),"code and residual bits must be 4 or 8");
    REQUIRE(!strcmp(c->method,"sq") || !strcmp(c->method,"pq"),"method must be sq or pq");
    REQUIRE(c->top_k>0 && c->top_k<=100 && c->pool>=c->top_k && c->pool<=2000,"top-k 1..100; pool top-k..2000");
    REQUIRE(c->expand_count>=1 && c->expand_count<=32 && c->graph_entry<c->n && c->rnn_s>0 && c->rnn_t1>0,"invalid expand-count/graph-entry/RNN parameters");
    REQUIRE(c->entry_points>0 && c->entry_points<=c->n && c->lanes>0 && c->lanes<=8 && c->source_window>=1 && c->source_window<=2,"invalid entry/lane/source-window capacity");
    REQUIRE(c->first_factor>0 && c->first_factor<=2000/c->top_k && c->refine_factor>0 && c->refine_factor<=2000/c->top_k,"candidate factor exceeds capacity");
    REQUIRE(c->cache_pages<=65536 && !(c->cache_pages&(c->cache_pages-1)) && c->refine_cache_pages<=65536 && !(c->refine_cache_pages&(c->refine_cache_pages-1)),"cache pages must be powers of two up to 65536");
    REQUIRE(c->gt_k>0 && c->gt_k<=1000000,"invalid GT width");
    REQUIRE(c->cache_pages>0 && (!c->third_residual || c->refine_cache_pages>0),"positive cache size required");
    REQUIRE(!refine || c->correction,"coarse-only search with final residual refinement is outside this release");
    REQUIRE(!c->split_distance || (!strcmp(c->method,"sq") && c->sq_bits==4 && c->correction && *c->bias),"split distance requires SQ4 online correction and a verified bias file");
    REQUIRE(!strcmp(c->search_mode,"sqmd") || !strcmp(c->search_mode,"db-pair"),"search-mode must be sqmd or db-pair");
    if(!strcmp(c->method,"pq")) {
        REQUIRE(strcmp(c->search_mode,"db-pair"),"PQ supports SQMD; DB Pair is an SQ path");
        REQUIRE((!!subdim+!!c->pq_m+!!c->pq_compression)<=1,"choose only one PQ grouping parameter");
        if(subdim){REQUIRE(c->dim%subdim==0,"PQ subdim must divide dim");c->pq_m=c->dim/subdim;}
        if(c->pq_compression){int64_t divisor=(int64_t)c->pq_compression*c->pq_bits;REQUIRE((c->dim*32)%divisor==0,"non-integral PQ M");c->pq_m=(int)((c->dim*32)/divisor);}
        if(!c->pq_m){REQUIRE(c->dim%16==0,"set pq-m or pq-subdim explicitly for this dimension");c->pq_m=c->dim/16;}
        REQUIRE(c->pq_m>0 && c->pq_m<=c->dim && c->dim%c->pq_m==0,"PQ M must divide dim");
        c->pq_compression=c->dim*4/((c->pq_m*c->pq_bits+7)/8);
        c->search_mode="pure_pq";
    } else if(!strcmp(c->search_mode,"db-pair"))c->search_mode="db_pair";
    c->refine_mode=refine?(c->third_residual?2:1):0;
    if(c->third_residual)REQUIRE(*c->error4 && !retrieval_build_hash(c->error4,c->error4_hash),"missing or unreadable third residual artifact");
    if(!strcmp(c->command,"search"))REQUIRE(c->nq>0 && c->top_k<=c->n && *c->query && *c->graph,"search requires query, graph, nq and top-k <= n");
    REQUIRE(c->train_iters>0 && c->train_samples>0,"positive PQ training settings required");
    char v[32];snprintf(v,sizeof(v),"%d",c->source_window);setenv("SQMD_MAX_INFLIGHT_SOURCES",v,1);
    setenv("SQMD_SPLIT_DISTANCE",c->split_distance?"1":"0",1);
    return 0;
}

static FILE *results;
static int *original_ids,*index_ids;
static pthread_mutex_t output_mutex=PTHREAD_MUTEX_INITIALIZER;
int rover_results_open(void) {
    RoverConfig *c=&rover_config;
    if(*c->order_map) {
        FILE *map=fopen(c->order_map,"r");
        original_ids=malloc((size_t)c->n*sizeof(int));index_ids=malloc((size_t)c->n*sizeof(int));
        if(!map || !original_ids || !index_ids){if(map)fclose(map);return -1;}
        for(int i=0;i<c->n;i++)index_ids[i]=-1;
        for(int i=0;i<c->n;i++) {int id;
            if(fscanf(map,"%d",&id)!=1 || id<0 || id>=c->n || index_ids[id]!=-1){fclose(map);return -1;}
            original_ids[i]=id;index_ids[id]=i;
        }
        int extra;if(fscanf(map,"%d",&extra)!=EOF){fclose(map);return -1;}fclose(map);
    }
    results=fopen(c->output,"wx");return results?0:-1;
}
int rover_original_to_index(int id){return index_ids && id>=0 && id<rover_config.n?index_ids[id]:id;}
int rover_write_result(int query,const int *ids,const float *distances,int count) {
    for(int i=0;i<count;i++)if(ids[i]<0 || ids[i]>=rover_config.n || (distances && !isfinite(distances[i])))return -1;
    int rc=0;pthread_mutex_lock(&output_mutex);
    if(!results){pthread_mutex_unlock(&output_mutex);return -1;}
    fprintf(results,"{\"query\":%d,\"ids\":[",query);
    for(int i=0;i<count;i++) {if(ids[i]<0 || ids[i]>=rover_config.n){rc=-1;break;}fprintf(results,"%s%d",i?",":"",original_ids?original_ids[ids[i]]:ids[i]);}
    fprintf(results,"],\"distances\":[");
    for(int i=0;i<count;i++)fprintf(results,"%s%.9g",i?",":"",distances?distances[i]:0.0);
    fprintf(results,"]}\n"); if(ferror(results))rc=-1;
    pthread_mutex_unlock(&output_mutex);return rc;
}
int rover_results_close(void){int rc=results?fclose(results):0;results=NULL;free(original_ids);free(index_ids);original_ids=index_ids=NULL;return rc;}

const int32_t *rover_graph_to_original(void){return original_ids;}
