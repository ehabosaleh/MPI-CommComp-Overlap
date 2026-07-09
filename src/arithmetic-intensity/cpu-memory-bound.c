#include<stdlib.h>
#include<stdio.h>
#include<string.h>
typedef enum {
    MEMORY_MODE_TRIAD=0,
    MEMORY_MODE_COPY=1,
    MEMORY_MODE_SCALE=2,
    MEMORY_MODE_ADD=3
} memory_mode_t;

#if defined(__GNUC__) || defined(__clang__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

#define MEMORY_CHUNK_ELEMS 1024
#define TIME_CHECK_INTERVAL_SHORT 1
#define TIME_CHECK_INTERVAL_LONG 32
#define A 2.0

volatile double host_sink=0.0;

static size_t mb_offset=0;

double *mb_a=NULL;
double *mb_b=NULL;
double *mb_c=NULL;
size_t mb_elems=0;


int init_memory_bound_buffers(size_t bytes){

	mb_elems=bytes/sizeof(double);

    if(posix_memalign((void **)&mb_a,64,mb_elems * sizeof(double)) != 0)
		return -1;

    if(posix_memalign((void **)&mb_b,64,mb_elems*sizeof(double)) != 0)
        return -1;

    if(posix_memalign((void **)&mb_c,64,mb_elems*sizeof(double)) != 0)
        return -1;

    for(size_t i=0; i<mb_elems;i++) {
        mb_a[i]=1.0;
        mb_b[i]=2.0;
        mb_c[i]=0.0;
    }

    return 0;
}

void free_memory_bound_buffers(void){
    free(mb_a);
    free(mb_b);
    free(mb_c);

    mb_a = NULL;
    mb_b = NULL;
    mb_c = NULL;
    mb_elems = 0;
}

NOINLINE void cpu_memory_bound_batch(memory_mode_t memory_mode){
    if (mb_a == NULL || mb_b == NULL || mb_c == NULL || mb_elems == 0) {
        fprintf(stderr, "Memory-bound buffers are not initialized.\n");
        return;
    }

    size_t chunk = MEMORY_CHUNK_ELEMS;

    if (chunk > mb_elems)
        chunk = mb_elems;

    if (mb_offset + chunk > mb_elems)
        mb_offset = 0;

    size_t start = mb_offset;
    size_t end   = mb_offset + chunk;
    switch (memory_mode) {
        case MEMORY_MODE_TRIAD:
            for (size_t i = start; i < end; i++) {
                mb_c[i] = mb_a[i] + A * mb_b[i];
            }
            break;
        case MEMORY_MODE_COPY:
            for (size_t i = start; i < end; i++) {
                mb_c[i] = mb_a[i];
            }
            break;
        case MEMORY_MODE_SCALE:
            for (size_t i = start; i < end; i++) {
                mb_c[i] = A * mb_a[i];
            }
            break;
        case MEMORY_MODE_ADD:
            for (size_t i = start; i < end; i++) {
                mb_c[i] = mb_a[i] + mb_b[i];
            }
            break;
        default:
            fprintf(stderr, "Invalid memory mode specified.\n");
            return;
    }

    mb_offset += chunk;

    host_sink += mb_c[end - 1];
}
int main(int argc,char**argv){

    char *mode_str=argv[1];
    memory_mode_t mode=MEMORY_MODE_TRIAD;
    if(strcmp(mode_str,"triad")==0){
        mode=MEMORY_MODE_TRIAD;
    }else if(strcmp(mode_str,"copy")==0){
        mode=MEMORY_MODE_COPY;
    }else if(strcmp(mode_str,"scale")==0){
        mode=MEMORY_MODE_SCALE;
    }else if(strcmp(mode_str,"add")==0){
        mode=MEMORY_MODE_ADD;
    }else{
        fprintf(stderr,"Error: unknown memory mode %s\n",mode_str);
        return 1;
    }   

	init_memory_bound_buffers(256UL*1024UL*1024UL);
	
	for(int i=0;i<1000000;i++)
		cpu_memory_bound_batch(mode);
        
	free_memory_bound_buffers();

}
