#ifndef COMPUTE_CPU_H
#define COMPUTE_CPU_H
#include <stdio.h>
#include <stdlib.h>
#include<mpi.h>
#include<pthread.h>


typedef void (*host_work_fn_t)(void);

typedef enum {
    MEMORY_MODE_TRIAD,
    MEMORY_MODE_COPY,
    MEMORY_MODE_SCALE,
    MEMORY_MODE_ADD
} memory_mode_t;

typedef struct{
    pthread_t thread;
    MPI_Request *requests;
    int num_requests;
    int stop_flag;
    int active;
    int done;
    int terminate;
    int cuda_device;
    int is_gpu;
    int is_thread;

} progress_thread_data_t;

#define atomic_load_int(ptr) \
    __atomic_load_n((ptr), __ATOMIC_ACQUIRE)

#define atomic_store_int(ptr, val) \
    __atomic_store_n((ptr), (val), __ATOMIC_RELEASE)

#if defined(__GNUC__) || defined(__clang__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

#define MIN_COMPUTE_SEC 1.0e-6
#define COMPUTE_INNER_ITERS 128
#define TIME_CHECK_INTERVAL_SHORT 1
#define TIME_CHECK_INTERVAL_LONG 32
#define MEMORY_CHUNK_ELEMS 1024

#ifdef __cplusplus
#define RESTRICT __restrict__
extern "C" {
#else
#define RESTRICT restrict
#endif

#define ARRAY_DIM 25
#define SIZE_THRESHOLD 65536
#define A 0.5

extern double x[ARRAY_DIM];
extern double a[ARRAY_DIM * ARRAY_DIM];
extern double y[ARRAY_DIM];


extern double *mb_a;
extern double *mb_b;
extern double *mb_c;
extern size_t mb_elems;

extern volatile double host_sink;

int init_memory_bound_buffers(size_t bytes);
void free_memory_bound_buffers(void);

NOINLINE void cpu_compute_bound_batch(void);
NOINLINE void cpu_memory_bound_triad();
NOINLINE void cpu_memory_bound_copy();
NOINLINE void cpu_memory_bound_add();
NOINLINE void cpu_memory_bound_scale();

void * progress_thread_func(void *arg);
int start_progress_thread(progress_thread_data_t *progress_data);
int post_progress_thread_requests(progress_thread_data_t *progress_data, MPI_Request *requests, int num_requests);
int terminate_progress_thread(progress_thread_data_t *progress_data);
int wait_progress_thread(progress_thread_data_t *progress_data);

//void compute_on_host(double latency, int size_threshold);

void compute_on_host(double latency, int compute_bound,memory_mode_t memory_mode);
#ifdef __cplusplus
}
#endif

#endif