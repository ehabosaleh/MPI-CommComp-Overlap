#ifndef COMPUTE_CPU_H
#define COMPUTE_CPU_H
#include <stdio.h>
#include <stdlib.h>
#include<mpi.h>

#if defined(__GNUC__) || defined(__clang__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

#define MIN_COMPUTE_SEC 1.0e-6
#define COMPUTE_INNER_ITERS 128
#define TIME_CHECK_INTERVAL_SHORT 1
#define TIME_CHECK_INTERVAL_LONG 32
#define MEMORY_CHUNK_ELEMS (1024*1024)

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
NOINLINE void cpu_memory_bound_batch(void);

//void compute_on_host(double latency, int size_threshold);

void compute_on_host(double latency, int compute_bound);

#ifdef __cplusplus
}
#endif

#endif