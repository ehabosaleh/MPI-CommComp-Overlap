#ifndef COMPUTE_CPU_H
#define COMPUTE_CPU_H

#if defined(__GNUC__) || defined(__clang__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

#define MIN_COMPUTE_SEC1.0e-6
#define COMPUTE_INNER_ITERS 128
#define TIME_CHECK_INTERVAL_SHORT 1
#define TIME_CHECK_INTERVAL_LONG 32


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


static double *mb_a = NULL;
static double *mb_b = NULL;
static double *mb_c = NULL;
static size_t mb_elems = 0;
static volatile double host_sink = 0.0;

void init_arrays(void);

void compute_on_host(double latency, int size_threshold);

#ifdef __cplusplus
}
#endif

#endif