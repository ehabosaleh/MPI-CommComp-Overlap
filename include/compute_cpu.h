#ifndef COMPUTE_CPU_H
#define COMPUTE_CPU_H

#ifdef __cplusplus
extern "C" {
#endif

#define ARRAY_DIM 25
#define A 2.0
#define SIZE_THRESHOLD 65536
extern float *a;
extern float *x;
extern float *y;

void init_arrays(void);

void compute_on_host(double latency, int size_threshold);

#ifdef __cplusplus
}
#endif

#endif