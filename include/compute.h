#ifndef COMPUTE_H
#define COMPUTE_H

#define ARRAY_DIM 25
#define A 2.0
#define SIZE_THRESHOLD 65536
extern float *a;
extern float *x;
extern float *y;

void init_arrays(void);

void compute_on_host(double latency, int size_threshold);

#endif
