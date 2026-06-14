#ifndef COMPUTE_H
#define COMPUTE_H

#define ARRAY_DIM 5

extern float *a;
extern float *x;
extern float *y;

void init_arrays(void);

void compute_on_host(double latency);

#endif
