#ifndef COMPUTE_H
#define COMPUTE_H

#define DIM 50

extern float **a;
extern float *x;
extern float *y;

void init_arrays(void);

void compute_on_host(double latency);

#endif
