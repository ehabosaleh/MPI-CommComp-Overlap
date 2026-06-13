#ifndef NMPM_H
#define NMPM_H

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<mpi.h>
#include<string.h>

#ifndef MAX_ITER
#define MAX_ITER 10000
#endif

#ifndef SKIP
#define SKIP 1000
#endif


#ifndef MAX_MESSAGE_SIZE
#define MAX_MESSAGE_SIZE (1 << 20)
#endif

#ifndef MIN_MESSAGE_SIZE
#define MIN_MESSAGE_SIZE (1 << 0)
#endif

#ifndef DIM
#define DIM 1
#endif

#ifndef COMP_TO_COMM_RATIO
#define COMP_TO_COMM_RATIO 100
#endif

void usage(char *prog_name);
int run_overlap_benchmark(int rank,int size, int dim, int compToPureCommRatio);

#endif

