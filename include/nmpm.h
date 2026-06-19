#ifndef NMPM_H
#define NMPM_H

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<mpi.h>
#include<string.h>

#include"compute_cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

#if HAVE_CUDA
#include"compute_gpu.cuh"
int run_overlap_benchmark_gpu(int rank, int size, int dim, int compToPureCommRatio);
#endif


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
#define MIN_MESSAGE_SIZE (1 << 10)
#endif

#ifndef DIM
#define DIM 1
#endif

#ifndef COMP_TO_COMM_RATIO
#define COMP_TO_COMM_RATIO 100
#endif

void usage(char *prog_name);
int run_overlap_benchmark(int rank,int size, int dim, int compToPureCommRatio);

#ifdef __cplusplus
}
#endif

#endif
