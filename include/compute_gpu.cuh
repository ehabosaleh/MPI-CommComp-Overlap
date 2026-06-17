#ifndef COMPUTE_GPU_CUH
#define COMPUTE_GPU_CUH

#include<cuda_runtime.h>
#include"nmpm.h"

#define VECTOR_DIM 1000000

typedef enum{
    H2D=0,
    D2H=1,
    D2D=2,
    ALLTOALL=3
} TransferType;

typedef enum{
    TPB_64=64,
    TPB_128=128,
    TPB_256=256,
    TPB_512=512,
    TPB_1024=1024
}threadsPerBlock;

extern cudaDeviceProp deviceProp;
extern int deviceCount;
extern int deviceID;
extern float *d_a;
extern float*h_a;


#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}


__global__ void compute_kernel(float *d_a, size_t n);

void gpu_init_arrays(int n);

#endif