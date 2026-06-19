#ifndef COMPUTE_GPU_CUH
#define COMPUTE_GPU_CUH

#include<cuda_runtime.h>
#include"nmpm.h"

#ifdef __cplusplus
extern "C" {
#endif

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

extern float *d_a;
extern float*h_a;


#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}



__global__ void compute_kernel_calibrat(float*d_a, size_t n, int repeat, int inner_iters);
double measure_gpu_kernel_us(float*d_a,cudaStream_t stream, int grid, int block,size_t n,int repeat,int inner_iters);
int calibrate_inner_iter(float *d_a, cudaStream_t stream,int grid, int block,size_t n,double target_unit_us);

 double compute_on_gpu(float*d_a, cudaStream_t stream, int grid, int block, size_t n, double latency_us,double unit_us, int inner_iters);
__global__ double compute_kernel(float *d_a, size_t n);
double compute_on_gpu(float*d_a, cudaStream_t stream, int grid, int block, size_t n, double latency_us);
void init_vector(int n);
void free_vector(void);
#ifdef __cplusplus
}
#endif

#endif