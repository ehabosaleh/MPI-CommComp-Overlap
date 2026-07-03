#ifndef COMPUTE_GPU_CUH
#define COMPUTE_GPU_CUH

#include<cuda_runtime.h>
#include"nmpm.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VECTOR_DIM 160000000

typedef enum{
    H2D=0,
    D2H=1,
    D2D=2,
    ALLTOALL=3
} TransferType;

typedef struct {
    size_t elems_per_pass;
    double measured_unit_us;
    double bytes_per_us;
    double gb_per_s;
} gpu_memory_calibration_t;

typedef enum{
    TPB_64=64, 
    TPB_128=128,
    TPB_256=256,
    TPB_512=512,
    TPB_1024=1024
}threadsPerBlock;

extern float *d_a;
extern float *d_b;
extern float*h_a;
extern float*h_b;
extern float *d_c;
extern float *h_c;


#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}



__global__ void compute_bound_kernel(float*d_a, size_t n, int repeat, int inner_iters);
double measure_gpu_compute_bound_kernel(float*d_a,cudaStream_t stream, int grid, int block,size_t n,int repeat,int inner_iters,int req_count, MPI_Request *reqs,int do_progress);
int calibrate_inner_iter(float *d_a, cudaStream_t stream,int grid, int block,size_t n,double target_unit_us);

__global__ void memory_bound_kernel(float *__restrict__ d_c, const float *__restrict__ d_a,const float *__restrict__ d_b,size_t elems_per_pass,int passes, size_t max_elems,float alpha);
double measure_gpu_memory_bound_kernel_us(float *d_c, const float *d_a,const float *d_b,cudaStream_t stream, int grid, int block,size_t elems_per_pass,int passes,size_t max_elems,float alpha,int req_count, MPI_Request *reqs,int do_progress);
gpu_memory_calibration_t calibrate_memory_bound_kernel(float *d_c, const float *d_a,const float *d_b, cudaStream_t stream, int grid, int block, size_t max_elems, double target_unit_us);

double compute_on_gpu(float*d_a, cudaStream_t stream, int grid, int block, size_t n, double latency_us,double unit_us, int inner_iters, size_t max_elems,gpu_memory_calibration_t cal, int req_count, MPI_Request *reqs , int do_progress, int compute_bound);
__global__ void compute_kernel(float *d_a, size_t n);
void init_vector(size_t n);
void free_vector(void);
#ifdef __cplusplus
}
#endif

#endif
