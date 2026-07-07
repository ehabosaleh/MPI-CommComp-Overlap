#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}
typedef enum{
    TPB_64=64, 
    TPB_128=128,
    TPB_256=256,
    TPB_512=512,
    TPB_1024=1024
}threadsPerBlock;

 float *d_a=NULL;
 float *d_b=NULL;
 float *h_a=NULL;
 float *h_b=NULL;
 float *d_c=NULL;
 float *h_c=NULL;

__global__ void compute_bound_kernel(float*d_a, size_t n, int repeat, int inner_iters){

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i=idx; i<n; i+=stride){
        float x=d_a[i];
        for(int r=0;r<repeat;r++){
            #pragma unroll 1
            for(int k = 0; k < inner_iters; k++) {
                x = x * 1.000001f + 0.000001f;
            }
        }

        d_a[i]=x;
    }
}

void init_vector(size_t n) {
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&h_a,n*sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a,n*sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&h_b,n*sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b,n*sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&h_c,n*sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_c,n*sizeof(float)));
    for(int i=0;i<n;i++){
            h_a[i]=1.0f;
            h_b[i]=2.0f;
            h_c[i]=0.0f;    
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_a,h_a,n*sizeof(float),cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b,h_b,n*sizeof(float),cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_c,h_c,n*sizeof(float),cudaMemcpyHostToDevice));
}
void free_vector(void){
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFreeHost(h_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFreeHost(h_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));
    CHECK_CUDA_ERROR(cudaFreeHost(h_c));
}

int main(){
    size_t n=1000000;
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    init_vector(n);
    
    compute_bound_kernel<<<32,TPB_256,0,stream>>>(d_a,n,100,100);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaStreamSynchronize(stream));
    
    free_vector();
    return 0;

}