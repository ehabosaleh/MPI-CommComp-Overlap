#include"compute_gpu.cuh"

float *d_a=nullptr;
float *h_a=nullptr;

__global__ void compute_kernel(float *d_a, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        d_a[i] = d_a[i] * d_a[i];
    }
}

void init_vector(int n) {
    CHECK(cudaMalloc((void**)&d_a,n));            
}

