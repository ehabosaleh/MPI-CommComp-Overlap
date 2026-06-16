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
void gpu_init_arrays(void) {
    h_a = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
    }
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, N * sizeof(float)));
}

