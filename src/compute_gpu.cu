#include"compute_gpu.cuh"

float *d_a=nullptr;
float *h_a=nullptr;

__global__ void compute_kernel(float *d_a, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        d_a[i] = d_a[i] * d_a[i]+1.0f;
    }
}

double compute_on_gpu(float*d_a, cudaStream_t stream, int grid, int block, size_t n, double latency_us){
    cudaEvent_t start, stop;
    double kernel_ms=0.0,elapsed_time_us=0.0,time_ms=0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    while(elapsed_time_us<=latency_us){

        cudaEventRecord(start,stream);
        compute_kernel<<<grid,block,0,stream>>>(d_a,n);
        cudaEventRecord(stop,stream);
        cudaEventSynchronize(stop);
        cudaError_t err=cudaGetLastError();

        if(err!=cudaSuccess){
            fprintf(stderr,"CUDA kernel error: %s\n", cudaGetErrorString(err));
            break;
        }
        cudaEventElapsedTime(&time_ms,start,stop);
        elapsed_time+=(double)time_ms*1000.0;
    }
    CUDA_CHECK_ERROR(cudaEventDestroy(start));
    CUDA_CHECK_ERROR(cudaEventDestroy(stop));

    return elapsed_time;
}

void init_vector(int n) {
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&h_a,n*sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a,n*sizeof(float))); 
    for(int i=0;i<n;i++)
            h_a[i]=i;
    CHECK_CUDA_ERROR(cudaMemcpy(d_a,h_a,sizeof(float)*n,cudaMemcppyHostToDevice));              
}
void free_vector(void){
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFreeHost(h_a));

}

