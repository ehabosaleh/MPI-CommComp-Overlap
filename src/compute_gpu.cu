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

__global__ void compute_kernel_calibrate(float*d_a, size_t n, int repeat, int inner_iters){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (size_t i=idx; i<n; i+=stride){
        float x=d_a[i];
        for (int r = 0; r < repeat; r++) {
            #pragma unroll 1
            for (int k=0;k<inner_iters;k++) {
                x=x*1.000001f+0.000001f;
            }
        }

        d_a[i]=x;
    }
}

double measure_gpu_kernel_us(float*d_a,cudaStream_t stream, int grid, int block,size_t n,int repeat,int inner_iters){
    float time_ms=0.0f;
    cudaEvent_t start,stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start,stream));
    compute_kernel_calibrate<<<block,grid,0,stream>>>(d_a,n,repeat,inner_iters);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
    CHECK_CUDA_ERROR(cudaEventRecord(stop,stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_ms,start,stop));
    
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    return (double)time_ms*1000.0;
}
int calibrate_inner_iter(float *d_a, cudaStream_t stream,int grid, int block,size_t n,double target_unit_us){
    const int calibration_repeat = 1000;
    int low=1;
    int high=100000;
    int best_inner_iters=1;
    double best_error=1e30;

    for (int iter=0;iter<30;iter++){
        int mid=low+(high-low)/2;

        double total_us = measure_gpu_kernel_us(d_a,stream,grid,block,n,calibration_repeat,mid);

        double unit_us=total_us/(double)calibration_repeat;
        double error=fabs(unit_us-target_unit_us);

        if(error<best_error){
            best_error=error;
            best_inner_iters=mid;
        }

        if(unit_us<target_unit_us) {
            low=mid+1;
        }else{
            high=mid-1;
        }
    }

    return best_inner_iters;
    
}
double compute_on_gpu(float*d_a, cudaStream_t stream, int grid, int block, size_t n, double latency_us,double unit_us, int inner_iters){
    if(latency_us<0.0){
        return 0.0;
    }
    int repeat = (int)ceil(latency_us/unit_us);
   return measure_gpu_kernel_us(d_a,stream,grid,block,n,repeat,inner_iters);
}

void init_vector(int n) {
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&h_a,n*sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a,n*sizeof(float))); 
    for(int i=0;i<n;i++)
            h_a[i]=i;
    CHECK_CUDA_ERROR(cudaMemcpy(d_a,h_a,sizeof(float)*n,cudaMemcpyHostToDevice));              
}
void free_vector(void){
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFreeHost(h_a));

}