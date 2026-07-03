#include"compute_gpu.cuh"

float *d_a=nullptr;
float *h_a=nullptr;
float *d_b=nullptr;
float *h_b=nullptr;
float *d_c=nullptr;
float *h_c=nullptr;


__global__ void compute_kernel(float *d_a, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        d_a[i] = d_a[i] * d_a[i]+1.0f;
    }
}
__global__ void memory_bound_kernel(float *__restrict__ d_c, const float *__restrict__ d_a,const float *__restrict__ d_b,size_t elems_per_pass,int passes, size_t max_elems,float alpha){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (int p=0;p<passes;p++){
        for (size_t i = idx; i < elems_per_pass; i += stride) {
                d_c[i] = alpha*d_a[i] + d_b[i];
        }
    
    }
}

double measure_gpu_memory_bound_kernel_us(float *d_c, const float *d_a,const float *d_b,cudaStream_t stream, int grid, int block,size_t elems_per_pass,int passes,size_t max_elems,float alpha,int req_count, MPI_Request *reqs,int do_progress){
    float time_ms=0.0f;
    cudaEvent_t start,stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start,stream));
    memory_bound_kernel<<<grid,block,0,stream>>>(d_c,d_a,d_b,elems_per_pass,passes,max_elems,alpha);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
    CHECK_CUDA_ERROR(cudaEventRecord(stop,stream));
    if(do_progress)
        if(req_count>0){
                int mpi_done=0;
                while(!mpi_done){
                    MPI_Testall(req_count,reqs,&mpi_done,MPI_STATUSES_IGNORE);
                }

        }
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_ms,start,stop));
    
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    return (double)time_ms*1000.0;
}

gpu_memory_calibration_t calibrate_memory_bound_kernel(float *d_c, const float *d_a,const float *d_b, cudaStream_t stream, int grid, int block, size_t max_elems, double target_unit_us){
    size_t low=1024;
    size_t high=max_elems;

    size_t best_elems=0;
    double best_error=1e30;
    double best_time_us=0.0;
    size_t warmup_elems = max_elems < (1UL << 20) ? max_elems : (1UL << 20);

    for(int i=0;i<5;i++){
        measure_gpu_memory_bound_kernel_us(d_c,d_a,d_b,stream,grid,block,warmup_elems,1,max_elems,1.0f,0,NULL,0);
        //printf("Warmup iteration %d completed for memory-bound kernel calibration\n", i + 1);
    }
    for (int iter = 0; iter < 30 && low <= high; iter++){
        size_t mid =low+(high-low)/2;

        double time_us=measure_gpu_memory_bound_kernel_us(d_c,d_a,d_b,stream,grid,block,mid,1,max_elems,1.0f,0,NULL,0);

        double error=fabs(time_us-target_unit_us);

        if(error<best_error){
            best_error=error;
            best_elems=mid;
            best_time_us=time_us;
        }

        if(time_us < target_unit_us){
            low=mid+1;
        }else{
            if(mid==0){
                break;
            }
            high=mid-1;
        }
        if (best_time_us <= 0.0) {
            fprintf(stderr, "Invalid memory-bound calibration time\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

    }

    double bytes=(double)best_elems * 3.0 * sizeof(float);
    double bytes_per_us=bytes/best_time_us;

    gpu_memory_calibration_t cal;
    cal.elems_per_pass = best_elems;
    cal.measured_unit_us = best_time_us;
    cal.bytes_per_us = bytes_per_us;
    cal.gb_per_s = bytes_per_us * 1e6 / 1e9;
    
    printf("Calibrated memory-bound kernel: %ld elements, %.3f us, %.3f bytes/us, %.3f GB/s\n", cal.elems_per_pass, cal.measured_unit_us, cal.bytes_per_us, cal.gb_per_s);
    return cal;
}
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
double measure_gpu_compute_bound_kernel(float*d_a,cudaStream_t stream, int grid, int block,size_t n,int repeat,int inner_iters, int req_count, MPI_Request *reqs, int do_progress){
    float time_ms=0.0f;
    cudaEvent_t start,stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start,stream));
    compute_bound_kernel<<<grid,block,0,stream>>>(d_a,n,repeat,inner_iters);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
    CHECK_CUDA_ERROR(cudaEventRecord(stop,stream));
    if(do_progress)
        if(req_count>0){
                int mpi_done=0;
                while(!mpi_done){
                    MPI_Testall(req_count,reqs,&mpi_done,MPI_STATUSES_IGNORE);
                }

        }
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_ms,start,stop));
    
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    return (double)time_ms*1000.0;
}
int calibrate_inner_iter(float *d_a, cudaStream_t stream,int grid, int block,size_t n,double target_unit_us){
    const int calibration_repeat = 100;
    int low=1;
    int high=10000;
    int best_inner_iters=1;
    double best_error=1e30;
    for(int i=0;i<5;i++){
        measure_gpu_compute_bound_kernel(d_a,stream,grid,block,n,calibration_repeat,100,0,NULL,0);
    }

    for (int iter=0;iter<30;iter++){
        int mid=low+(high-low)/2;

        double total_us = measure_gpu_compute_bound_kernel(d_a,stream,grid,block,n,calibration_repeat,mid,0,NULL,0);

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
    printf("Calibrated inner iterations: %d (measured unit time: %.3f us, target: %.3f us)\n", best_inner_iters, best_error + target_unit_us, target_unit_us);

    return best_inner_iters;
    
}
double compute_on_gpu(float*d_a, cudaStream_t stream, int grid, int block, size_t n, double latency_us,double unit_us, int inner_iters, size_t max_elems,gpu_memory_calibration_t cal, int req_count, MPI_Request *reqs , int do_progress, int compute_bound){
    if(latency_us<=0.0){
        return 0.0;
    }
    if(compute_bound){
        int repeat = (int)ceil(latency_us/unit_us);
        return measure_gpu_compute_bound_kernel(d_a,stream,grid,block,n,repeat,inner_iters, req_count,reqs,do_progress);
    }else{  

        
        int passes = (int)ceil(latency_us/cal.measured_unit_us);
        size_t required_elems = cal.elems_per_pass*(size_t)passes;

        if(required_elems>max_elems){
            passes = (int)ceil((double)max_elems/(double)cal.elems_per_pass);
        }
        if(passes<1){
            passes=1;
        }
        return measure_gpu_memory_bound_kernel_us(d_c,d_a,d_b,stream,grid,block,cal.elems_per_pass,passes, max_elems,1.0f,req_count,reqs,do_progress);
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
            h_a[i]=i;
            h_b[i]=i;
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
