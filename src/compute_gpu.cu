#include"compute_gpu.cuh"
#include"compute_cpu.h"

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


__global__ void memory_bound_kernel(float *__restrict__ d_c, const float *__restrict__ d_a,const float *__restrict__ d_b,size_t elems_per_pass,int passes, size_t max_elems,float alpha, memory_mode_t mode){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    if(elems_per_pass==0)
        return;
    const size_t num_chunks = max_elems/elems_per_pass;
    if (num_chunks == 0) {
        return;
    }

    for (int p=0;p<passes;p++){
        const size_t chunk=(size_t)p%num_chunks;
	    size_t base=(size_t)chunk*elems_per_pass;
        

	    for (size_t i = idx; i < elems_per_pass; i += stride) {
		    size_t j=base+i;

            switch(mode){
                case MEMORY_MODE_TRIAD:
                    d_c[j] = d_a[j] + alpha*d_b[j];
                    break;
                case MEMORY_MODE_COPY:
                    d_c[j] = d_a[j];
                    break;
                case MEMORY_MODE_SCALE:
                    d_c[j] = alpha*d_a[j];
                    break;
                case MEMORY_MODE_ADD:
                    d_c[j] = d_a[j] + d_b[j];
                    break;
                default:
                    d_c[j] = d_a[j] + alpha*d_b[j]; 
            }
    
        }
    }
}
double measure_gpu_memory_bound_kernel_us(float *d_c, const float *d_a,const float *d_b,cudaStream_t stream, int grid, int block,size_t elems_per_pass,int passes,size_t max_elems,float alpha, memory_mode_t mode,progress_thread_data_t *progress_data){
    if(elems_per_pass==0 || passes<=0){
        return 0.0;
    }

    /*
    size_t required_elems=(size_t)passes*elems_per_pass;
    
    if(required_elems>max_elems){
        fprintf(stderr,"Error: required elements (%zu) exceed max_elems (%zu)\n",required_elems,max_elems);
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    */
    float time_ms=0.0f;
    cudaEvent_t start,stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start,stream));
    memory_bound_kernel<<<grid,block,0,stream>>>(d_c,d_a,d_b,elems_per_pass,passes,max_elems,alpha,mode);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
    CHECK_CUDA_ERROR(cudaEventRecord(stop,stream));
    if(progress_data!=NULL&&!progress_data->is_thread){
        int req_flag=0;
        while(!req_flag){
            cudaError_t event_status = cudaEventQuery(stop);
            if(event_status ==cudaSuccess)
                break;
            if (event_status!=cudaErrorNotReady){
                CHECK_CUDA_ERROR(event_status);
            }
            MPI_Testall(progress_data->num_requests,progress_data->requests,&req_flag,MPI_STATUSES_IGNORE);
        }
    }
    
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_ms,start,stop));
    
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    return (double)time_ms*1000.0;
}


gpu_memory_calibration_t calibrate_memory_bound_kernel(float *d_c, const float *d_a,const float *d_b, cudaStream_t stream, int grid, int block, size_t elems_per_pass, size_t max_elems, double target_unit_us, memory_mode_t mode){
    //size_t best_elems=max_elems;

    int low=1;
    int high=(int)(max_elems / elems_per_pass);
    if(high<1){
        fprintf(stderr,"Error: max_elems (%zu) is less than elems_per_pass (%zu)\n",max_elems,elems_per_pass);
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    int best_passes=1;

    double best_error=1e30;
    double best_time_us=0.0;

    for(int i=0;i<5;i++){
        measure_gpu_memory_bound_kernel_us(d_c,d_a,d_b,stream,grid,block,elems_per_pass,1,max_elems,50.0f,mode,NULL);
    }

    for(int iter=0;iter<30 && low<=high;iter++){
        int mid=low+(high-low)/2;

        double time_us=measure_gpu_memory_bound_kernel_us(d_c,d_a,d_b,stream,grid,block,elems_per_pass,mid,max_elems,50.0f,mode,NULL);

        if(time_us<=0.0){
            fprintf(stderr,"Invalid memory-bound calibration time\n");
            MPI_Abort(MPI_COMM_WORLD,1);
        }

        double error=fabs(time_us-target_unit_us);

        if(error<best_error){
            best_error=error;
            best_passes=mid;
            best_time_us=time_us;
        }

        if(time_us<target_unit_us){
            low=mid+1;
        }else{
            high=mid-1;
        }
    }
   
    double measured_pass_us=best_time_us/(double)best_passes;
    double logical_bytes_per_element=0.0;
    switch (mode) {
        case MEMORY_MODE_COPY:
        case MEMORY_MODE_SCALE:
           logical_bytes_per_element=2.0*sizeof(float);

        case MEMORY_MODE_ADD:
        case MEMORY_MODE_TRIAD:
        default:
            logical_bytes_per_element=3.0*sizeof(float);
    }

    double bytes=(double)elems_per_pass*logical_bytes_per_element*(double)best_passes;
    double bytes_per_us=bytes/best_time_us;

    gpu_memory_calibration_t cal;
    cal.elems_per_pass=elems_per_pass;
    cal.measured_unit_us=measured_pass_us;
    cal.bytes_per_us=bytes_per_us;
    cal.gb_per_s=bytes_per_us*1e6/1e9;
    printf("Calibrated memory-bound kernel: %zu elements, %d passes, %.6f us total, %.6f us/pass, %.3f bytes/us, %.3f GB/s\n",cal.elems_per_pass,best_passes,best_time_us,cal.measured_unit_us,cal.bytes_per_us,cal.gb_per_s);
    fflush(stdout);
    return cal;
}

__global__ void compute_bound_kernel(float*d_a, size_t n, int repeat, int inner_iters){

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i=idx; i<n; i+=stride){
        float x=d_a[i];
        for(int r=0;r<repeat;r++){
            #pragma unroll 1
            for(int k=0;k<inner_iters;k++) {
                x=x*1.000001f+0.000001f;
            }
        }

        d_a[i]=x;
    }
}
double measure_gpu_compute_bound_kernel(float*d_a,cudaStream_t stream, int grid, int block,size_t n,int repeat,int inner_iters,progress_thread_data_t *progress_data){
    float time_ms=0.0f;
    cudaEvent_t start,stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start,stream));
    compute_bound_kernel<<<grid,block,0,stream>>>(d_a,n,repeat,inner_iters);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
    CHECK_CUDA_ERROR(cudaEventRecord(stop,stream));
    if(progress_data!=NULL&&!progress_data->is_thread){
        int req_flag=0;
        while(!req_flag){
            cudaError_t event_status = cudaEventQuery(stop);
            if(event_status ==cudaSuccess)
                break;
            if (event_status!=cudaErrorNotReady){
                CHECK_CUDA_ERROR(event_status);
            }
            MPI_Testall(progress_data->num_requests,progress_data->requests,&req_flag,MPI_STATUSES_IGNORE);
        }
    }
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_ms,start,stop));
    
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    return (double)time_ms*1000.0;
}
int calibrate_inner_iter(float *d_a, cudaStream_t stream,int grid, int block,size_t n,double target_unit_us, double* measured_unit_us){
    const int calibration_repeat = 100;
    int low=1;
    int high=10000;
    int best_inner_iters=1;
    double best_error=1e30;
    double best_unit_us=0.0;
    int rank=0;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    for(int i=0;i<100;i++){
        measure_gpu_compute_bound_kernel(d_a,stream,grid,block,n,calibration_repeat,100, NULL);
    }

    for (int iter=0;iter<100;iter++){
        int mid=low+(high-low)/2;

        double total_us=measure_gpu_compute_bound_kernel(d_a,stream,grid,block,n,calibration_repeat,mid,NULL);

        double unit_us=total_us/(double)calibration_repeat;
         // unit_us is the cost of a single repeat; mid represents the corresponding inner_repeat//
         
        double error=fabs(unit_us-target_unit_us);

        if(error<best_error){
            best_error=error;
            best_inner_iters=mid;
            best_unit_us=unit_us;
        }

        if(unit_us<target_unit_us) {
            low=mid+1;
        }else{
            high=mid-1;
        }
    }
    if(best_unit_us<=0.0){
        fprintf(stderr,"Invalid compute-bound calibration time\n");
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    *measured_unit_us=best_unit_us;
    printf("Rank %d: Calibrated inner iterations: %d (measured unit time: %.3f us, target: %.3f us)\n",rank, best_inner_iters, best_unit_us, target_unit_us);
    fflush(stdout);
    return best_inner_iters; 
}
double compute_on_gpu(float*d_a, cudaStream_t stream, int grid, int block, size_t n, double latency_us,double unit_us, int inner_iters, size_t max_elems,gpu_memory_calibration_t cal, int compute_bound, memory_mode_t mode,progress_thread_data_t *progress_data){
    if(latency_us<=0.0){
        return 0.0;
    }
    if(compute_bound){
        int repeat = (int)ceil(latency_us/unit_us);
        return measure_gpu_compute_bound_kernel(d_a,stream,grid,block,n,repeat,inner_iters,progress_data);
    }else{ 
        int passes = (int)ceil(latency_us/cal.measured_unit_us);

        if(passes<1){
            passes=1;
        }
        return measure_gpu_memory_bound_kernel_us(d_c,d_a,d_b,stream,grid,block,cal.elems_per_pass,passes,max_elems,1.0f,mode,progress_data);
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
