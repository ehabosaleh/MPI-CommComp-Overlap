#include<stdio.h>
#include<cuda_runtime.h>
#include<stdlib.h>

#define ELEMS_PER_PASS 1000000UL
#define MAX_UNIQUE_PASSES 160UL
#define VECTOR_DIM_MEM (ELEMS_PER_PASS * MAX_UNIQUE_PASSES)

typedef enum {
    MEMORY_MODE_TRIAD=0,
    MEMORY_MODE_COPY=1,
    MEMORY_MODE_SCALE=2,
    MEMORY_MODE_ADD=3
} memory_mode_t;

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

int main(int argc, char **argv){
    char *mode_str=argv[1];
    memory_mode_t mode=MEMORY_MODE_TRIAD;
    if(strcmp(mode_str,"triad")==0){
        mode=MEMORY_MODE_TRIAD;
    }else if(strcmp(mode_str,"copy")==0){
        mode=MEMORY_MODE_COPY;
    }else if(strcmp(mode_str,"scale")==0){
        mode=MEMORY_MODE_SCALE;
    }else if(strcmp(mode_str,"add")==0){
        mode=MEMORY_MODE_ADD;
    }else{
        fprintf(stderr,"Error: unknown memory mode %s\n",mode_str);
        return 1;
    }   
    int device_num=0;
    int device_count=0;

    CHECK_CUDA_ERROR(cudaGetDeviceCount(&device_count));
    if(device_count<=0){
        fprintf(stderr,"No CUDA devices found\n");
        return 1;
    }
	CHECK_CUDA_ERROR(cudaSetDevice(device_num));
	cudaDeviceProp prop;
	CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop,device_num));
    int grid=prop.multiProcessorCount*4;
	int block=TPB_256;

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));


    size_t n=VECTOR_DIM_MEM;
    init_vector(n);
    memory_bound_kernel<<<grid,block,0,stream>>>(d_c,d_a,d_b,ELEMS_PER_PASS,MAX_UNIQUE_PASSES,n,1.0f,mode);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    free_vector();
    return 0;
}