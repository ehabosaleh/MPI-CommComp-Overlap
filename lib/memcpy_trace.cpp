#define _GNU_SOURCE
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

static int get_rank(){
    const char *e=nullptr;
    e=getenv("PMI_RANK");
    if(e)
        return atoi(e);

    e=getenv("OMPI_COMM_WORLD_RANK");
    if(e)
        return atoi(e);
    
    e=getenv("SLURM_PROCID");
    if(e)
		return atoi(e);
    
        e=getenv("MPI_LOCALRANKID");
    if(e)
	    return atoi(e);

    return -1;
}

static const char *kind_str(cudaMemcpyKind kind){
    switch (kind){
        case cudaMemcpyHostToHost:
            return "HtoH";
        case cudaMemcpyHostToDevice:
            return "HtoD";
        case cudaMemcpyDeviceToHost:
            return "DtoH";
        case cudaMemcpyDeviceToDevice:
            return "DtoD";
        case cudaMemcpyDefault:
            return "Default";
        default:
            return "Unknown";
    }
}

extern "C" cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
{
    using fn_t = cudaError_t (*)(void *, const void *, size_t, cudaMemcpyKind);
    static fn_t real_fn = nullptr;

    if (!real_fn)
        real_fn=(fn_t)dlsym(RTLD_NEXT, "cudaMemcpy");

    fprintf(stderr,"[CUDA_COPY_TRACE] rank=%d pid=%d cudaMemcpy kind=%s bytes=%zu\n",get_rank(), getpid(), kind_str(kind), count);
    fflush(stderr);

    return real_fn(dst, src, count, kind);
}

extern "C" cudaError_t cudaMemcpyAsync(void *dst, const void *src,size_t count, cudaMemcpyKind kind,cudaStream_t stream)
{
    using fn_t = cudaError_t(*)(void *,const void *, size_t,cudaMemcpyKind, cudaStream_t);
    static fn_t real_fn = nullptr;

    if (!real_fn)
        real_fn = (fn_t)dlsym(RTLD_NEXT, "cudaMemcpyAsync");

    fprintf(stderr,"[CUDA_COPY_TRACE] rank=%d pid=%d cudaMemcpyAsync kind=%s bytes=%zu\n",
            get_rank(), getpid(), kind_str(kind), count);
    fflush(stderr);

    return real_fn(dst, src, count, kind, stream);
}

extern "C" CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice,size_t bytes)
{
    using fn_t=CUresult(*)(void *,CUdeviceptr, size_t);
    static fn_t real_fn=nullptr;

    if(!real_fn)
        real_fn=(fn_t)dlsym(RTLD_NEXT,"cuMemcpyDtoH");

    fprintf(stderr,"[CUDA_COPY_TRACE] rank=%d pid=%d cuMemcpyDtoH bytes=%zu\n",get_rank(), getpid(), bytes);
    fflush(stderr);

    return real_fn(dstHost, srcDevice, bytes);
}

extern "C" CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost,size_t bytes)
{
    using fn_t=CUresult(*)(CUdeviceptr, const void *, size_t);
    static fn_t real_fn = nullptr;

    if (!real_fn)
        real_fn=(fn_t)dlsym(RTLD_NEXT, "cuMemcpyHtoD");

    fprintf(stderr,"[CUDA_COPY_TRACE] rank=%d pid=%d cuMemcpyHtoD bytes=%zu\n",get_rank(), getpid(), bytes);
    fflush(stderr);

    return real_fn(dstDevice, srcHost, bytes);
}

extern "C" CUresult cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice,size_t bytes, CUstream stream)
{
    using fn_t = CUresult (*)(void *,CUdeviceptr, size_t,CUstream);
    static fn_t real_fn=nullptr;

    if (!real_fn)
        real_fn = (fn_t)dlsym(RTLD_NEXT, "cuMemcpyDtoHAsync");

    fprintf(stderr,"[CUDA_COPY_TRACE] rank=%d pid=%d cuMemcpyDtoHAsync bytes=%zu\n",get_rank(), getpid(), bytes);
    fflush(stderr);

    return real_fn(dstHost, srcDevice, bytes, stream);
}

extern "C" CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice,const void *srcHost,size_t bytes, CUstream stream)
{
    using fn_t=CUresult(*)(CUdeviceptr, const void *, size_t, CUstream);
    static fn_t real_fn = nullptr;

    if (!real_fn)
        real_fn = (fn_t)dlsym(RTLD_NEXT, "cuMemcpyHtoDAsync");

    fprintf(stderr,"[CUDA_COPY_TRACE] rank=%d pid=%d cuMemcpyHtoDAsync bytes=%zu\n",get_rank(), getpid(), bytes);
    fflush(stderr);

    return real_fn(dstDevice, srcHost, bytes, stream);
}

extern "C" CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t bytes)
{
    using fn_t = CUresult (*)(CUdeviceptr, CUdeviceptr, size_t);
    static fn_t real_fn = nullptr;

    if (!real_fn)
        real_fn = (fn_t)dlsym(RTLD_NEXT, "cuMemcpy");

    fprintf(stderr,"[CUDA_COPY_TRACE] rank=%d pid=%d cuMemcpy bytes=%zu\n",get_rank(), getpid(), bytes);
    fflush(stderr);

    return real_fn(dst, src, bytes);
}

extern "C" CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src,size_t bytes, CUstream stream)
{
    using fn_t = CUresult (*)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
    static fn_t real_fn =nullptr;

    if (!real_fn)
        real_fn = (fn_t)dlsym(RTLD_NEXT, "cuMemcpyAsync");

    fprintf(stderr,"[CUDA_COPY_TRACE] rank=%d pid=%d cuMemcpyAsync bytes=%zu\n",get_rank(), getpid(), bytes);
    fflush(stderr);

    return real_fn(dst, src, bytes, stream);
}
