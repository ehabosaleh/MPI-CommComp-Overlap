#define __GNU_SOURCE

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

static constexpr unsigned long DEFAULT_TRACE_LIMIT = 200;

static std::atomic<unsigned long> trace_counter{0};


static int get_rank(){
	const char *value = nullptr;

    value = std::getenv("PMI_RANK");
    if (value != nullptr)
        return std::atoi(value);

    value = std::getenv("PMIX_RANK");
    if (value != nullptr)
        return std::atoi(value);

    value = std::getenv("OMPI_COMM_WORLD_RANK");
    if (value != nullptr)
        return std::atoi(value);

    value = std::getenv("SLURM_PROCID");
    if (value != nullptr)
        return std::atoi(value);

    value = std::getenv("MPI_LOCALRANKID");
    if (value != nullptr)
        return std::atoi(value);

    value = std::getenv("MV2_COMM_WORLD_RANK");
    if (value != nullptr)
        return std::atoi(value);

    return -1;
}

/* -------------------------------------------------------------------------- */
/* Trace configuration                                                         */
/* -------------------------------------------------------------------------- */

static unsigned long get_trace_limit()
{
    static const unsigned long limit = []() {
        const char *value = std::getenv("CUDA_COPY_TRACE_LIMIT");

        if (value == nullptr)
            return DEFAULT_TRACE_LIMIT;

        char *end = nullptr;

        const unsigned long parsed =
            std::strtoul(value, &end, 10);

        if (end == value || *end != '\0')
            return DEFAULT_TRACE_LIMIT;

        return parsed;
    }();

    return limit;
}

static bool should_trace()
{
    const unsigned long limit = get_trace_limit();

    /*
     * A limit of zero means unlimited tracing.
     */
    if (limit == 0)
        return true;

    const unsigned long index =
        trace_counter.fetch_add(1, std::memory_order_relaxed);

    return index < limit;
}

static const char *kind_string(cudaMemcpyKind kind)
{
    switch (kind) {
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

/* -------------------------------------------------------------------------- */
/* Logging                                                                     */
/* -------------------------------------------------------------------------- */

static void trace_runtime_copy(
    const char *function_name,
    cudaMemcpyKind kind,
    size_t bytes)
{
    if (!should_trace())
        return;

    char buffer[256];

    const int length = std::snprintf(
        buffer,
        sizeof(buffer),
        "[CUDA_COPY_TRACE] rank=%d pid=%ld "
        "%s kind=%s bytes=%zu\n",
        get_rank(),
        static_cast<long>(getpid()),
        function_name,
        kind_string(kind),
        bytes);

    if (length <= 0)
        return;

    size_t output_length = static_cast<size_t>(length);

    if (output_length >= sizeof(buffer))
        output_length = sizeof(buffer) - 1;

    (void)::write(STDERR_FILENO, buffer, output_length);
}

static void trace_driver_copy(
    const char *function_name,
    size_t bytes)
{
    if (!should_trace())
        return;

    char buffer[256];

    const int length = std::snprintf(
        buffer,
        sizeof(buffer),
        "[CUDA_COPY_TRACE] rank=%d pid=%ld "
        "%s bytes=%zu\n",
        get_rank(),
        static_cast<long>(getpid()),
        function_name,
        bytes);

    if (length <= 0)
        return;

    size_t output_length = static_cast<size_t>(length);

    if (output_length >= sizeof(buffer))
        output_length = sizeof(buffer) - 1;

    (void)::write(STDERR_FILENO, buffer, output_length);
}

static void report_resolution_error(
    const char *library_name,
    const char *symbol_name,
    const char *error_message)
{
    char buffer[512];

    const int length = std::snprintf(
        buffer,
        sizeof(buffer),
        "[CUDA_COPY_TRACE] rank=%d pid=%ld "
        "cannot resolve %s from %s: %s\n",
        get_rank(),
        static_cast<long>(getpid()),
        symbol_name,
        library_name,
        error_message != nullptr
            ? error_message
            : "symbol not found");

    if (length <= 0)
        return;

    size_t output_length = static_cast<size_t>(length);

    if (output_length >= sizeof(buffer))
        output_length = sizeof(buffer) - 1;

    (void)::write(STDERR_FILENO, buffer, output_length);
}

/* -------------------------------------------------------------------------- */
/* CUDA library handles                                                        */
/* -------------------------------------------------------------------------- */

static void *get_libcuda_handle()
{
    static void *handle = []() -> void * {
        dlerror();

        void *result =
            dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);

        const char *error = dlerror();

        if (result == nullptr) {
            report_resolution_error(
                "libcuda.so.1",
                "<library>",
                error);
        }

        return result;
    }();

    return handle;
}

static void *get_libcudart_handle()
{
    static void *handle = []() -> void * {
        const char *candidates[] = {
            "libcudart.so",
            "libcudart.so.13",
            "libcudart.so.12",
            "libcudart.so.11.0"
        };

        for (const char *library_name : candidates) {
            dlerror();

            void *result =
                dlopen(library_name, RTLD_NOW | RTLD_LOCAL);

            if (result != nullptr)
                return result;
        }

        report_resolution_error(
            "libcudart",
            "<library>",
            dlerror());

        return nullptr;
    }();

    return handle;
}

/* -------------------------------------------------------------------------- */
/* Symbol resolution                                                           */
/* -------------------------------------------------------------------------- */

template <typename FunctionType>
static FunctionType resolve_driver_symbol(
    const char *symbol_name)
{
    void *handle = get_libcuda_handle();

    if (handle == nullptr)
        return nullptr;

    dlerror();

    void *symbol = dlsym(handle, symbol_name);

    const char *error = dlerror();

    if (error != nullptr || symbol == nullptr) {
        report_resolution_error(
            "libcuda.so.1",
            symbol_name,
            error);

        return nullptr;
    }

    return reinterpret_cast<FunctionType>(symbol);
}

template <typename FunctionType>
static FunctionType resolve_runtime_symbol(
    const char *symbol_name)
{
    void *handle = get_libcudart_handle();

    if (handle == nullptr)
        return nullptr;

    dlerror();

    void *symbol = dlsym(handle, symbol_name);

    const char *error = dlerror();

    if (error != nullptr || symbol == nullptr) {
        report_resolution_error(
            "libcudart",
            symbol_name,
            error);

        return nullptr;
    }

    return reinterpret_cast<FunctionType>(symbol);
}

/* ========================================================================== */
/* CUDA Runtime API wrappers                                                   */
/* ========================================================================== */

extern "C"
cudaError_t cudaMemcpy(
    void *dst,
    const void *src,
    size_t count,
    cudaMemcpyKind kind)
{
    using function_t = cudaError_t (*)(
        void *,
        const void *,
        size_t,
        cudaMemcpyKind);

    static function_t real_function =
        resolve_runtime_symbol<function_t>("cudaMemcpy");

    if (real_function == nullptr)
        return cudaErrorSharedObjectSymbolNotFound;

    static thread_local bool inside_wrapper = false;

    if (inside_wrapper)
        return real_function(dst, src, count, kind);

    inside_wrapper = true;

    trace_runtime_copy(
        "cudaMemcpy",
        kind,
        count);

    const cudaError_t result =
        real_function(dst, src, count, kind);

    inside_wrapper = false;

    return result;
}

extern "C"
cudaError_t cudaMemcpyAsync(
    void *dst,
    const void *src,
    size_t count,
    cudaMemcpyKind kind,
    cudaStream_t stream)
{
    using function_t = cudaError_t (*)(
        void *,
        const void *,
        size_t,
        cudaMemcpyKind,
        cudaStream_t);

    static function_t real_function =
        resolve_runtime_symbol<function_t>(
            "cudaMemcpyAsync");

    if (real_function == nullptr)
        return cudaErrorSharedObjectSymbolNotFound;

    static thread_local bool inside_wrapper = false;

    if (inside_wrapper) {
        return real_function(
            dst,
            src,
            count,
            kind,
            stream);
    }

    inside_wrapper = true;

    trace_runtime_copy(
        "cudaMemcpyAsync",
        kind,
        count);

    const cudaError_t result =
        real_function(
            dst,
            src,
            count,
            kind,
            stream);

    inside_wrapper = false;

    return result;
}

/* ========================================================================== */
/* CUDA Driver API wrappers                                                    */
/* ========================================================================== */

/*
 * Explicit _v2 names are used because cuda.h normally maps the public
 * cuMemcpyDtoH/cuMemcpyHtoD names to their versioned ABI symbols.
 */

extern "C"
CUresult cuMemcpyDtoH_v2(
    void *dst_host,
    CUdeviceptr src_device,
    size_t bytes)
{
    using function_t = CUresult (*)(
        void *,
        CUdeviceptr,
        size_t);

    static function_t real_function =
        resolve_driver_symbol<function_t>(
            "cuMemcpyDtoH_v2");

    if (real_function == nullptr)
        return CUDA_ERROR_NOT_FOUND;

    static thread_local bool inside_wrapper = false;

    if (inside_wrapper)
        return real_function(dst_host, src_device, bytes);

    inside_wrapper = true;

    trace_driver_copy(
        "cuMemcpyDtoH_v2",
        bytes);

    const CUresult result =
        real_function(dst_host, src_device, bytes);

    inside_wrapper = false;

    return result;
}

extern "C"
CUresult cuMemcpyHtoD_v2(
    CUdeviceptr dst_device,
    const void *src_host,
    size_t bytes)
{
    using function_t = CUresult (*)(
        CUdeviceptr,
        const void *,
        size_t);

    static function_t real_function =
        resolve_driver_symbol<function_t>(
            "cuMemcpyHtoD_v2");

    if (real_function == nullptr)
        return CUDA_ERROR_NOT_FOUND;

    static thread_local bool inside_wrapper = false;

    if (inside_wrapper)
        return real_function(dst_device, src_host, bytes);

    inside_wrapper = true;

    trace_driver_copy(
        "cuMemcpyHtoD_v2",
        bytes);

    const CUresult result =
        real_function(dst_device, src_host, bytes);

    inside_wrapper = false;

    return result;
}

extern "C"
CUresult cuMemcpyDtoHAsync_v2(
    void *dst_host,
    CUdeviceptr src_device,
    size_t bytes,
    CUstream stream)
{
    using function_t = CUresult (*)(
        void *,
        CUdeviceptr,
        size_t,
        CUstream);

    static function_t real_function =
        resolve_driver_symbol<function_t>(
            "cuMemcpyDtoHAsync_v2");

    if (real_function == nullptr)
        return CUDA_ERROR_NOT_FOUND;

    static thread_local bool inside_wrapper = false;

    if (inside_wrapper) {
        return real_function(
            dst_host,
            src_device,
            bytes,
            stream);
    }

    inside_wrapper = true;

    trace_driver_copy(
        "cuMemcpyDtoHAsync_v2",
        bytes);

    const CUresult result =
        real_function(
            dst_host,
            src_device,
            bytes,
            stream);

    inside_wrapper = false;

    return result;
}

extern "C"
CUresult cuMemcpyHtoDAsync_v2(
    CUdeviceptr dst_device,
    const void *src_host,
    size_t bytes,
    CUstream stream)
{
    using function_t = CUresult (*)(
        CUdeviceptr,
        const void *,
        size_t,
        CUstream);

    static function_t real_function =
        resolve_driver_symbol<function_t>(
            "cuMemcpyHtoDAsync_v2");

    if (real_function == nullptr)
        return CUDA_ERROR_NOT_FOUND;

    static thread_local bool inside_wrapper = false;

    if (inside_wrapper) {
        return real_function(
            dst_device,
            src_host,
            bytes,
            stream);
    }

    inside_wrapper = true;

    trace_driver_copy(
        "cuMemcpyHtoDAsync_v2",
        bytes);

    const CUresult result =
        real_function(
            dst_device,
            src_host,
            bytes,
            stream);

    inside_wrapper = false;

    return result;
}

/*
 * UCX commonly calls these generic unified-addressing Driver API functions
 * for GPU copies.
 */

extern "C"
CUresult cuMemcpy(
    CUdeviceptr dst,
    CUdeviceptr src,
    size_t bytes)
{
    using function_t = CUresult (*)(
        CUdeviceptr,
        CUdeviceptr,
        size_t);

    static function_t real_function =
        resolve_driver_symbol<function_t>("cuMemcpy");

    if (real_function == nullptr)
        return CUDA_ERROR_NOT_FOUND;

    static thread_local bool inside_wrapper = false;

    if (inside_wrapper)
        return real_function(dst, src, bytes);

    inside_wrapper = true;

    trace_driver_copy("cuMemcpy", bytes);

    const CUresult result =
        real_function(dst, src, bytes);

    inside_wrapper = false;

    return result;
}

extern "C"
CUresult cuMemcpyAsync(
    CUdeviceptr dst,
    CUdeviceptr src,
    size_t bytes,
    CUstream stream)
{
    using function_t = CUresult (*)(
        CUdeviceptr,
        CUdeviceptr,
        size_t,
        CUstream);

    static function_t real_function =
        resolve_driver_symbol<function_t>(
            "cuMemcpyAsync");

    if (real_function == nullptr)
        return CUDA_ERROR_NOT_FOUND;

    static thread_local bool inside_wrapper = false;

    if (inside_wrapper)
        return real_function(dst, src, bytes, stream);

    inside_wrapper = true;

    trace_driver_copy(
        "cuMemcpyAsync",
        bytes);

    const CUresult result =
        real_function(dst, src, bytes, stream);

    inside_wrapper = false;

    return result;
}
