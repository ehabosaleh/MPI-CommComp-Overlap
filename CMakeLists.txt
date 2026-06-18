cmake_minimum_required(VERSION 3.17)

project(Neighbor-Overlap-Microbenchmark LANGUAGES C)

option(ENABLE_CUDA "Enable CUDA support" ON)

find_package(MPI REQUIRED)

set(GPU_SOURCES "")
set(GPU_HEADERS "")
set(HAVE_CUDA 0)

if(ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)

    if(CUDAToolkit_FOUND)
        enable_language(CUDA)

        set(GPU_SOURCES
            src/compute_gpu.cu
        )

        set(GPU_HEADERS
            include/compute_gpu.cuh
        )

        set(HAVE_CUDA 1)

        message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
        message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
    else()
        message(STATUS "CUDA not found, GPU support disabled")
    endif()
else()
    message(STATUS "GPU support disabled (-DENABLE_CUDA=OFF)")
endif()

add_library(nmpm STATIC
    src/nmpm.c
    src/compute_cpu.c
    ${GPU_SOURCES}
    ${GPU_HEADERS}
)

target_include_directories(nmpm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(nmpm PUBLIC
    HAVE_CUDA=${HAVE_CUDA}
)

target_compile_features(nmpm PUBLIC
    c_std_11
)

target_compile_options(nmpm PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-O3 -Wall -Wextra -Wno-unused-parameter -fPIC>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-fPIC>
)

target_link_libraries(nmpm PUBLIC
    MPI::MPI_C
    pthread
    m
)

if(HAVE_CUDA)
    target_link_libraries(nmpm PUBLIC
        CUDA::cudart
    )

    set_target_properties(nmpm PROPERTIES
        CUDA_STANDARD 11
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
endif()

add_executable(test
    src/main/main.c
)

target_link_libraries(test PRIVATE
    nmpm
)

install(TARGETS nmpm
    ARCHIVE DESTINATION lib
)

install(TARGETS test
    RUNTIME DESTINATION bin
)
