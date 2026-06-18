cmake_minimum_required(VERSION 3.15)
project(Neighbor-Overlap-Microbenchmark C)

find_package(MPI REQUIRED)

if(NOT DEFINED ENABLE_CUDA)
    set(ENABLE_CUDA ON)
endif()

if(ENABLE_CUDA)
    enable_language(CXX CUDA)
    find_package(CUDA)
    
    if(CUDA_FOUND)
        set(GPU_SOURCES
            src/compute_gpu.cu
        )
        set(GPU_HEADERS
            include/compute_gpu.cuh
        )
        message(STATUS "CUDA found: ${CUDA_VERSION_STRING}")
        set(HAVE_GPU 1)
        add_compile_definitions(HAVE_GPU=1)
    else()
        message(STATUS "CUDA not found, GPU support disabled (use -DENABLE_CUDA=OFF to suppress this message)")
        set(GPU_SOURCES "")
        set(GPU_HEADERS "")
        set(HAVE_GPU 0)
    endif()
else()
    message(STATUS "GPU support disabled (-DENABLE_CUDA=OFF)")
    set(GPU_SOURCES "")
    set(GPU_HEADERS "")
    set(HAVE_GPU 0)
endif()

add_library(nmpm STATIC
    src/nmpm.c
    src/compute_cpu.c
    ${GPU_SOURCES}
)

target_include_directories(nmpm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

target_compile_features(nmpm PUBLIC c_std_11)

target_compile_options(nmpm PRIVATE
    -O3 -Wall -Wextra -Wno-unused-parameter -fPIC
)

if(CUDA_FOUND)
    set_target_properties(nmpm PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 11
    )
    target_compile_options(nmpm PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options -fPIC>
    )
endif()

target_link_libraries(nmpm PUBLIC
    MPI::MPI_C
    pthread
    m
    ${CUDA_LIBRARIES}
)

add_executable(test
    src/main/main.c
)

target_link_libraries(test PRIVATE nmpm)

install(TARGETS nmpm
    ARCHIVE DESTINATION lib
)

install(TARGETS test
    RUNTIME DESTINATION bin
)

