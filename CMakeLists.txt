cmake_minimum_required(VERSION 3.17)

project(Neighbor-Overlap-Microbenchmark LANGUAGES C)

option(ENABLE_CUDA "Enable CUDA support" ON)

find_package(MPI REQUIRED)
find_package(Threads REQUIRED)

set(HAVE_CUDA 0)

set(NMPM_SOURCES
    src/nmpm.c
    src/compute_cpu.c
)

set(TEST_SOURCES
    src/main/main.c
)

if(ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)

    if(CUDAToolkit_FOUND)
        enable_language(CUDA)

        set(HAVE_CUDA 1)

        message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
        message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")

        list(APPEND NMPM_SOURCES
            src/compute_gpu.cu
        )

        # nmpm.c contains CUDA code under #if HAVE_CUDA,
        # so compile it with NVCC when CUDA is enabled.
        set_source_files_properties(
            src/nmpm.c
            PROPERTIES LANGUAGE CUDA
        )
    else()
        message(STATUS "CUDA not found, GPU support disabled")
    endif()
else()
    message(STATUS "GPU support disabled (-DENABLE_CUDA=OFF)")
endif()

add_library(nmpm STATIC
    ${NMPM_SOURCES}
)

target_include_directories(nmpm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(nmpm PUBLIC
    HAVE_CUDA=${HAVE_CUDA}
)

target_link_libraries(nmpm PUBLIC
    MPI::MPI_C
    Threads::Threads
    m
)

target_compile_options(nmpm PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-O3 -Wall -Wextra -Wno-unused-parameter -fPIC>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-fPIC>
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
else()
    target_compile_features(nmpm PUBLIC
        c_std_11
    )
endif()

add_executable(test
    ${TEST_SOURCES}
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
