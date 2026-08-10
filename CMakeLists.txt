cmake_minimum_required(VERSION 3.17)
cmake_policy(SET CMP0104 NEW)

project(MPI-COMMUNICATION-COMPUTATION-BENCHMARK LANGUAGES C CXX)

option(ENABLE_CUDA "Enable CUDA support" ON)

find_package(MPI REQUIRED)
find_package(Threads REQUIRED)

set(HAVE_CUDA 0)
set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_ROOT}/bin)

set(CCOB_SOURCES src/ccob.c src/compute_cpu.c)

if(ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)

    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        set(HAVE_CUDA 1)

        add_library(cuda_memcpy_trace SHARED src/cuda_memcpy_trace.cpp)
        target_include_directories(cuda_memcpy_trace PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

        target_link_libraries(cuda_memcpy_trace PRIVATE dl)

        set_target_properties(cuda_memcpy_trace PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            LIBRARY_OUTPUT_DIRECTORY ${PROJECT_ROOT}/lib
            RUNTIME_OUTPUT_DIRECTORY ${PROJECT_ROOT}/lib
            ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_ROOT}/lib
        )

        set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++ CACHE FILEPATH "CUDA host compiler")
	    
	
	    #set(CMAKE_CUDA_ARCHITECTURES 80)

	    set(CMAKE_CUDA_ARCHITECTURES 90)
        

        message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
        message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")

	list(APPEND CCOB_SOURCES src/compute_gpu.cu)

        set_source_files_properties(src/ccob.c PROPERTIES LANGUAGE CUDA)
    else()
        message(STATUS "CUDA not found, GPU support disabled")
    endif()
else()
    message(STATUS "GPU support disabled (-DENABLE_CUDA=OFF)")
endif()

add_library(ccob STATIC ${CCOB_SOURCES})

target_include_directories(ccob PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_definitions(ccob PUBLIC HAVE_CUDA=${HAVE_CUDA})

target_link_libraries(ccob PUBLIC
    MPI::MPI_C
    Threads::Threads
    m
)

target_compile_options(ccob PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-O3 -Wall -Wextra -Wno-unused-parameter -fPIC>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-fPIC>
)

if(HAVE_CUDA)
    target_link_libraries(ccob PUBLIC CUDA::cudart)

    set_target_properties(ccob PROPERTIES
        CUDA_STANDARD 11
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
else()
    target_compile_features(ccob PUBLIC c_std_11)
endif()

add_executable(overlapX src/main/main.c )

target_link_libraries(overlapX PRIVATE ccob )
if(HAVE_CUDA)
    set_target_properties(overlapX PROPERTIES
        LINKER_LANGUAGE CUDA
    )
endif()



#install(TARGETS ccob    ARCHIVE DESTINATION lib)

#install(TARGETS test    RUNTIME DESTINATION bin)
