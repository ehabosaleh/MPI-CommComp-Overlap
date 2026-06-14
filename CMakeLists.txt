cmake_minimum_required(VERSION 3.15)
project(Neighbor-Overlap-Microbenchmark C)

find_package(MPI REQUIRED)

add_library(nmpm STATIC
    src/nmpm.c
    src/compute.c
)

target_include_directories(nmpm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(nmpm PUBLIC c_std_11)

target_compile_options(nmpm PRIVATE
    -O3 -Wall -Wextra -Wno-unused-parameter -fPIC
)

target_link_libraries(nmpm PUBLIC
    MPI::MPI_C
    pthread
    m
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

