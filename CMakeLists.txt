cmake_minimum_required(VERSION 3.15)
project(Neighbor-Overlap-Microbenchmark C)

find_package(MPI REQUIRED)

add_library(nom STATIC
    src/nom.c
    src/grid.c
    src/compute.c
)

target_include_directories(nom PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(nom PUBLIC c_std_11)

target_compile_options(nom PRIVATE
    -O3 -Wall -Wextra -Wno-unused-parameter -fPIC
)

target_link_libraries(nom PUBLIC
    MPI::MPI_C
    pthread
    m
)

add_executable(neighbor-overlap-microbenchmark
    src/main/main.c
)

target_link_libraries(neighbor-overlap-microbenchmark PRIVATE nom)

install(TARGETS nom
    ARCHIVE DESTINATION lib
)

install(TARGETS neighbor-overlap-microbenchmark
    RUNTIME DESTINATION bin
)

