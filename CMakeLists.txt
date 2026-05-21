cmake_minimum_required(VERSION 3.5)
project(NMPM 
	VERSION 1.0.0 
	LANGUAGES C CXX Fortran
	)

find_package(MPI REQUIRED)

add_library(nom STATIC
    src/nmpm.c
    src/grid.c
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

add_executable(NMPM
    src/main/main.c
)

target_link_libraries(NMPM PRIVATE nmpm)

install(TARGETS nmpm
    ARCHIVE DESTINATION lib
)

install(TARGETS NMPM
    RUNTIME DESTINATION bin
)

