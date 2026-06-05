#!/bin/sh

rm -rf build
cmake -S . -B build
cmake --build build
cmake --install build --prefix $PWD

rm -rf build
echo;
echo "microbenchmark is compiled on $PWD"
echo;
echo "To run the microbenchmakr use mpirun -np X ./executable_file"
