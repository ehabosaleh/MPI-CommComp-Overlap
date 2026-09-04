#!/bin/sh

rm -rf build
cmake -S . -B build
cmake --build build
cmake --install build --prefix $PWD

rm -rf build
echo "Build and installation completed successfully."

printf "You can now run the benchmark using the installed executable ./bin/overlapX \nUse the --help option to see available command-line options.\n"
printf "to run the benchmark with Nsight Systems to capture profiling data during last iteration of overlap, use the following command:\n"
printf "NSYS_NVTX_PROFILER_REGISTER_ONLY=0 mpirun -np <num_processes> nsys profile --trace=cuda,nvtx,mpi,ucx --capture-range=nvtx  --capture-range-end=stop-shutdown --nvtx-capture=OVERLAP_MEASUREMENT --output='../profiles/%h_rank%q{OMPI_COMM_WORLD_RANK}' ./bin/overlapX [options]\n"
