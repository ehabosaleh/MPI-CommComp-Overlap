# MPI Communication-Computation Overlap Benchmark (MPI-CCOB)
MPI-CCOB is a multidimensional benchmark for evaluating MPI computation–communication overlap on CPUs and GPUs systems.
measures how much communication can be hidden behind computation when using non-blocking MPI operations (MPI_Isend, MPI_Irecv) in neighbor-exchange patterns (1D, 2D, 3D). The benchmark supports CPU-only and GPU-aware MPI. 

- MPI-CCOB reports several performance metrics, including pure-communication time, concurrent-phase time, requested and measured computation-to-communication ratios, overlap ratio, and average overlap ratio along with its standard deviation.

- For GPU runs, use a GPU-aware MPI (Open MPI built with CUDA support, MVAPICH2-GDR, MPICH with device support, vendor MPI with GPUDirect). The benchmark tries to detect whether GPU buffers are transferred without host staging.
  
- On some MPI implementations (MPICH), asynchronous progress can be toggled via environment variables (e.g. MPIR_CVAR_ASYNC_PROGRESS=1). Try toggling async progress to observe its effect on measured overlap.

## Measurement Methodology

For each message size, every MPI process participates in two measurement phases:

1. **Pure-communication phase:** Measures the completion time of the non-blocking communication without concurrent computation.
2. **Concurrent phase:** Performs the same non-blocking communication while executing the selected CPU or GPU workload.

The computation time is measured separately during the concurrent phase. The overlap ratio is then calculated from:

- $t_{\mathrm{comm}}$: pure-communication time
- $t_{\mathrm{comp}}$: computation time
- $t_{\mathrm{ovlp}}$: concurrent-phase time
- $t_{\min}=\min(t_{\mathrm{comm}},t_{\mathrm{comp}})$
- $t_{\mathrm{hidden}} = t_{\mathrm{comm}} + t_{\mathrm{comp}} - t_{\mathrm{ovlp}}$

$R_{\mathrm{ovlp}} = 100 \times \max\left(0,\min\left(1,\frac{T_{\mathrm{hidden}} }{ \min\left(T_{\mathrm{comm}},T_{\mathrm{comp}}\right)}\right)\right)$

Interpretation:

- **0%:** No effective overlap; communication and computation are effectively serialized.
- **Between 0% and 100%:** Partial overlap.
- **100%:** Maximum achievable overlap; the shorter operation is completely hidden by the longer one.

---

## Build
The repository uses the project-provided autogen script which configures/builds the project (CMake). Typical sequence:

```sh
./autogen.sh 
```

Alternatively, a manual CMake build:

```sh
mkdir -p build && cd build
cmake ..
make -j
```

Running the benchmark
A typical run (example):

```sh
mpirun -np <num-ranks> ./bin/overlapX --dim=2 --ratio=100 --dev=cpu --with-progress=1 --compute-bound=0 --memory-mode=triad
```

Options (common)
- --dim=1|2|3           : neighbor exchange dimensionality (1D/2D/3D)
- --ratio=<percent>     : target computation-to-pure-communication ratio (percent)
- --dev=CPU|GPU         :  target device
- --with-prpgress=0|1   : 0 to disable progress thread, 1 to fork progress thread for each CPU rank. When using GPUs, it enables CPU Polling without forking a thread.
- --progress-thread=0|1 (GPU only): 1 for launching one thread per rank dedicated solely to progress, 0 to use the default CPU Polling if --with-prpgress was enabled.
- --help                : show usage

To run the benchmark with Nsight Systems to capture profiling data during last iteration of overlap on GPU systems, use the following command:

```sh
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 mpirun -np <num_processes> nsys profile --trace=cuda,nvtx,mpi,ucx --capture-range=nvtx  --capture-range-end=stop-shutdown --nvtx-capture=OVERLAP_MEASUREMENT --output='../profiles/profile_%h_rank%q{PMI_RANK}' ./bin/overlapX [options]
```

