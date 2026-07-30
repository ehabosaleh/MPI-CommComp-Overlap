# MPI Communication-Computation Overlap Benchmark (MPI-CCOB): A Multidimensional Benchmark for Evaluating MPI Computation–Communication Overlap on CPUs and GPUs Systems

MPI-CCOB measures how much communication can be hidden behind computation when using non-blocking MPI operations (MPI_Isend, MPI_Irecv) in neighbor-exchange patterns (1D, 2D, 3D). The benchmark supports CPU-only and GPU-aware MPI. 

MPI-CCOB reports several performance metrics, including pure-communication time, concurrent-phase time, requested and measured computation-to-communication ratios, overlap ratio, and average overlap ratio along with its standard deviation.


## Measurement Methodology

For each message size, every MPI process participates in two measurement phases:

1. **Pure-communication phase:** Measures the completion time of the non-blocking communication without concurrent computation.
2. **Concurrent phase:** Performs the same non-blocking communication while executing the selected CPU or GPU workload.

The computation time is measured separately during the concurrent phase. The overlap ratio is then calculated from:

- \(t_{\mathrm{comm}}\): pure-communication time
- \(t_{\mathrm{comp}}\): computation time
- \(t_{\mathrm{ovlp}}\): concurrent-phase time
- \(t_{\min}=\min(t_{\mathrm{comm}},t_{\mathrm{comp}})\)

$$
R_{\mathrm{ovlp}}
=
100 \times
\frac{
\max\left(
0,\;
\min\left(
t_{\mathrm{comm}} + t_{\mathrm{comp}} - t_{\mathrm{ovlp}},
t_{\min}
\right)
\right)
}{
t_{\min}
}.
$$

Interpretation:

- **0%:** No effective overlap; communication and computation are effectively serialized.
- **Between 0% and 100%:** Partial overlap.
- **100%:** Maximum achievable overlap; the shorter operation is completely hidden by the longer one.

---


## Features
- Measure overlap for 1D, 2D and 3D neighbor exchanges
- CPU and GPU buffer support (detects GPU-aware MPI / GPUDirect)
- Reports per-message-size timings: pure communication, pure computation, combined, actual ratio, overlap percentage
- Works with major MPI implementations (Open MPI, MPICH, MVAPICH, Intel MPI)
- Configurable computation-to-communication ratio



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
mpirun -np <num-ranks> ./bin/nmpm --dim=1 --ratio=50 --dev=1 --with-prpgress=1 --progress-thread=1 --compute-bound=0 --memory-mode=triad
```

Options (common)
- --dim=1|2|3           : neighbor exchange dimensionality (1D/2D/3D)
- --ratio=<percent>     : target computation-to-pure-communication ratio (percent)
- --dev=0|1             : 0 for CPU, 1for GPU
- --with-prpgress=0|1   : 0 to disable manual progress, 1 to enable it.
- --progress-thread=0|1 : 0 for the asynchronous call of the MPI_Test routine (GPU only), 1 for launching one thread per rank dedicated solely to progress (CPU or GPU)
- --help              : show usage

## GPU considerations
- For GPU runs, use a GPU-aware MPI (Open MPI built with CUDA support, MVAPICH2-GDR, MPICH with device support, vendor MPI with GPUDirect). The benchmark tries to detect whether GPU buffers are transferred without host staging.
- On some MPI implementations (MPICH), asynchronous progress can be toggled via environment variables (e.g. MPIR_CVAR_ASYNC_PROGRESS=1). Try toggling async progress to observe its effect on measured overlap.

## Interpreting output
- The benchmark prints a table of message sizes and timings: pure communication, pure computation, combined, actual ratio, requested ratio and computed overlap.
- Compare runs with asynchronous progress enabled vs disabled to see how much overlap is achieved in practice.


