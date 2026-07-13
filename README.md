# Non-blocking Multi-player Ping-pong Microbenchmark (NMPM)

NMPM measures how much communication can be hidden behind computation when using nonblocking MPI operations (MPI_Isend, MPI_Irecv, MPI_Waitall) in neighbor-exchange patterns (1D, 2D, 3D). The benchmark supports CPU-only and GPU-aware MPI builds and includes instrumentation to detect whether GPU communication is performed without host staging (GPUDirect / NVLink) and to evaluate MPI asynchronous progress behavior.

## Key goals:
- Quantify communication/computation overlap in realistic neighbor-exchange kernels.
- Compare pure communication, pure computation and combined run-times and compute an overlap metric.
- Support GPU-enabled runs and detect GPU-aware MPI capabilities.


## For each process, the benchmark runs three phases:

1. **Pure communication**  
2. **Pure computation**  
3. **Communication + computation combined**

Using these timings, the benchmark computes the overlap ratio:

$$
\text{overlap} = 100 \times 
\frac{\max\left(0,\; \min\left(t_{\text{pure}} + t_{\text{comp}} - t_{\text{ovrl}},\; t_{\min}\right)\right)}
     {t_{\min}},
\quad
t_{\min} = \min\left(t_{\text{pure}},\; t_{\text{comp}}\right)
$$

Interpretation:

- **0%** → no overlap (communication and computation are serialized)  
- **100%** → full overlap (communication fully hidden behind compute)

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


