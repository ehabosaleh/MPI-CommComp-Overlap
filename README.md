# MPI Microbenchmarks

This repository contains a collection of MPI microbenchmarks designed to evaluate message-passing performance, including communication latency, bandwidth, neighbor exchange patterns, and communication–computation overlap.

The primary benchmark included is the **Non-blocking Multi-player Ping-pong Microbenchmark (NMPM)**, which measures the ability of an MPI stack and network hardware to overlap nonblocking communication with computation.

---

## Non-blocking Multi-player Ping-pong Microbenchmark (NMPM)

NOB quantifies how much communication can be hidden behind computation when using nonblocking MPI operations (`MPI_Isend`, `MPI_Irecv`, `MPI_Waitall`) in a 2D neighbor-exchange pattern.

The benchmark runs three phases:

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

### Features

- 2D Cartesian neighbor exchange: left, right, top, bottom
- Nonblocking communication: `MPI_Isend`, `MPI_Irecv`
- Flexible message sizes (1 MiB → 256 MiB)
- Calibrated compute workload to match communication time
- Reports:
  - pure communication time  
  - pure computation time  
  - combined time  
  - overlap percentage  
- Works with all major MPI implementations (Open MPI, MPICH, MVAPICH, Intel MPI)
- Detects:
  - NIC offload capabilities  
  - MPI asynchronous progress behavior  
  - UCX/OFI progress engines  
---

### Build nd Install
If directly cloning the git repository use:
```sh
$ ./autogen.sh
$ cd examples/
$ mpirun [ Runtime options] ./executable
```
### Example output:
```
Size (Bytes)       Communication(us)  Computation(us)   Overall(us)        Overlapping %
65536               12.653              13.892              26.758              3.544
131072              18.492              19.534              37.685              1.865
262144              30.030              31.562              60.026              5.232
524288              68.748              70.046              138.584             0.961
1048576             141.297             143.109             277.034             5.218
2097152             267.361             269.240             538.615             0.140
4194304             617.473             618.236             1165.087            11.309
8388608             1363.551            1362.518            2776.560            0.000
16777216            3885.356            3878.962            7623.089            3.640

```



