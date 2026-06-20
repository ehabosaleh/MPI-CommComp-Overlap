# Non-blocking Multi-player Ping-pong Microbenchmark (NMPM) (V2.1.0 — Added GPU communication/computation overlap support)

NMPM quantifies how much communication can be hidden behind computation when using nonblocking MPI operations (`MPI_Isend`, `MPI_Irecv`, `MPI_Waitall`) in a 1D, 2D and 3D neighbor-exchange pattern.

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

## Features
- Addes GPU communication/computation overlap measurement
- 1D neighbor exchange: left, right
- 2D neighbor exchange: left, right, top, bottom
- 3D neighbor exchange: left, right, top, bottom, front, back
- compution to pure communication ratio (--ratio)
- Reports:
  - pure communication time  
  - pure computation time  
  - combined time
  - actual computation to pure communication ratio
  - overlap percentage  
- Works with all major MPI implementations (Open MPI, MPICH, MVAPICH, Intel MPI)
- Detects:
  - whether GPU buffers are transferred directly through GPU-aware MPI
  - RDMA/P2P GPUDirect or NVLink-based data movement without staging through host memory
  - NIC offload capabilities  
  - MPI asynchronous progress behavior

---

## Build nd Install
If directly cloning the git repository use:
```sh
$ ./autogen.sh
$ cd bin
$ mpirun -np numberOfRanks ./executable --dim=1/2/3 --ratio=compToCommRatio
```
## Example output:
when 
```sh
export I_MPI_ASYNC_PROGRESS=1
```

```
Size (Bytes)        Communication(us)   Computation(us)     Actual Ratio %      Requested Ratio %   Overall             Overlapping %
1                   13.883              7.012               50.505              50                  12.231              100.000
2                   13.869              7.002               50.486              50                  12.373              100.000
4                   14.519              7.327               50.466              50                  13.251              100.000
8                   14.532              7.334               50.471              50                  12.944              100.000
16                  14.595              7.365               50.462              50                  12.856              100.000
32                  14.268              7.201               50.470              50                  12.505              100.000
64                  14.399              7.267               50.469              50                  12.638              100.000
128                 14.037              7.087               50.485              50                  12.391              100.000
256                 13.994              7.064               50.476              50                  12.331              100.000
512                 14.408              7.276               50.498              50                  13.871              100.000
1024                15.179              7.659               50.456              50                  14.688              100.000
2048                15.423              7.778               50.433              50                  15.152              100.000
4096                16.275              8.208               50.433              50                  15.591              100.000
8192                17.823              8.986               50.421              50                  16.168              100.000
16384               20.916              10.533              50.356              50                  18.594              100.000
32768               27.743              13.946              50.268              50                  25.885              100.000
65536               41.236              22.802              55.297              50                  39.543              100.000
131072              100.475             51.745              51.501              50                  100.024             100.000
262144              233.361             118.065             50.593              50                  235.374             98.295
524288              919.529             461.071             50.142              50                  922.943             99.259
1048576             1272.526            637.543             50.101              50                  1280.642            98.727

```
when 
```sh
export I_MPI_ASYNC_PROGRESS=0
```
```
Size (Bytes)        Communication(us)   Computation(us)     Actual Ratio %      Requested Ratio %   Overall             Overlapping %
1                   1.747               0.944               53.997              50                  2.504               19.798
2                   1.535               0.835               54.395              50                  2.414               0.000
4                   1.534               0.833               54.296              50                  2.410               0.000
8                   1.538               0.846               55.008              50                  2.419               0.000
16                  1.530               0.820               53.627              50                  2.390               0.000
32                  1.529               0.821               53.670              50                  2.391               0.000
64                  1.561               0.843               53.966              50                  2.435               0.000
128                 1.599               0.862               53.936              50                  2.492               0.000
256                 1.685               0.903               53.589              50                  2.615               0.000
512                 2.251               1.195               53.087              50                  3.414               2.619
1024                2.642               1.386               52.461              50                  3.988               2.935
2048                3.209               1.674               52.164              50                  4.899               0.000
4096                4.327               2.235               51.642              50                  6.479               3.716
8192                6.980               3.563               51.038              50                  10.591              0.000
16384               9.955               5.058               50.806              50                  14.768              4.834
32768               15.834              7.995               50.493              50                  23.577              3.155
65536               28.485              15.337              53.842              50                  43.268              3.616
131072              69.125              35.755              51.725              50                  104.722             0.441
262144              147.513             74.762              50.681              50                  223.962             0.000
524288              301.266             151.198             50.188              50                  445.105             4.867
1048576             613.711             307.809             50.155              50                  911.696             3.191
```

