# CUDA kernels journey

In this repository, you can see my progress from my very first matmul, which achieved **355.9 GFLOP/s**, around **5% of cuBLAS**, to my SGEMM kernel, which reaches **4317 GFLOP/s**, around **93.0% of cuBLAS**, and now my latest working Tensor Core kernel, which reaches **21603 GFLOP/s**, around **78.2% of cuBLAS**, on an NVIDIA T4 in Google Colab.

I am a student learning CUDA by writing these kernels from scratch, cuBLAS is only my reference to see how far I still am from NVIDIA's own implementation. The current benchmarks use A `4096x2048` and B `2048x4096` with `alpha = 1` and `beta = 0`, and right now I am working on `beta = 1` and an unfinished `ldmatrix`/`mma.sync` version. My kernels still expect aligned dimensions and the results come from Google Colab, so this is a learning repository and not production code or a perfectly controlled benchmark.

## What is in this repository

- [Vector addition](vector-add/vector_add.cu), where I learned the basic CUDA memory and thread commands.
- [Naïve and tiled matmul](matmul/naive-matmul.cu), where I started learning global memory, shared memory and 2D indexing.
- [Blocktiling kernels](blocktiling/blocktiling2d.cu), where every thread calculates multiple outputs and reuses values through registers.
- [My latest SGEMM comparison](double-buffering/cublas_transposed_double_buffering_2d_blocktiling.cu), using float inputs and output.
- [My latest working Tensor Core comparison](tensor-cores/cuBLAS-2d-tensor-cores.cu), using half inputs with float accumulation and output.
- [My unfinished `ldmatrix` and `mma.sync` version](tensor-cores/cuBLAS-2d-ldmatrixmma-tensor-cores.cu), which is the next Tensor Core implementation I am learning.

The commit history is important to this repository because I kept the broken versions, wrong assumptions and benchmark results while I was learning instead of only uploading the final kernels.

## SGEMM progress

| Version | Main change | Performance |
|---|---|---:|
| Naïve matmul | One result per thread | 355.9 GFLOP/s |
| First 1D tiled matmul | First correct shared-memory version | 195.1 GFLOP/s |
| Padded tile | Reduced shared-memory bank conflicts | 470.0 GFLOP/s |
| Coalesced B loads | Consecutive lanes load consecutive values | 654.1 GFLOP/s |
| 4x4 blocktiling | 16 results per thread | 2381.6 GFLOP/s |
| `float4` loads | Vectorized global-memory loads | 2981.6 GFLOP/s |
| 8x8 blocktiling | Larger matrices and 64 results per thread | 3750 GFLOP/s |
| Double buffering | Overlapped loads with calculation | 3884 GFLOP/s |
| `__launch_bounds__(256, 2)` | Reduced 129 registers to 128 and fitted two blocks | **4317 best / 4248 median** |
| cuBLAS | Same final comparison | **4642 best / 4622 median** |

My first tiled kernel being slower than the naïve one was probably my first important result. Shared memory did not make it fast automatically because I still had bank conflicts and uncoalesced loads, while the naïve version was also benefiting from cache more than I understood at the time.

The biggest improvements came from padding shared memory, coalescing the global loads, giving each thread more results and using registers to reuse values. Near the end `ptxas` reported 129 registers per thread, just one too many for a second block. `__launch_bounds__(256, 2)` reduced this to 128, raised achieved occupancy from around **25% to 49%** and reduced elapsed cycles by **6.47%**. This kernel reported **0 spill stores and 0 spill loads**.

## Tensor Core progress

| Version | Main change | Best | Median |
|---|---|---:|---:|
| First WMMA kernel | First functional Tensor Core code | 342.3 | — |
| Fixed tile loads | Stopped every thread loading the same data 8 times | 818.1 | — |
| One active warp | Stopped all 8 warps repeating the same WMMA work | 1516.2 | — |
| 2D blocktiling | 128x128 block tile and warp tiling | 5743 | 5641 |
| Shared padding | Reduced bank conflicts | 8169 | 7906 |
| Register loads | Kept the next tile out of local memory | 12537 | 12269 |
| Half inputs | Removed the unnecessary float input path | 21184 | 14027 |
| Remapped A loads | Distributed accesses across shared-memory banks | 18796 | 16257 |
| 64x64 warp tile | More work per warp | 17707 | 17413 |
| Removed double buffering | Fitted two blocks instead of one | **21603** | **18030** |
| cuBLAS | Same latest comparison | **27615** | **23059** |

All the values in this table are GFLOP/s. The latest direct comparison puts my WMMA kernel at about **78.2% of cuBLAS** using either the best launches or the medians, and the last recorded relative error was **0**.

One problem I found here was that `store_values_a` and `store_values_b` were going into local memory because the compiler could not scalarize the arrays into registers. Making the array size and loop limit compile-time constants allowed SROA/mem2reg to work, raised the reported register use to 131 and moved the best result from 8169 to 12537 GFLOP/s.

I also learned that more occupancy is not always faster. Reducing `BK` from 32 to 16 raised occupancy from **25% to 49.14%**, but the median dropped to **14376 GFLOP/s**. Later I changed each warp tile from 64x32 to 64x64, but double buffering used **37.89 KB of shared memory per block** and limited the kernel to one block. Removing the double buffer finally allowed two blocks and produced the current 21603 GFLOP/s result.

The latest smaller changes split the shared-memory store loops and reduced reported bank conflicts from **35.3% to 33.9%**, then a new index mapping removed the remaining Nsight Compute bank-conflict warning. I also removed more unnecessary `__syncthreads()`, but these changes have not produced another measurable throughput improvement yet.

I compare the complete result against the CPU or cuBLAS and I added CUDA/cuBLAS error checks, dimension checks, static assertions and Compute Sanitizer checks. The timings only measure the kernel, SGEMM uses float while the Tensor Core version uses half inputs with float accumulation and output, and older results used different matrix sizes and timing loops, so they should be read as my progress instead of as a direct comparison between every kernel.

I used model help for parts of the benchmark and error-checking code and for organizing this README, the kernels and their indexing/debugging are my work.
