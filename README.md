# CUDA kernels journey

This README documents my path from my first vector addition to a handwritten **2774.8 GFLOP/s** (peak) 2D block-tiled matrix multiplication kernel on an NVIDIA T4, with a **2381.6 GFLOP/s average over 20 launches**.

My code is not perfect and I am deliberately keeping my mistakes and intermediate kernels because I want to show the wrong assumptions, indexing problems and the small changes behind the speedups, the peak is only the best launch I saw and not the result I get every time which is why I also include the average and median.

## Complete performance journey so far

| Kernel iteration | What changed | Recorded GFLOP/s |
|---|---|---:|
| Naïve matmul | One thread calculates one result | 355.9 |
| 1D 32x32 tiled | First correct shared-memory tiles | 195.1 |
| Padded 1D tiles | Removed shared-memory bank conflicts | 470.0 |
| Coalesced B loads | Changed how B arrives from global memory | 654.1 |
| 2D 16x16 tiled | Moved to 2D blocks and shared memory | 639.9 |
| 1D blocktiling | Four results per thread | around 400 |
| 2D blocktiling | Average over 20 launches | 2381.6 |
| 2D blocktiling | Best individual launch | **2774.8** |

![My CUDA matmul performance journey so far](assets/full-performance-journey.svg)

## Progress

I started with [`vector_add.cu`](vector-add/vector_add.cu) to understand allocation, copies and how threads map to values, then I wrote a [naïve matmul](matmul/naive-matmul.cu) where my first problems were actually outside the kernel because I allocated B with the wrong size, initialized the wrong pointer and copied A twice, after fixing it the kernel reached **355.9 GFLOP/s**.

My first [tiled version](tiled-matmul/tiled_matmul_1d.cu) used flattened shared memory because I had not learned 2D arrays yet and this made the indexing much harder for me, the first correct version only reached **195.1 GFLOP/s**, then I padded the B tile and later changed how B arrived from global memory so a warp could load consecutive values from a row:

```cpp
__shared__ float shared_memory_2[32*33];

shared_memory_2[columna_carga_2*33 + fila_carga_2] =
    mat2[j*K_col_2*32 + fila_carga_2*K_col_2 + block_col_num_2*32 + columna_carga_2];
```

The original stride of 32 made threads hit the same shared-memory banks, so changing it to 33 was enough to move the kernel from **195.1 to 470 GFLOP/s**, after that I noticed that B was still arriving through strided global loads and remapped the threads so each warp loads consecutive columns from a row, which reached **654.1 GFLOP/s**, it was the first time I could see two small changes related to the hardware make a large difference instead of shared memory just making the kernel faster automatically.

I later made a small [2D indexing exercise](2d-transposed/2d-transposed.cu), then a [2D naïve matmul](matmul/2d-naive-matmul.cu) and finally a [2D tiled version](tiled-matmul/tiled_matmul_2d.cu), using `threadIdx.x` and `threadIdx.y` with real 2D shared-memory tiles was much easier to follow than flattening everything and this version reached around **640 GFLOP/s**, but I also found that one version had `__syncthreads()` inside a bounds check which my dimensions were hiding and an edge block would hang.

My first [blocktiling attempt](blocktiling/blocktiling1d.cu) made each thread calculate four outputs in one direction, this was where I first tried to do more work per thread but the mapping is honestly overcomplicated and I had trouble following variables like `threadIdx_r4_1` and `threadIdxy4_1` while debugging it, when it finally worked it was also slower at around **400 GFLOP/s** which showed me that giving a thread more outputs was not automatically an optimization, after this file I started paying more attention to names, comments and removing dead code because I did not want the next kernel to become just as difficult for me to read.

## Current kernel

In [`blocktiling2d.cu`](blocktiling/blocktiling2d.cu) every thread calculates a 4x4 section of C and keeps the results and the values being reused in these arrays:

```cpp
float sum[TM][TN] = {0.0f};
float regA[TM];
float regB[TN];
```

Inside the compute loop those values form the small outer product with:

```cpp
sum[j][t] += regA[j]*regB[t];
```

For every position in K a thread brings four values from A and four from B into those arrays and uses them to update 16 results, so the values read from shared memory do more work before the next ones are loaded, this reuse is the main difference from my previous version where a thread only owned results in one direction.

For the tile sizes I was comparing a smaller 32x32 block tile with the 64x64 one that is now in the file, both could work but the smaller version only needed 64 threads while the current one uses 256 and also reuses the loaded tiles more, I kept K in chunks of 32 and made each thread own 4x4 results because it looked like a reasonable balance before knowing the real register count, the `BM`, `BN`, `BK`, `TM` and `TN` names were also part of trying to make this version easier to understand than my previous blocktiling code.

I was thinking about registers, shared memory and having enough threads when I chose the dimensions but most of it was still an estimate while writing the code, once it compiled I checked what the compiler had actually done and `ptxas` reported this:

```text
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
Used 64 registers, used 1 barriers, 16384 bytes smem
```

It used more registers than I expected but nothing spilled to local memory, with the block size in the file this still fits four resident blocks on the T4 although it lands just at the register limit, so this was useful because the estimate from only counting the arrays did not include everything else the compiler needed.

The first commit of this kernel was barely a sketch with missing types, wrong global addresses and even `dim3 threads[16][16]` instead of `dim3 threads(16,16)`, I also made every block repeat work that the grid was already doing and the following commits are where I slowly removed those mistakes until the complete output matched the CPU.

This is the output I recorded for `2048x1024 * 1024x2048`:

```text
GPU:      3.607 ms  (2381.6 GFLOP/s, 20-launch average)
min:      3.096 ms  (2774.8 GFLOP/s)
mediana:  3.612 ms  (2378.3 GFLOP/s)
max:      3.617 ms  (2375.0 GFLOP/s)
Max abs diff: 0.000106812 (at index 3831606)
```

## Benchmark notes

The numbers were recorded in Google Colab on an NVIDIA T4 and only measure the kernel, older kernels used a different matrix shape so the last bar is not a completely controlled comparison, and the current kernel only handles the aligned dimensions used here and has not been compared with cuBLAS yet.

I used model help for parts of the benchmark code and for organizing parts of this README, the kernels and their indexing/debugging are my work, and the commit history has much more detail including the versions where the kernels were wrong.
