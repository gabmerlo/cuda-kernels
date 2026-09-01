# CUDA kernels journey

In this repository, you can see my progress from my very first matmul, which achieved **355.9 GFLOP/s** (**5% of cuBLAS performance**), to my SGEMM kernel, which reaches **4317 GFLOP/s**, **92.9% of cuBLAS**, and now my latest tensor core kernel, which reaches **18796 GFLOP/s**, **64.7% of cuBLAS**, on an NVIDIA Tesla T4 in Google Colab.

cuBLAS is NVIDIA's own optimized library for matrix multiplication, so I use it to see how close my kernels get to NVIDIA's own implementation on the same operation as me.

But neither my SGEMM nor my tensor core kernel with `alpha = 1` and `beta = 0` on A `4096x2048` and B `2048x4096` are intended to be my best kernels, as my only objective with this repository is to learn, so what's most likely is that I will be adding more kernels in the future.

For that reason, the code I show here is not perfect, you can see this both in the current code, which I am still trying to improve, and especially throughout my commit history, where I gradually improved my SGEMM **from 5% of cuBLAS to 92.9% of cuBLAS**, and later my tensor core kernel from **342.3 to 18796 GFLOP/s**, correcting many, many, mistakes, as I learned new concepts and implemented stuff by hand.

One disclaimer that I want to make is that my measurements may not always be perfect, since until now I have worked mostly in Google Colab, and I have also changed both the matrix sizes and the benchmarking loops over time, however, I have tried to document all of those changes as thoroughly as possible along the way.

## My SGEMM performance over time

| Kernel iteration | Changes | The GFLOP/s |
|---|---|---:|
| Naïve matmul | Each thread calculated just one result | 355.9 GFLOP/s |
| 1D 32x32 tiled | My first ever correct shared-memory tiles | 195.1 GFLOP/s (very disappointing) |
| 1D 32x32 tiled (Padded) | I padded the shared memory to reduce bank conflicts | 470.0 GFLOP/s (progress) |
| 1D 32x32 tiled (B Coalesced) | Changed how B arrives from global memory | 654.1 GFLOP/s (more progress) |
| 2D 16x16 tiled | Moved to 2D blocks and shared memory | 639.9 GFLOP/s (almost equal to 1D 32x32)|
| 1D blocktiling 4x1 | I reused a column for 4 rows, four results per thread | 402.5 GFLOP/s (not the best)|
| 2D blocktiling 4x4 | 16 results per thread, better readability, avoided bank conflicts | 2381.6 GFLOP/s|
| 2D blocktiling 4x4 using `float4`| Vectorized global loads | 2981.6 GFLOP/s |
| 2D blocktiling 8x8 | Even more work per thread (64 results) but initially slower than 4x4 | 2575 GFLOP/s |
| A `4096x2048` B `2048x4096` and `float4` C stores | dimensions change, vectorized output stores | 3750 GFLOP/s |
| Transposed + Double buffering | Two alternating shared-memory tiles | 3884 GFLOP/s |
| Remapped shared B reads | Bank-conflict fix, older timing loop | 3946* GFLOP/s |
| `__launch_bounds__(256, 2)` | Two resident blocks now fit instead of 1 | **4248** GFLOP/s |
| cuBLAS reference | Same comparison, median | **4622** GFLOP/s |


![SGEMM performance so far](assets/full-performance-journey.svg)

## My Progress

I started with [`vector_add.cu`](vector-add/vector_add.cu) to understand allocation, how threads map to values, and to learn the most basic cuda commands, there is not much to remark about that kernel.

After that, I wrote a [naïve matmul](matmul/naive-matmul.cu), as my first introduction to matmul, I chose matmul after vector addition since I wanted to write a kernel where the GPU implementation could beat my CPU by a wide margin, and so I chose matmul. When I wrote it, I was making many rookie mistakes, like allocating B with the wrong size, initializing pointers wrongly, or copying A twice, but after fixing everything my kernel reached **355.9 GFLOP/s**.

The problem my naïve matmul had, was that for every C[i], I had to read an entire row from A and an entire column for B, from global memory, and each access can take up to a couple hundred cycles of waiting (even tho my cache was kinda saving me as we will see).
So the solution to this problem is using the shared memory, to which an access, without any bank conflicts can take just around 20 to 22 cycles. So I had to find a way to load info from global into shared, and then to reuse that information in shared as much as I could, and that way I would try to minimize my accesses to global.

The solution to the problem I just described was my first [tiled matmul](tiled-matmul/tiled_matmul_1d.cu), where I ended up using a flattened shared memory (1D) and overcomplicating my indexes to the point debugging actually took time because I had not learned 2D arrays yet.

But something surprising happened, since theoretically what I was doing should have achieved a better result than my [naïve matmul](matmul/naive-matmul.cu), but my first ever correct output tiled matmul version only reached **195.1 GFLOP/s**, almost 2 times slower than my naïve matmul, which did not use tiling.

There were 2 main reasons for it being slower than my naïve matmul, the first one, is that unknowingly at the moment, my naïve matmul was using the cache to actually not have to make that many calls to the global memory, and the second reason was that my implementation was suffering from bank conflicts, and it was also not coalesced.  then I padded the B tile and later changed how B arrived from global memory so a warp could load consecutive values from a row:

```cpp
__shared__ float shared_memory_2[32*33];

shared_memory_2[columna_carga_2*33 + fila_carga_2] =
    mat2[j*K_col_2*32 + fila_carga_2*K_col_2 + block_col_num_2*32 + columna_carga_2];
```

The original stride of 32 made threads hit the same shared-memory banks, so changing it to 33 was enough to move the kernel from **195.1 to 470 GFLOP/s**, after that I noticed that B was still arriving through strided global loads and remapped the threads so each warp loads consecutive columns from a row, which reached **654.1 GFLOP/s**, it was the first time I could see two small changes related to the hardware make a large difference instead of shared memory just making the kernel faster automatically.

I later made a small [2D indexing exercise](2d-transposed/2d-transposed.cu), then a [2D naïve matmul](matmul/2d-naive-matmul.cu) and finally a [2D tiled version](tiled-matmul/tiled_matmul_2d.cu), using `threadIdx.x` and `threadIdx.y` with real 2D shared-memory tiles was much easier to follow than flattening everything and this version reached around **640 GFLOP/s**, but I also found that one version had `__syncthreads()` inside a bounds check which my dimensions were hiding and an edge block would hang.

My first [blocktiling attempt](blocktiling/blocktiling1d.cu) made each thread calculate four outputs in one direction, this was where I first tried to do more work per thread but the mapping is honestly overcomplicated and I had trouble following variables like `threadIdx_r4_1` and `threadIdxy4_1` while debugging it, when it finally worked it was also slower at around **400 GFLOP/s** which showed me that giving a thread more outputs was not automatically an optimization, after this file I started paying more attention to names, comments and removing dead code because I did not want the next kernel to become just as difficult for me to read.

## First 2D blocktiling kernel

In [`blocktiling2d.cu`](blocktiling/blocktiling2d.cu) every thread calculates a 4x4 section of C and keeps the results and the values being reused in these arrays:

```cpp
float sum[TM][TN] = {0.0f};
float regA[TM];
float regB[TN];
```

For every position in K a thread brings four values from A and four from B into those arrays and uses them to update 16 results, so the values read from shared memory do more work before the next ones are loaded, this reuse was the main difference from my previous version where a thread only owned results in one direction.

For the tile sizes I was comparing a smaller 32x32 block tile with the 64x64 one in the file, both could work but the smaller version only needed 64 threads while the 64x64 version uses 256 and also reuses the loaded tiles more, I kept K in chunks of 32 and made each thread own 4x4 results because it looked like a reasonable balance before knowing the real register count.

```text
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
Used 64 registers, used 1 barriers, 16384 bytes smem
```

The real result from `ptxas` was more registers than I expected but nothing spilled to local memory, this kernel reached a **2381.6 GFLOP/s average** and a best launch of 2774.8 GFLOP/s on `2048x1024 * 1024x2048`.

## From 4x4 blocktiling to the current kernels

The next thing I tried was using `float4` for the global loads in [`float4blocktiling2d.cu`](blocktiling/float4blocktiling2d.cu), instead of moving one float with each expression I adapted the indexes so the aligned loads bring four values:

```cpp
float4 f4_a = reinterpret_cast<const float4*>(A)[a_pointer / 4];
float4 f4_b = reinterpret_cast<const float4*>(B)[b_pointer / 4];
```

That moved the median from around **2382 to 2982 GFLOP/s**, I also tried using `float4` when moving B from shared memory into registers but both versions performed almost the same, so it looked like the compiler may already have been doing that part for me.

After the 4x4 version I adapted it into an [8x8 thread tile](blocktiling/8x8blocktiling2d.cu) with a 128x128 block tile, the idea was to perform more FFMA for the values loaded but the first result was actually slower at around **2575 GFLOP/s**, then I increased the multiplication to `4096x2048 * 2048x4096` because I suspected the earlier benchmark was not giving this kernel enough work, the time and GFLOP/s copied into that specific commit message do not agree with each other so the graph uses the next consistent large-matrix run at **3750 GFLOP/s** instead.

I also tried transposing A inside shared memory so that its register loads could use `float4`, the first 4x4 transposed version dropped from around 3007 to 2426 GFLOP/s and Nsight Compute showed many more conflicts when writing the transposed shared tile, padding helped some later 8x8 versions but this was another case where an optimization that looked reasonable in the code did not automatically help once measured.

On the padded 8x8 version some of the stall values I recorded with `ncu` were:

```text
barrier:         1.04
long scoreboard: 0.97
MIO throttle:    2.48
```

The kernel looked more balanced than my older 4x4 version, so the next experiment was [double buffering](double-buffering/double-buffering_2d_blocktiling.cu), I added two copies of each shared tile and alternate between them so the loads for the next K tile can be issued before finishing the compute for the current one:

```cpp
__shared__ float shared_memory_1[2][BM][BK];
__shared__ float shared_memory_2[2][BK][BN];

actual = 1 - actual;
```

This reached around **3884 GFLOP/s median**, only a moderate improvement which was not too surprising after seeing the low long-scoreboard value, I also initially skipped one K tile with the loop limit and the maximum difference immediately jumped to around 17 which helped me find that mistake.

The latest transposed double-buffering work came from noticing that the 8 values each thread read from shared B left gaps between neighbouring threads and created bank conflicts, I changed the ownership so threads first cover one contiguous half of the 128 columns and then the other:

```cpp
threadIdx.x*n_float + j*(BN-(blockDim.x*n_float))
```

That also meant the C output index had to follow the new ownership, the first version calculated correctly inside the kernel but wrote the results into the old positions and fixing that one expression was less straightforward than I expected, after the complete change this variant recorded **4262 GFLOP/s median** with the older timing loop.

## First separated comparison with cuBLAS

I first measured my kernel and cuBLAS inside the same loop and the results moved more than I expected, in the latest comparison I put all launches of my kernel in one loop and all cuBLAS launches in another so they do not alternate and affect each measurement in the same way:

```text
My kernel  min 16.144 ms (4257 GFLOP/s)  mediana 17.416 ms (3946 GFLOP/s)
cuBLAS     min 14.131 ms (4863 GFLOP/s)  mediana 14.842 ms (4630 GFLOP/s)
```

Using the medians my kernel reaches about **85.2% of cuBLAS**, while comparing the two best launches gives about 87.5%, I prefer the median as the main number because it represents the complete set of launches better and the result has been consistent across the runs I made in Google Colab.

The final separated comparison used the non-transposed double-buffering file while the 4262 GFLOP/s bank-conflict fix was in the transposed variant, the next change finally continued from that optimized transposed kernel and made its occupancy problem much clearer.

## Closing this version with launch bounds

When I profiled the latest kernel, `ptxas` reported 129 registers per thread and the block already had 256 threads, this was just one register above the limit that would allow another block to be resident so I added one declaration to the kernel:

```cpp
__global__ void __launch_bounds__(256, 2) blocktiling_2d_float4rb(...)
```

The first value says the block will have at most 256 threads and the second asks the compiler to make at least two blocks per SM possible, in this case it was enough to reduce the generated kernel from 129 to 128 registers per thread and the shared-memory carveout also moved from around 32 KB to 64 KB, so the second block was no longer stopped by either resource.

| Nsight Compute metric | Before | After |
|---|---:|---:|
| Registers per thread | 129 | 128 |
| Shared-memory configuration | 32.77 KB | 65.54 KB |
| Block limit from registers / shared | 1 / 1 | 2 / 3 |
| Theoretical / achieved occupancy | 25% / 24.97% | 50% / 49.06% |
| Active warps per scheduler | 2.00 | 3.92 |
| Issued warps per scheduler | 0.49 | 0.52 |
| SM busy | 85.82% | 91.81% |
| Elapsed cycles | 16,017,465 | 14,981,601 |
| Waves per SM | 25.6 | 12.8 |

It is a slightly strange result because one register changed the amount of work that could live on the SM at once, but the measured cycles dropped by **6.47%** and the regular timing improved by about 7%, close enough that the two measurements support each other instead of only showing a faster isolated launch.

```text
My kernel  min 15.917 ms (4317 GFLOP/s)  mediana 16.175 ms (4248 GFLOP/s)
cuBLAS     min 14.804 ms (4642 GFLOP/s)  mediana 14.869 ms (4622 GFLOP/s)
```

That is **91.9% of cuBLAS using the medians** and about **93.0% using the best launches**, I also recorded 4565 GFLOP/s when running my kernel alone but I keep that separate because it is not the same direct comparison.

During the profiled run the T4 sustained around 941 MHz, which gives a clock-specific FP32 ceiling of about `2560 * 2 * 941 MHz = 4818 GFLOP/s`, the 4317 GFLOP/s launch is around **89.6% of that ceiling** while cuBLAS is around 96%, and Nsight Compute attributed 89.59% of the cycle budget to FFMA which is a useful second way of seeing that most of the kernel is now the multiplication work itself.

The remaining reports also showed zero spills, no shared-memory bank conflicts, no divergence and 100% branch efficiency, the largest stall category left was Math Pipe Throttle which in this situation mostly means the FP32 pipe is already busy executing the FFMA instructions, there are still smaller things that could be studied in SASS or instruction scheduling but the easy large bottlenecks that moved the previous versions are not there anymore.

For me this is a reasonable place to close this branch of the project, not because the kernel cannot improve but because the remaining theoretical headroom is only around 11.6% even before considering how difficult perfect scheduling would be, continuing from here would mean spending much more time on register allocation and small instruction-level changes for a very different cost and benefit than the earlier optimizations.

## Moving to tensor cores

After closing my SGEMM work I started learning WMMA and tensor cores in [`naive-tensor_cores.cu`](tensor-cores/naive-tensor_cores.cu), where one warp works together on a 16x16x16 operation instead of every thread owning its own independent results like I was used to. My first functional version reached only **342.3 GFLOP/s** and still had an occasional wrong result because `__syncthreads()` was outside the K loop, so a thread could start replacing the shared-memory tile while it was still being used.

## My tensor core performance so far

| Kernel iteration | Changes | Best | Median |
|---|---|---:|---:|
| First WMMA | My first functional 16x16x16 implementation | 342.3 GFLOP/s | — |
| Fixed tile loads | All 256 threads load the tile once instead of 8 times | 818.1 GFLOP/s | — |
| One active warp | Stopped 8 warps from repeating the same tensor core work | 1516.2 GFLOP/s | — |
| 2D blocktiling | 128x128 block tile, warp tiling and double buffering | 5743 GFLOP/s | 5641 GFLOP/s |
| Shared padding | Added 8 half values to avoid bank conflicts | 8169 GFLOP/s | 7906 GFLOP/s |
| Register loads | Kept the next tile in registers instead of local memory | 12537 GFLOP/s | 12269 GFLOP/s |
| Half inputs | Removed the unnecessary float input path | **21184 GFLOP/s** | 14027 GFLOP/s |
| Remapped A loads | Distributed the rows across shared-memory banks | 18796 GFLOP/s | **16257 GFLOP/s** |
| cuBLAS reference | Same latest comparison | 29051 GFLOP/s | 25304 GFLOP/s |

![Tensor core performance so far](assets/tensor-core-performance.svg)

The first large speedups did not come from making the tensor cores do more, they came from noticing I was doing the same work many times. I originally made every thread loop 8 times while loading only 256 values with a 256-thread block, then every one of the 8 warps was also performing the exact same WMMA operation, fixing the first mistake reached **818.1 GFLOP/s** and letting only warp 0 compute in [`tiled_tensor_cores.cu`](tensor-cores/tiled_tensor_cores.cu) reached **1516.2 GFLOP/s**. That was much faster, but leaving 7 warps almost idle was obviously not where I wanted to stop.

I then started [`2dblocktiling-tensor-cores.cu`](tensor-cores/2dblocktiling-tensor-cores.cu), using my previous 2D blocktiling and double-buffering SGEMM as the base. This was a much bigger jump than my naïve file because the block owns a 128x128 output tile, each of its 8 warps owns a 64x32 section, and every warp keeps a 4x2 array of accumulator fragments:

```cpp
wmma::fragment<wmma::accumulator, 16, 16, 16, float>
    accumulator_frag[dim_WM][dim_WN];
```

Adapting all my previous techniques at the same time, `float4` loads, double buffering, blocktiling and now warp mapping, made the indexing harder than I expected. One version compiled and reported **6147 GFLOP/s**, my highest result at the time, but its maximum difference was 319 and parts of C were still zero, so I did not count it. After fixing the dimensions and how the warps mapped the fragments, the first correct version reached **5743 GFLOP/s**.

For the comparison in [`cuBLAS-2d-tensor-cores.cu`](tensor-cores/cuBLAS-2d-tensor-cores.cu) I use half inputs and float accumulation/output for both kernels. My first comparison was accidentally still using the old SGEMM cuBLAS call, which made my kernel look faster but was not the same operation, after changing it to `cublasGemmEx` with tensor ops the real difference became clear: **5797 against 31721 GFLOP/s**, or about 18% using the best launches.

The next improvement was again shared-memory padding, this time I had to change both the declarations and the leading dimensions passed to `wmma::load_matrix_sync`, because changing only the shared arrays produced `inf` differences. Adding 8 half values moved the best result from **5797 to 8169 GFLOP/s** just by avoiding those bank conflicts.

I later found that my arrays for the next tile, `store_values_a` and `store_values_b`, were going through local memory because the compiler was not scalarizing them into registers. Making their sizes compile-time constants allowed SROA/mem2reg to work, increased the reported use to 131 registers, and reached **12537 GFLOP/s**. After that I changed the kernel inputs from float to half, the WMMA fragments and cuBLAS were already working with half and the conversion happens outside the measured region, using the same half data directly brought my best launch to **21184 GFLOP/s**, around 65% of cuBLAS.

My latest work has been guided by Nsight Compute. At first I thought low compute and memory throughput meant I should reduce shared-memory use and try to fit another block, but L1/TEX was already around 80% and the profiler pointed to shared-memory conflicts first. The old mapping sent consecutive groups of threads to consecutive rows of A, I replaced it with a mapping that separates those groups across the banks, which moved the median from **14027 to 16257 GFLOP/s**. The best launch is lower and cuBLAS was also slower in that Colab run, but in the direct latest comparison the kernel reaches **64.7% of cuBLAS**, and removing or reducing the double buffer is the next thing I want to explore.

## Benchmark notes

All these numbers are kernel-only measurements recorded on an NVIDIA T4 in Google Colab, the profiled SGEMM launch used for the clock-specific ceiling sustained around 941 MHz under the 70 W limit while other runs can report different clocks, the early SGEMM kernels, the first 4x4 blocktiling and the naïve tensor core kernel used smaller matrices while the newer SGEMM and tensor core kernels use `4096x2048 * 2048x4096`, so the comparisons are most useful inside each stage and the commit history contains the exact setup.

The kernels are still written for the aligned dimensions used in their tests and do not handle arbitrary leftover M, N or K tiles yet, the result checks compare the complete output against a CPU reference or cuBLAS but there is still benchmark variance. The SGEMM comparisons use float inputs and outputs while the tensor core comparison uses half inputs with float accumulation and output, so their GFLOP/s should not be read as a direct comparison between the two datatypes.

I used model help for parts of the benchmark code and for organizing parts of this README, the kernels and their indexing/debugging are my work.
