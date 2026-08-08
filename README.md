# CUDA kernels journey

This is a report on my path from my first vector addition to a handwritten **654.1 GFLOP/s** 32x32 1D memory matrix multiplication kernel on an NVIDIA T4.

My code is not perfect, there are mistakes, missed optimizations, and I am deliberately keeping the intermediate versions for all my kernels. Because I not only want to show my current best number, but I also want to show the trail of indexing mistakes, incorrect assumptions, and small hardware aware changes that I made to get speedup after speedup.

My clearest example has been my 32x32 1D memory tiled matrix multiplication. 
I had not learned to use 2D shared-memory arrays yet, so I flattened every tile manually in 1D, including its transposition and padding, which made the indexing much harder.

My first working 32×32 tiled kernel reached **195.1 GFLOP/s**, underperforming even my naïve matmul version's **355.9 GFLOP/s**, which should not be happening. 
On this README, I'll be documenting optimizations like padding my shared memory to remove bank conflicts, which raised **195.1 GFLOP/s** to **470 GFLOP/s**. 
Or remapping how each warp loads B from global memory, which then pushed it to **654.1 GFLOP/s**.

## Performance journey on my 1D 32x32 Tiled Matmul

| Tiled matmul iteration | What changed | GFLOP/s | Improvement |
|---|---|---:|---:|
| Correct 32×32 tiles | First validated baseline | 195.1 | 1.00× |
| Padded shared memory | Removed bank conflicts with a 32×33 layout | 470.0 | 2.41× |
| Coalesced B loads | Each warp loads consecutive values from one row | **654.1** | **3.35×** |

![Performance journey from 195.1 to 654.1 GFLOP/s](assets/tiled-matmul-performance.svg)

The latest change added **184.1 GFLOP/s** over the padded version, which would be about **39%**, or **1.39×**. Calling it “around 200 GFLOP/s” is a fair rough description, but 184.1 GFLOP/s is the difference between the two recorded results.

## How I got here

### 1. Starting with vector addition

My first kernel was [`vector-add/vector_add.cu`](vector-add/vector_add.cu). 
It basically taught me me the basic CUDA loop: allocate host and device memory, copy the inputs, map one thread to one output, launch the kernel, and copy the result back.

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < N) C[i] = A[i] + B[i];
```

The operation is very simple, but it helped me land the correct mental model I needed before moving to a kernel where indexing and memory access were more difficult.

### 2. Writing a naïve matmul—and discovering that my first bug was outside the kernel

In [`matmul/naive-matmul.cu`](matmul/naive-matmul.cu), each thread computes one output value. And basically my first ever attempt produced invalid results, including `inf`. And looking back through my commits, my matrix setup had three main separate problems:

```cpp
// My original version
size_t bytes_m2 = M_fil_1 * K_col_2 * sizeof(float); // Here I had the wrong number of rows

for (int i = 0; i < M_fil_2 * K_col_2; i++)
    h_mat1[i] = 3.0f;                                // here wrote into A again

cudaMemcpy(d_mat2, h_mat1, bytes_m2,
           cudaMemcpyHostToDevice);                 // And here I copied A instead of B, (rookie mistake)
```

I had under-allocated B, initialized the wrong host pointer, and then copied the wrong matrix to the device, very bad stuff.

I fixed all three together:

```cpp
size_t bytes_m2 = M_fil_2 * K_col_2 * sizeof(float);

for (int i = 0; i < M_fil_2 * K_col_2; i++)
    h_mat2[i] = dist(gen);

cudaMemcpy(d_mat2, h_mat2, bytes_m2,
           cudaMemcpyHostToDevice);
```

I also switched to random floats in the range `[-2, 2]`, to not get giant numbers. 

### 3. Building a benchmark instead of trusting one launch

Once the naïve version was working, I added a single-threaded CPU implementation as a reference, to measure the Speedup compared to a CPU, and also to check if my kernel worked correctly (this will come very handy in the tiled matmul later).

I then changed the GPU measurement from one launch to three warm-up launches followed by 20 timed iterations using CUDA events (I was helped by claude Opus 5 in medium setting to build this benchmark step, since at this point I was very early at learning CUDA, and needed this right away):

```cpp
for (int i = 0; i < 3; i++)
    matmul<<<blocks, threads>>>(/* ... */);
cudaDeviceSynchronize();

cudaEventRecord(start);
for (int i = 0; i < 20; i++)
    matmul<<<blocks, threads>>>(/* ... */);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
```

This benchmark basically gave me a repeatable kernel time, a CPU comparison, and a way to calculate throughput rather than relying only on milliseconds.

### 4. My first shared-memory design did not work

In [`tiled-matmul/tiled-matmul-it1.cu`](tiled-matmul/tiled-matmul-it1.cu), I tried to cache the complete data needed by each output in two block-wide shared arrays:

```cpp
shared_memory_2[threadIdx.x] =
    mat2[col_num_1 + threadIdx.x * K_col_2];

sum += shared_memory_1[j] * shared_memory_2[j];
```

The problem I had just unknowingly created was a design problem, not just a typo. 
Every thread was computing a different output column, but all 1024 threads were writing their values into the same shared array as if it were private to each thread. 
After synchronization, that array was a mess, intended for different outputs, and of course this did not work at all.

I temporarily removed the second shared array and went back to reading B from global memory. It was not the optimization I wanted, but it gave me a correct stepping stone and helped me understand that shared memory has to represent data a whole block can genuinely reuse.

### 5. Moving to real 32×32 tiles in 1D

My next attempt, [`tiled-matmul/tiled-matmul-it2.cu`](tiled-matmul/tiled-matmul-it2.cu), made each 1024-thread block responsible for a 32×32 output tile. 

Keeping the tiles in 1D meant that I had to map every row and column by hand instead of using cleaner 2D indexing.

My first indexing still treated the flattened global thread ID as if it directly described the tile:

```cpp
// Earlier indexing
int block_row = i / (K_col_1 * 32);
int block_col = i / (M_filas_2 * 32);

shared_memory_1[threadIdx.x] = mat1[/* ... */ + threadIdx.x];
result[i] = sum;
```

That did not correctly separate the block position from the thread position, so I refactored the mapping around `blockIdx.x`, used `% 32` for a thread's column inside the tile, and explicitly mapped the result back into the output matrix:

```cpp
int block_row = blockIdx.x / 32;
int block_col = blockIdx.x % 32;

int tile_row = threadIdx.x / 32;
int tile_col = threadIdx.x % 32;
```

I then added a complete CPU/GPU comparison instead of checking only the first few values (because I had an error that could not be detected by comparing the first values from both the CPU and GPU alone) (again, for benchmarks like this one, I asked for it to Claude Opus 5 in the medium effort setting):

```cpp
float max_error = 0.0f;
for (int i = 0; i < N; i++)
    max_error = fmaxf(max_error, fabsf(h_result[i] - h_C_cpu[i]));
```

After having refactored my mapping, my kernel's first version measured **197.1 GFLOP/s**, which is worse than my naïve matmul (my naÏve matmul implementation benefits a lot from cache). 

At this point I was surprised, because I was expecting tiled matmul to do much better than my naïve matmul, but even tho my tiled matmul was producing the correct output, my kernel was slow, adding shared memory had not automatically made the kernel faster.

### 6. Removing shared-memory bank conflicts: 195.1 → 470 GFLOP/s

Finding this speedup opportunity took me 5 hours of sitting around and looking at my poorly written code, which I had overcomplicated, because I had written it to work with a 1D memory, since I hadn't learned to use 2D memories at this point, and I overcomplicated debugging also by not commenting at all my code.

To understand what was going wrong, in the inner product, each warp read B from shared memory with a stride of 32:

```cpp
// 32×32 layout
__shared__ float shared_memory_2[32 * 32];
value = shared_memory_2[column * 32 + k];
```

And because the T4 has 32 shared-memory banks, a stride of 32 mapped those per-thread addresses back onto the same bank, which basically means my accesses were seralizing instead of happening in parallel, and that's something we don't want to happen if we want to go fast.

So I added a one padding element per row and changed the stride to 33:

```cpp
// 32×33 padded layout
__shared__ float shared_memory_2[32 * 33];
value = shared_memory_2[column * 33 + k];
```

That extra column is only 32 additional floats, but it shifts each row onto a different bank. 
This change alone made my throughput jump from **195.1 to 470 GFLOP/s**, a **2.41×** improvement.

This was a turning point for me because it was the first time where I could connect a very small code change to a specific piece of my T4 GPU hardware and then see the effect clearly in the benchmark.

### 7. Coalescing the global loads for B:   470 → 654.1 GFLOP/s

When I fixed the padding, it fixed how B was *read from my shared memory*, but there was another problem, which took me even more time to find than the 5h I had to spend reading the code for the first. 

Basically now I was reading from shared memory correctly, but my data still had to get there from global memory, and the problem was HOW it was arriving from global memory. 

Before my latest change, the lanes in a warp loaded values from the same column across different rows:

```cpp
// simplification extracted from my code, renamed variables to make it more understandable for everyone
int column = threadIdx.x / 32; 
int row    = threadIdx.x % 32; 

shared_memory_2[column * 33 + row] =
    mat2[tile_base + row * K_col_2 + column];
```

This meant that lanes were separated by `K_col_2` elements in global memory, which was a distance of 1024 floats. Which means that I wasn't doing coalescing correly, what I had were heavily strided loads.

I thought of remapping each warp so its lanes load consecutive columns from one row (addapting to row-mayor format):

```cpp
// After: the lanes in a warp move across one row of B
int row    = threadIdx.x / 32; // constant within a warp
int column = threadIdx.x % 32; // changes across the warp

shared_memory_2[column * 33 + row] =
    mat2[tile_base + row * K_col_2 + column];
```

This meant my global reads were now coalesced, while the transposed write into `shared_memory_2[column * 33 + row]` preserves the padded layout used by the computation.

![Here is a before and after diagram of the global-memory load pattern for B](assets/coalesced-b-loads.svg)

This raised my result from **470 to 654.1 GFLOP/s**: **+184.1 GFLOP/s**, about **39%** faster than the previous version and **3.35×** faster than the first validated tiled kernel.

## Benchmark details

I recorded the results that I'm presenting in **Google Colab on an NVIDIA T4** with the following fixed multiplication:

```text
A: 1024 × 2048
B: 2048 × 1024
C: 1024 × 1024
Work: 2 × M × N × K ≈ 4.295 billion floating-point operations
```

I calculate throughput with:

```text
GFLOP/s = (2 × M × N × K) / elapsed_seconds / 1e9
```

These are **kernel-only measurements**. 

Host-to-device and device-to-host transfers happen outside this timed region, also, the CPU implementation is my simple single-threaded correctness reference, not an optimized BLAS implementation, so any CPU/GPU speedup printed should be read in that context.

The 195.1 and 470 results come from their corresponding commit notes, 654.1 is my latest local measurement. I have not committed a raw benchmark-results file yet, so I treat this graph as an honest snapshot of my learning progress, and not a universal performance claim. 

GPU clocks, compiler versions, and repeated runs can all move the exact values.

## Repository map

| File | What it represents | Status |
|---|---|---|
| [`vector-add/vector_add.cu`](vector-add/vector_add.cu) | My first CUDA kernel | Working |
| [`matmul/naive-matmul.cu`](matmul/naive-matmul.cu) | One thread per output, reading directly from global memory | Working baseline |
| [`tiled-matmul/tiled-matmul-it1.cu`](tiled-matmul/tiled-matmul-it1.cu) | First shared-memory experiment and its design flaw | Historical iteration |
| [`tiled-matmul/tiled-matmul-it2.cu`](tiled-matmul/tiled-matmul-it2.cu) | 32×32 tiling, validation, padding, and coalesced B loads | Current iteration |

The tiled files still include the `%%writefile` notebook magic I used in Colab. That first line needs to be removed before compiling the files directly with local `nvcc`.

The standalone kernels can be compiled with:

```bash
nvcc -O3 vector-add/vector_add.cu -o vector-add/vector_add
nvcc -O3 matmul/naive-matmul.cu -o matmul/naive-matmul
```

## Disclaimer

I am not trying to replace cuBLAS with this repository, I am only trying to reach the point where I can explain *why* each version behaves the way it does and then prove the explanation with the next measurement.
