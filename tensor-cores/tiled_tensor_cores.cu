%%writefile cudatensorcores.cu
#include <cstdio>
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <mma.h>

using namespace nvcuda;
using namespace std;

constexpr int tensor_M = 16;
constexpr int tensor_N = 16;
constexpr int tensor_K = 16;

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<float> dist(-2.0, 2.0);


__global__ void tensor_cores(int A_num_fil, int A_num_col,float *A, int B_num_fil, int B_num_col,float *B, float *C){

    //Sigue la siguiente estructura (Use, M, N, K, T (type), Layout)
    wmma::fragment<wmma::matrix_a, tensor_M,tensor_N,tensor_K,half, wmma::row_major> a_fragment;
    wmma::fragment<wmma::matrix_b,tensor_M,tensor_N,tensor_K,half, wmma::row_major> b_fragment;
    wmma::fragment<wmma::accumulator,tensor_M,tensor_N,tensor_K,float> accumulator_frag;

    wmma::fill_fragment(accumulator_frag, 0.0f);

    //Índices
    int local_warp_index = (threadIdx.y/2);
    int global_column = blockDim.x * blockIdx.x;
    int global_row = blockDim.y * blockIdx.y;

    //Shared memory
    __shared__ half shared_memory_1[tensor_M][tensor_K];
    __shared__ half shared_memory_2[tensor_K][tensor_N];

    //Looping through tiles
    for (int i = 0; i < A_num_col/tensor_K; i ++){

        int A_tile_col = i * tensor_K;
        int A_tile_row = global_row;
        int B_tile_col = global_column;
        int B_tile_row = i * tensor_K;
        
        int A_pointer = A_tile_row * A_num_col + A_tile_col;
        int B_pointer = B_tile_row * B_num_col + B_tile_col;

        int thread_A_pointer = A_pointer + threadIdx.x%16 + threadIdx.y*A_num_col;
        int thread_B_pointer = B_pointer + threadIdx.x%16 + threadIdx.y*B_num_col;

        shared_memory_1[threadIdx.y][threadIdx.x] = A[thread_A_pointer];
        shared_memory_2[threadIdx.y][threadIdx.x] = B[thread_B_pointer];
        
        __syncthreads();

        if (local_warp_index == 0){
            wmma::load_matrix_sync(a_fragment, &shared_memory_1[0][0], tensor_K);
            wmma::load_matrix_sync(b_fragment, &shared_memory_2[0][0], tensor_N);

            wmma::mma_sync(accumulator_frag, a_fragment, b_fragment, accumulator_frag);
        }
        __syncthreads();

    }
    
    int puntero_resultado_C = global_row*B_num_col + global_column;

    if (local_warp_index == 0){
        wmma::store_matrix_sync(&C[puntero_resultado_C], accumulator_frag, B_num_col, wmma::mem_row_major);

    }

}

int main(){

    int A_num_fil = 1024;
    int A_num_col = 2048;
    int B_num_fil = 2048;
    int B_num_col = 1024;
    int N_A_B = A_num_fil*A_num_col;
    int N_C = A_num_fil*B_num_col;

    size_t bytes_AB = N_A_B * sizeof(float);
    size_t bytes_C = N_C * sizeof(float);

    float *h_A = (float*)malloc(bytes_AB);
    float *h_B = (float*)malloc(bytes_AB);
    float *h_C = (float*)malloc(bytes_C);
    float *h_C_2 = (float*)malloc(bytes_C);

    for (int i = 0; i < N_A_B; i ++){
        h_A[i] = dist(gen);
        h_B[i] = dist(gen);
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < A_num_fil; i++) {
        for (int j = 0; j < B_num_col; j++) {
            float sum = 0.0f;
            for (int k = 0; k < A_num_col; k++)
                sum += h_A[i*A_num_col + k] * h_B[k*B_num_col + j];
            h_C[i*B_num_col + j] = sum;
        }
    }
    for(int i = 0; i < 5; i ++){
        printf("%f\n",h_C[N_C-i]);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("CPU: %.1f ms\n", cpu_ms);

    float *d_A;
    float *d_B;
    float *d_C;
    float *d_C_2;

    cudaMalloc(&d_A, bytes_AB);
    cudaMalloc(&d_B, bytes_AB);
    cudaMalloc(&d_C, bytes_C);

    cudaMemcpy(d_A, h_A, bytes_AB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_AB, cudaMemcpyHostToDevice);

    dim3 threads(16,16);
    dim3 blocks(
        (B_num_col + threads.x - 1)/threads.x,
        (A_num_fil + threads.y - 1)/threads.y
    );

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < 3; i++)
        tensor_cores<<<blocks,threads>>>(A_num_fil, A_num_col, d_A, B_num_fil, B_num_col, d_B, d_C);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < 20; i++)
        tensor_cores<<<blocks,threads>>>(A_num_fil, A_num_col, d_A, B_num_fil, B_num_col, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    gpu_ms /= 20.0f;

    double gflops = 2.0 * A_num_fil * B_num_col * A_num_col / (gpu_ms / 1000.0) / 1e9;
    printf("GPU: %.3f ms  (%.1f GFLOP/s)\n", gpu_ms, gflops);
    printf("Speedup: %.1fx\n", cpu_ms / gpu_ms);


    cudaMemcpy(h_C_2, d_C, bytes_C, cudaMemcpyDeviceToHost);

    //New bench, to help me not miss anything
    double max_diff = 0.0;
    int bad_index = -1;
    for (int i = 0; i < N_C; i++) {
        double d = fabs((double)h_C[i] - (double)h_C_2[i]);
        if (d > max_diff) { max_diff = d; bad_index = i; }
    }
    printf("Max abs diff: %g  (at index %d)\n", max_diff, bad_index);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    for(int i = 0; i < 5; i ++){
        printf("%f\n",h_C_2[N_C-i]);
    }

    return 0;

}