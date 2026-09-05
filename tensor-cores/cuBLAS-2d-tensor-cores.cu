%%writefile cudatensorcores.cu
#include <cstdio>
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <mma.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdlib>
#define CUDA_CHECK(call) check_error((call),__LINE__,__FILE__,#call)

using namespace nvcuda;

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr float alfa = 1.0f;
constexpr float beta_gemm = 0.0f;

constexpr int dim_WM = 4;
constexpr int dim_WN = 2;

constexpr int W_tile_M = 64;
constexpr int W_tile_N = 32;


constexpr int tensor_M = 16;
constexpr int tensor_N = 16;
constexpr int tensor_K = 16;
constexpr int num_threads = 256;

constexpr int n_float = 4;

constexpr int carga_cada_thread = (BM * BK) / (num_threads * n_float);

using namespace std;

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<float> dist(-2.0, 2.0);

struct __align__(8) half4 { half x, y, z, w; };

__global__ void blocktiling_2d_float4rb(int A_num_fil, int A_num_col,const half *A, int B_num_fil, int B_num_col,const half *B, float *C){


    wmma::fragment<wmma::matrix_a, tensor_M,tensor_N,tensor_K,half, wmma::row_major> a_fragment[dim_WM];
    wmma::fragment<wmma::matrix_b,tensor_M,tensor_N,tensor_K,half, wmma::row_major> b_fragment[dim_WN];
    wmma::fragment<wmma::accumulator,tensor_M,tensor_N,tensor_K,float> accumulator_frag[dim_WM][dim_WN];

    for(int i = 0; i < dim_WM; i ++){
        for(int j = 0; j < dim_WN; j ++){
            wmma::fill_fragment(accumulator_frag[i][j], 0.0f);
        }
    }

    int local_warp_index = threadIdx.y/2;
    int actual = 0;

    __shared__ half shared_memory_1[2][BM][BK+8];
    __shared__ half shared_memory_2[2][BK][BN+8];


    //Primera fase

    //Coordenadas iniciales de nuestro tile mientras va iterando
    int global_column = blockIdx.x * BN;
    int global_row = blockIdx.y * BM;

    int A_tile_col = 0; // i * tensor_K;
    int A_tile_row = global_row;
    int B_tile_col = global_column;
    int B_tile_row = 0; // i * tensor_K;

    half4 store_values_a[carga_cada_thread];
    half4 store_values_b[carga_cada_thread];

    //Esto era usado por thread pero lo dejo comentado por si me inspira
    //Position inside the dim3 threads adapted to fit A
    // int col_pos_a = (threadIdx.x%(BK/n_float))*n_float;
    // int row_pos_a = threadIdx.y*8 + threadIdx.x/2;

    //I forgot to declare these

    int a_pointer = 0;
    int b_pointer = 0;
    //Thread Distribution inside A fragment
    for(int i = 0; i < carga_cada_thread; i ++){
        int id_thread = threadIdx.y*16 + threadIdx.x;

        int a_col = ((threadIdx.y*16 + threadIdx.x)%8)*4;
        int a_row = (((threadIdx.y*16 + threadIdx.x)/8)%8)*4 + ((threadIdx.y*16 + threadIdx.x)/8)/8 + i*32;

        int b_row = threadIdx.y/2 + i*8;
        int b_col = threadIdx.x*4 + (threadIdx.y%2)*64;

        a_pointer = A_tile_row * A_num_col + A_tile_col + a_col + a_row*A_num_col;
        b_pointer = B_tile_row * B_num_col + B_tile_col + b_col + b_row*B_num_col;

        half4 f4_a = reinterpret_cast<const half4*>(A)[a_pointer / 4];
        half4 f4_b = reinterpret_cast<const half4*>(B)[b_pointer / 4];

        shared_memory_1[actual][a_row][a_col + 0] = f4_a.x;
        shared_memory_1[actual][a_row][a_col + 1] = f4_a.y;
        shared_memory_1[actual][a_row][a_col + 2] = f4_a.z;
        shared_memory_1[actual][a_row][a_col + 3] = f4_a.w;

        shared_memory_2[actual][b_row][b_col + 0] = f4_b.x;
        shared_memory_2[actual][b_row][b_col + 1] = f4_b.y;
        shared_memory_2[actual][b_row][b_col + 2] = f4_b.z;
        shared_memory_2[actual][b_row][b_col + 3] = f4_b.w;

        __syncthreads();

}
    actual = 1 - actual;


    for (int k = 1; k < (A_num_col / BK); k ++){

        A_tile_col = k*BK;
        B_tile_row = k*BK;


        for(int i = 0; i < carga_cada_thread; i ++){
            int a_col = ((threadIdx.y*16 + threadIdx.x)%8)*4;
            int a_row = (((threadIdx.y*16 + threadIdx.x)/8)%8)*4 + ((threadIdx.y*16 + threadIdx.x)/8)/8 + i*32;

            int b_row = threadIdx.y/2 + i*8;
            int b_col = threadIdx.x*4 + (threadIdx.y%2)*64;

            a_pointer = A_tile_row * A_num_col + A_tile_col + a_col + a_row*A_num_col;
            b_pointer = B_tile_row * B_num_col + B_tile_col + b_col + b_row*B_num_col;

            half4 f4_a = reinterpret_cast<const half4*>(A)[a_pointer / 4];
            half4 f4_b = reinterpret_cast<const half4*>(B)[b_pointer / 4];

            store_values_a[i] = f4_a;
            store_values_b[i] = f4_b;
}


        for (int i = 0; i < BK/tensor_K; i ++){

            for (int j = 0; j < W_tile_M/tensor_M; j++){
                wmma::load_matrix_sync(a_fragment[j], &shared_memory_1[1-actual][(local_warp_index/4)*64 + j*tensor_K][i*tensor_K], BK + 8);
            }

            for (int j = 0; j < W_tile_N/tensor_N; j++){
                wmma::load_matrix_sync(b_fragment[j], &shared_memory_2[1-actual][i*tensor_K][(local_warp_index%4)*(BN/4) + j*tensor_K], BN + 8);
            }

            for(int j = 0; j < dim_WM; j++){
                for (int t = 0; t < dim_WN; t ++){
                    wmma::mma_sync(accumulator_frag[j][t],a_fragment[j],b_fragment[t],accumulator_frag[j][t]);
                }
            }

    }

    __syncthreads();

        //repito mi código
        for(int i = 0; i < carga_cada_thread; i ++){
            int a_col = ((threadIdx.y*16 + threadIdx.x)%8)*4;
            int a_row = (((threadIdx.y*16 + threadIdx.x)/8)%8)*4 + ((threadIdx.y*16 + threadIdx.x)/8)/8 + i*32;

            int b_row = threadIdx.y/2 + i*8;
            int b_col = threadIdx.x*4 + (threadIdx.y%2)*64;

            shared_memory_1[actual][a_row][a_col + 0] = store_values_a[i].x;
            shared_memory_1[actual][a_row][a_col + 1] = store_values_a[i].y;
            shared_memory_1[actual][a_row][a_col + 2] = store_values_a[i].z;
            shared_memory_1[actual][a_row][a_col + 3] = store_values_a[i].w;

            shared_memory_2[actual][b_row][b_col + 0] = store_values_b[i].x;
            shared_memory_2[actual][b_row][b_col + 1] = store_values_b[i].y;
            shared_memory_2[actual][b_row][b_col + 2] = store_values_b[i].z;
            shared_memory_2[actual][b_row][b_col + 3] = store_values_b[i].w;
        }


        actual = 1 - actual;


    __syncthreads();


}
    //Operando el último tile

    for (int i = 0; i < BK/tensor_K; i ++){

            for (int j = 0; j < W_tile_M/tensor_M; j++){
                wmma::load_matrix_sync(a_fragment[j], &shared_memory_1[1-actual][(local_warp_index/4)*(BM/2) + j*tensor_K][i*tensor_K], BK + 8);
            }

            for (int j = 0; j < W_tile_N/tensor_N; j++){
                wmma::load_matrix_sync(b_fragment[j], &shared_memory_2[1-actual][i*tensor_K][(local_warp_index%4)*(BN/4) + j*tensor_K], BN + 8);
            }

            for(int j = 0; j < dim_WM; j++){
                for (int t = 0; t < dim_WN; t ++){
                    wmma::mma_sync(accumulator_frag[j][t],a_fragment[j],b_fragment[t],accumulator_frag[j][t]);
                }
            }

    }

    __syncthreads();

    //subimos nuestros resultados, ya no uso float4

    for (int j = 0; j < W_tile_M/tensor_M; j++){
        for (int s = 0; s < W_tile_N/tensor_N; s++){

            wmma::store_matrix_sync(&C[(global_row + (local_warp_index / 4)*64 + j*tensor_M)*B_num_col + (global_column + (local_warp_index % 4)*32  + s*tensor_N)],accumulator_frag[j][s],B_num_col,wmma::mem_row_major);
        }
    }




}

__global__ void f32_to_f16(const float* __restrict__ src, half* __restrict__ dst, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

void report(const char* nombre, float* t, int n, double flops) {
        std::sort(t, t + n);
        printf("%-8s  min %.3f ms (%.0f GFLOP/s)   mediana %.3f ms (%.0f GFLOP/s)\n",
            nombre, t[0], flops/(t[0]/1000.0)/1e9,
            t[n/2], flops/(t[n/2]/1000.0)/1e9);
    }

void check_error(cudaError_t error, int line, const char* file, const char* error_line){
    if(error != cudaSuccess){
        printf("\nOn line %d: %s",line, error_line);
        printf("\nError: %s\nFound at file: %s\n", cudaGetErrorString(error), file);
        exit(EXIT_FAILURE);
    }
}

int main(){

    int A_num_fil = 4096;
    int A_num_col = 2048;
    int B_num_fil = 2048;
    int B_num_col = 4096;
    int N_A = A_num_fil*A_num_col;
    int N_B = B_num_fil*B_num_col;
    int N_C = A_num_fil*B_num_col;
    cublasHandle_t handle;

    size_t bytes_A = (size_t)A_num_fil*A_num_col * sizeof(float);
    size_t bytes_B = (size_t)B_num_fil*B_num_col * sizeof(float);
    size_t bytes_C = (size_t)A_num_fil*B_num_col * sizeof(float);

    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    float *h_C = (float*)malloc(bytes_C);
    float *h_C_2 = (float*)malloc(bytes_C);



    for (int i = 0; i < N_A; i ++){
        h_A[i] = dist(gen);
    }

    for (int i = 0; i < N_B; i ++){
        h_B[i] = dist(gen);
    }




    //I usually remove this part if I'm working with 4096, but I reduce dimensions and leave this part if
    // I'm checking whether my code works or not.

    float *d_Af;
    float *d_Bf;
    float *d_C;
    float *d_C_cub;
    half *d_Ah;
    half *d_Bh;

    // FP32 temporary buffers
    CUDA_CHECK(cudaMalloc(&d_Af, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_Bf, bytes_B));

    CUDA_CHECK(cudaMalloc(&d_Ah, (size_t)N_A * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_Bh, (size_t)N_B * sizeof(half)));


    CUDA_CHECK(cudaMalloc(&d_Af, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_Bf, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    CUDA_CHECK(cudaMalloc(&d_C_cub, bytes_C));
    cublasCreate(&handle);

    CUDA_CHECK(cudaMemcpy(d_Af, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bf, h_B, bytes_B, cudaMemcpyHostToDevice));


    f32_to_f16<<<(N_A + 255) / 256, 256>>>(d_Af, d_Ah, N_A);
    f32_to_f16<<<(N_B + 255) / 256, 256>>>(d_Bf, d_Bh, N_B);
    cudaDeviceSynchronize();

    dim3 threads(16,16);
    dim3 blocks(
        (B_num_col + BN - 1)/BN,
        (A_num_fil + BM - 1)/BM
    );

    //Calentamiento
    for (int i = 0; i < 3; i++){

        blocktiling_2d_float4rb<<<blocks,threads>>>(
            A_num_fil, A_num_col, d_Ah,
            B_num_fil, B_num_col, d_Bh, d_C
        );
        CUDA_CHECK(cudaGetLastError());

        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, B_num_col, A_num_fil, A_num_col, &alfa, d_Bh, CUDA_R_16F, B_num_col,
             d_Ah, CUDA_R_16F, A_num_col, &beta_gemm, d_C_cub, CUDA_R_32F, B_num_col, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    }

    cudaDeviceSynchronize();


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int N_ITER = 50;
    float times[N_ITER] = {0.0f};
    float times_2[N_ITER] = {0.0f};

    for (int i = 0; i < N_ITER; i++) {

        cudaEventRecord(start);
        blocktiling_2d_float4rb<<<blocks,threads>>>(A_num_fil, A_num_col, d_Ah, B_num_fil, B_num_col, d_Bh, d_C);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times_2[i], start, stop);
        CUDA_CHECK(cudaGetLastError());

        cudaEventRecord(start);
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, B_num_col, A_num_fil, A_num_col, &alfa, d_Bh, CUDA_R_16F, B_num_col,
             d_Ah, CUDA_R_16F, A_num_col, &beta_gemm, d_C_cub, CUDA_R_32F, B_num_col, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);

    }



    double flops = 2.0 * A_num_fil * B_num_col * A_num_col;


    report("Tensor Core Kernel", times_2, N_ITER, flops);
    report("GEMM cuBLAS", times, N_ITER, flops);


    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CUDA_CHECK(cudaMemcpy(h_C_2, d_C, bytes_C, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C, d_C_cub, bytes_C, cudaMemcpyDeviceToHost));

    //New bench, to help me not miss anything
    double max_diff = 0.0;
    int bad_index = -1;
    for (int i = 0; i < N_C; i++) {
        double d = fabs((double)h_C[i] - (double)h_C_2[i]);
        if (d > max_diff) { max_diff = d; bad_index = i; }
    }
    printf("Max abs diff: %g  (at index %d)\n", max_diff, bad_index);

    cudaFree(d_Af);
    cudaFree(d_Bf);
    cudaFree(d_C);
    cudaFree(d_C_cub);


    return 0;

}