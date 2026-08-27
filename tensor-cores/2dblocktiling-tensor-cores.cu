%%writefile cudatensorcores.cu
#include <cstdio>
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <mma.h>

using namespace nvcuda;

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;

constexpr int dim_WM = 2;
constexpr int dim_WN = 4;

constexpr W_tile_M = 32;
constexpr W_tile_N = 64;


constexpr int tensor_M = 16;
constexpr int tensor_N = 16;
constexpr int tensor_K = 16;

constexpr int n_float = 4;

using namespace std;

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<float> dist(-2.0, 2.0);


__global__ void blocktiling_2d_float4rb(int A_num_fil, int A_num_col,float *A, int B_num_fil, int B_num_col,float *B, float *C){


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

    __shared__ float shared_memory_1[2][BM][BK];
    __shared__ float shared_memory_2[2][BK][BN];


    //Primera fase

    //Coordenadas iniciales de nuestro tile mientras va iterando
    int global_column = blockIdx.x * BN;
    int global_row = blockIdx.y * BM;

    int A_tile_col = 0; // i * tensor_K;
    int A_tile_row = global_row;
    int B_tile_col = global_column;
    int B_tile_row = 0; // i * tensor_K;

    float4 store_values_a[4];
    float4 store_values_b[4];
    
    //Esto era usado por thread pero lo dejo comentado por si me inspira
    //Position inside the dim3 threads adapted to fit A
    // int col_pos_a = (threadIdx.x%(BK/n_float))*n_float;
    // int row_pos_a = threadIdx.y*8 + threadIdx.x/2;

    //Thread Distribution inside A fragment
    for(int i = 0; i < (BM*BK)/(blockDim.x*blockDim.y*n_float); i ++){
        int a_col = (threadIdx.x%8)*4;
        int a_row = threadIdx.y*2 + i*32;
        
        int b_row = threadIdx.y/2 + i*8;
        int b_col = threadIdx.x*4 + (threadIdx.y%2)*64;

        a_pointer = A_tile_row * A_num_col + A_tile_col + a_col + a_row*A_num_col;
        b_pointer = B_tile_row * B_num_col + B_tile_col + b_col + b_row*B_num_col;

        float4 f4_a = reinterpret_cast<const float4*>(A)[a_pointer / 4];
        float4 f4_b = reinterpret_cast<const float4*>(B)[b_pointer / 4];

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


        for(int i = 0; i < (BM*BK)/(blockDim.x*blockDim.y*n_float); i ++){
            int a_col = (threadIdx.x%8)*4;
            int a_row = threadIdx.y*2 + i*32;
            
            int b_row = threadIdx.y/2 + i*8;
            int b_col = threadIdx.x*4 + (threadIdx.y%2)*64;

            a_pointer = A_tile_row * A_num_col + A_tile_col + a_col + a_row*A_num_col;
            b_pointer = B_tile_row * B_num_col + B_tile_col + b_col + b_row*B_num_col;

            float4 f4_a = reinterpret_cast<const float4*>(A)[a_pointer / 4];
            float4 f4_b = reinterpret_cast<const float4*>(B)[b_pointer / 4];

            store_values_a[i] = f4_a;
            store_values_a[i] = f4_b;
}


        for (int i = 0; i < (W_tile_M*W_tile_N)/(tensor_M*tensor_N); i ++){

            for (int j = 0; j < W_tile_M/tensor_K; j++){
                regA[j] = shared_memory_1[1 - actual][(threadIdx.y*TM)+j][(i)];
            }

            for (int j = 0; j < TN/n_float; j++){
                float4 f4_rb = reinterpret_cast<const float4*>(&shared_memory_2[1 - actual][i][threadIdx.x*TN +j*n_float])[0];

                regB[j*n_float] = f4_rb.x;
                regB[j*n_float + 1] = f4_rb.y;
                regB[j*n_float + 2] = f4_rb.z;
                regB[j*n_float + 3] = f4_rb.w;

            }

            for(int j = 0; j < TM; j++){
                for (int t = 0; t < TN; t ++){
                    sum[j][t] += regA[j]*regB[t];
                }
            }

    }



        shared_memory_1[actual][a_row][a_col + 0] = f4_a.x;
        shared_memory_1[actual][a_row][a_col + 1] = f4_a.y;
        shared_memory_1[actual][a_row][a_col + 2] = f4_a.z;
        shared_memory_1[actual][a_row][a_col + 3] = f4_a.w;

        shared_memory_2[actual][b_row][b_col + 0] = f4_b.x;
        shared_memory_2[actual][b_row][b_col + 1] = f4_b.y;
        shared_memory_2[actual][b_row][b_col + 2] = f4_b.z;
        shared_memory_2[actual][b_row][b_col + 3] = f4_b.w;


        actual = 1 - actual;


    __syncthreads();


}
    //Operando el último tile

    for (int i = 0; i < BK; i ++){

            for (int j = 0; j < TM; j++){
                regA[j] = shared_memory_1[1 - actual][(threadIdx.y*TM)+j][(i)];
            }

            for (int j = 0; j < TN/n_float; j++){
                float4 f4_rb = reinterpret_cast<const float4*>(&shared_memory_2[1 - actual][i][threadIdx.x*TN +j*n_float])[0];

                regB[j*n_float] = f4_rb.x;
                regB[j*n_float + 1] = f4_rb.y;
                regB[j*n_float + 2] = f4_rb.z;
                regB[j*n_float + 3] = f4_rb.w;

            }

            for(int j = 0; j < TM; j++){
                for (int t = 0; t < TN; t ++){
                    sum[j][t] += regA[j]*regB[t];
                }
            }
    }

    __syncthreads();

    //subimos nuestros resultados, estoy probando con float4, por lo que he tenido que cambiar cuánto aumenta i

    for (int i = 0; i < TM*TN; i += n_float){
        int c_address = B_num_col* (row_ini_bloque + threadIdx.y*TM + (i/TN)) + col_ini_bloque + i%TN + threadIdx.x*TN;

        float4 results = make_float4(sum[i/TM][i%TN],sum[i/TM][(i%TN) + 1], sum[i/TM][(i%TN) + 2], sum[i/TM][(i%TN) + 3]);
        reinterpret_cast<float4*>(&C[c_address])[0] = results;

        }


}

void report(const char* nombre, float* t, int n, double flops) {
        std::sort(t, t + n);
        printf("%-8s  min %.3f ms (%.0f GFLOP/s)   mediana %.3f ms (%.0f GFLOP/s)\n",
            nombre, t[0], flops/(t[0]/1000.0)/1e9,
            t[n/2], flops/(t[n/2]/1000.0)/1e9);
    }

int main(){

    int A_num_fil = 4096;
    int A_num_col = 2048;
    int B_num_fil = 2048;
    int B_num_col = 4096;
    int N_A = A_num_fil*A_num_col;
    int N_B = B_num_fil*B_num_col;
    int N_C = A_num_fil*B_num_col;

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

    float *d_A;
    float *d_B;
    float *d_C;

    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    dim3 threads(16,16);
    dim3 blocks(
        (B_num_col + BN - 1)/BN,
        (A_num_fil + BM - 1)/BM
    );

    //Calentamiento
    for (int i = 0; i < 3; i++){

        blocktiling_2d_float4rb<<<blocks,threads>>>(
            A_num_fil, A_num_col, d_A,
            B_num_fil, B_num_col, d_B, d_C
        );


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

        cudaEventRecord(start);
        blocktiling_2d_float4rb<<<blocks,threads>>>(A_num_fil, A_num_col, d_A, B_num_fil, B_num_col, d_B, d_C);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times_2[i], start, stop);
    }

    double flops = 2.0 * A_num_fil * B_num_col * A_num_col;


    report("Float4 en registrob", times_2, N_ITER, flops);


    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);

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
        printf("%f\n",h_C[i]);
    }
    printf("Últimos: %f\n", h_C[N_C - 1]);
    printf("Últimos: %f\n", h_C[N_C - 2]);
    printf("Últimos: %f\n", h_C[N_C - 3]);
    printf("Últimos: %f\n", h_C[N_C - 4]);

    return 0;

}