#include <cstdio>
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cublas_v2.h>
#include <thread>

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr float alfa = 1.0f;
constexpr float beta_gemm = 0.0f;


constexpr int TM = 8;
constexpr int TN = 8;
constexpr int n_float = 4;


constexpr int num_threads = (BM*BN) / (TM*TN);


using namespace std;

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<float> dist(-2.0, 2.0);


__global__ void blocktiling_2d_float4rb(int A_num_fil, int A_num_col,float *A, int B_num_fil, int B_num_col,float *B, float *C){

    __shared__ float shared_memory_1[BK][BM+4];
    __shared__ float shared_memory_2[BK][BN];

    float sum[TM][TN] = {0.0f};
    float registerA[TM];
    float registerB[TN];


    //Coordenadas iniciales de nuestro tile mientras va iterando
    int col_ini_bloque = blockIdx.x * BN;
    int row_ini_bloque = blockIdx.y * BM;

    for (int k = 0; k < A_num_col / BK; k ++){

        int col_ini_a = k * BK;
        int row_ini_b = k * BK;

        for (int i = 0; i < (BK*BM)/(num_threads*n_float); i ++){

            //Position inside the dim3 threads adapted to fit A
            int col_pos_a = (threadIdx.x%(BK/n_float))*n_float;
            int row_pos_a = threadIdx.y*8 + threadIdx.x/2 + (i*BM)/((BK*BM)/(num_threads*n_float));

            //Position inside A
            int a_pointer = row_ini_bloque*A_num_col + col_ini_a + col_pos_a + row_pos_a*A_num_col;

            float4 f4_a = reinterpret_cast<const float4*>(A)[a_pointer / 4];

            shared_memory_1[col_pos_a + 0][row_pos_a] = f4_a.x;
            shared_memory_1[col_pos_a + 1][row_pos_a] = f4_a.y;
            shared_memory_1[col_pos_a + 2][row_pos_a] = f4_a.z;
            shared_memory_1[col_pos_a + 3][row_pos_a] = f4_a.w;

            //Position inside the dim3 threads adapted to fit B
            int col_pos_b = threadIdx.x*4 + (threadIdx.y%2)*64;
            int row_pos_b = threadIdx.y/2;

            int b_pointer = col_ini_bloque + col_pos_b + row_ini_b*B_num_col + row_pos_b*B_num_col ;

            float4 f4_b = reinterpret_cast<const float4*>(B)[b_pointer / 4];

            shared_memory_2[row_pos_b][col_pos_b + 0] = f4_b.x;
            shared_memory_2[row_pos_b][col_pos_b + 1] = f4_b.y;
            shared_memory_2[row_pos_b][col_pos_b + 2] = f4_b.z;
            shared_memory_2[row_pos_b][col_pos_b + 3] = f4_b.w;

    }
    __syncthreads();



        for (int i = 0; i < BK; i ++){

            for (int s = 0; s < TN/n_float; s++){
                float4 f4_registera = reinterpret_cast<const float4*>(&shared_memory_1[i][threadIdx.y*TN + s*n_float])[0];
                registerA[s*n_float + 0] = f4_registera.x;
                registerA[s*n_float + 1] = f4_registera.y;
                registerA[s*n_float + 2] = f4_registera.z;
                registerA[s*n_float + 3] = f4_registera.w;
            }

            for (int j = 0; j < TN/n_float; j++){
                float4 f4_rb = reinterpret_cast<const float4*>(&shared_memory_2[i][threadIdx.x*TN +j*n_float])[0];

                registerB[j*n_float] = f4_rb.x;
                registerB[j*n_float + 1] = f4_rb.y;
                registerB[j*n_float + 2] = f4_rb.z;
                registerB[j*n_float + 3] = f4_rb.w;

            }

            for(int j = 0; j < TM; j++){
                for (int t = 0; t < TN; t ++){
                    sum[j][t] += registerA[j]*registerB[t];
                }
            }
    }
    __syncthreads();
}

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


    cublasHandle_t handle;
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
    float *h_C_cub = (float*)malloc(bytes_C);

    for (int i = 0; i < N_A; i ++){
        h_A[i] = dist(gen);
    }

    for (int i = 0; i < N_B; i ++){
        h_B[i] = dist(gen);
    }

    

    float *d_A;
    float *d_B;
    float *d_C;
    float *d_C_cub;

    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);
    cudaMalloc(&d_C_cub, bytes_C);
    cublasCreate(&handle);

    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    dim3 threads(16,16);
    dim3 blocks(
        (B_num_col + BN - 1)/BN,
        (A_num_fil + BM - 1)/BM
    );

    //Calentamiento
    for (int i = 0; i < 10; i++){

        blocktiling_2d_float4rb<<<blocks,threads>>>(
            A_num_fil, A_num_col, d_A,
            B_num_fil, B_num_col, d_B, d_C
        );

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                B_num_col, A_num_fil, A_num_col, &alfa, d_B, B_num_col, d_A, A_num_col, &beta_gemm, d_C_cub, B_num_col);

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
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                B_num_col, A_num_fil, A_num_col, &alfa, d_B, B_num_col, d_A, A_num_col, &beta_gemm, d_C_cub, B_num_col);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);

        cudaEventRecord(start);
        blocktiling_2d_float4rb<<<blocks,threads>>>(A_num_fil, A_num_col, d_A, B_num_fil, B_num_col, d_B, d_C);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times_2[i], start, stop);
    }

    double flops = 2.0 * A_num_fil * B_num_col * A_num_col;


    report("Sin cublass", times_2, N_ITER, flops);
    report("Con cublass", times, N_ITER, flops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_cub, d_C_cub, bytes_C, cudaMemcpyDeviceToHost);

    //New bench, to help me not miss anything
    double max_diff = 0.0;
    int bad_index = -1;
    for (int i = 0; i < N_C; i++) {
        double d = fabs((double)h_C[i] - (double)h_C_cub[i]);
        if (d > max_diff) { max_diff = d; bad_index = i; }
    }
    printf("Max abs diff: %g  (at index %d)\n", max_diff, bad_index);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    for(int i = 0; i < 5; i ++){
        printf("%f\n",h_C[i]);
    }
    printf("Últimos: %f\n", h_C[N_C - 1]);
    printf("Últimos: %f\n", h_C[N_C - 2]);
    printf("Últimos: %f\n", h_C[N_C - 3]);
    printf("Últimos: %f\n", h_C[N_C - 4]);

    return 0;

}