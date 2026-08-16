%%writefile blocktiling-2d.cu
#include <cstdio>
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 32;


constexpr int TM = 4;
constexpr int TN = 4;
constexpr int n_float = 4;


constexpr int num_threads = (BM*BN) / (TM*TN);


using namespace std;

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<float> dist(-2.0, 2.0);

__global__ void blocktiling_2d_float4transposed(int A_num_fil, int A_num_col,float *A, int B_num_fil, int B_num_col,float *B, float *C){

    __shared__ float shared_memory_1[BK][BM];
    __shared__ float shared_memory_2[BK][BN];

    float sum[TM][TN] = {0.0f};
    float registroA[TM];
    float registroB[TN];


    //Coordenadas iniciales de nuestro tile mientras va iterando
    int col_ini_bloque = blockIdx.x * BN;
    int row_ini_bloque = blockIdx.y * BM;

    for (int k = 0; k < A_num_col / BK; k ++){

        int col_ini_a = k * BK;
        int row_ini_b = k * BK;

        for (int i = 0; i < (BK*BM)/(num_threads*n_float); i ++){

            //Position inside the dim3 threads adapted to fit A
            int col_pos_a = (threadIdx.x%(BK/n_float))*4;
            int row_pos_a = threadIdx.y*2 + threadIdx.x/8 + i*BM/((BK*BM)/(num_threads*n_float));

            //Position inside A
            int a_pointer = row_ini_bloque*A_num_col + col_ini_a + col_pos_a + row_pos_a*A_num_col;

            float4 f4_a = reinterpret_cast<const float4*>(A)[a_pointer / 4];

            shared_memory_1[col_pos_a + 0][row_pos_a] = f4_a.x;
            shared_memory_1[col_pos_a + 1][row_pos_a] = f4_a.y;
            shared_memory_1[col_pos_a + 2][row_pos_a] = f4_a.z;
            shared_memory_1[col_pos_a + 3][row_pos_a] = f4_a.w;

            //Position inside the dim3 threads adapted to fit B
            int col_pos_b = threadIdx.x*4;
            int row_pos_b = threadIdx.y + i*16;

            int b_pointer = col_ini_bloque + col_pos_b + row_ini_b*B_num_col + row_pos_b*B_num_col ;

            float4 f4_b = reinterpret_cast<const float4*>(B)[b_pointer / 4];

            shared_memory_2[row_pos_b][col_pos_b + 0] = f4_b.x;
            shared_memory_2[row_pos_b][col_pos_b + 1] = f4_b.y;
            shared_memory_2[row_pos_b][col_pos_b + 2] = f4_b.z;
            shared_memory_2[row_pos_b][col_pos_b + 3] = f4_b.w;

    }
    __syncthreads();



        for (int i = 0; i < BK; i ++){
            
            float4 f4_registera = reinterpret_cast<const float4*>(&shared_memory_1[i][threadIdx.y*TM])[0];
            registroA[0] = f4_registera.x;
            registroA[1] = f4_registera.y;
            registroA[2] = f4_registera.z;
            registroA[3] = f4_registera.w;

            
            float4 f4_registerb = reinterpret_cast<const float4*>(&shared_memory_2[i][threadIdx.x*TN])[0];
            
            registroB[0] = f4_registerb.x;
            registroB[1] = f4_registerb.y;
            registroB[2] = f4_registerb.z;
            registroB[3] = f4_registerb.w;

            for(int j = 0; j < TM; j++){
                for (int t = 0; t < TN; t ++){
                    sum[j][t] += registroA[j]*registroB[t];
                }
            }
    }
    __syncthreads();
}

    //subimos nuestros resultados
    for (int i = 0; i < TM*TN; i++){
        int c_address = B_num_col* (row_ini_bloque + threadIdx.y*4 + (i/4)) + col_ini_bloque + i%4 + threadIdx.x*4;
        C[c_address] = sum[i/4][i%4];
    }


}


__global__ void blocktiling_2d_float4rb(int A_num_fil, int A_num_col,float *A, int B_num_fil, int B_num_col,float *B, float *C){

    __shared__ float shared_memory_1[BM][BK];
    __shared__ float shared_memory_2[BK][BN];

    float sum[TM][TN] = {0.0f};
    float regA[TM];
    float regB[TN];


    //Coordenadas iniciales de nuestro tile mientras va iterando
    int col_ini_bloque = blockIdx.x * BN;
    int row_ini_bloque = blockIdx.y * BM;

    for (int k = 0; k < A_num_col / BK; k ++){

        int col_ini_a = k * BK;
        int row_ini_b = k * BK;

        for (int i = 0; i < (BK*BM)/(num_threads*n_float); i ++){

            //Position inside the dim3 threads adapted to fit A
            int col_pos_a = (threadIdx.x%(BK/n_float))*4;
            int row_pos_a = threadIdx.y*2 + threadIdx.x/8 + i*BM/((BK*BM)/(num_threads*n_float));

            //Position inside A
            int a_pointer = row_ini_bloque*A_num_col + col_ini_a + col_pos_a + row_pos_a*A_num_col;

            float4 f4_a = reinterpret_cast<const float4*>(A)[a_pointer / 4];

            shared_memory_1[row_pos_a][col_pos_a + 0] = f4_a.x;
            shared_memory_1[row_pos_a][col_pos_a + 1] = f4_a.y;
            shared_memory_1[row_pos_a][col_pos_a + 2] = f4_a.z;
            shared_memory_1[row_pos_a][col_pos_a + 3] = f4_a.w;

            //Position inside the dim3 threads adapted to fit B
            int col_pos_b = threadIdx.x*4;
            int row_pos_b = threadIdx.y + i*16;

            int b_pointer = col_ini_bloque + col_pos_b + row_ini_b*B_num_col + row_pos_b*B_num_col ;

            float4 f4_b = reinterpret_cast<const float4*>(B)[b_pointer / 4];

            shared_memory_2[row_pos_b][col_pos_b + 0] = f4_b.x;
            shared_memory_2[row_pos_b][col_pos_b + 1] = f4_b.y;
            shared_memory_2[row_pos_b][col_pos_b + 2] = f4_b.z;
            shared_memory_2[row_pos_b][col_pos_b + 3] = f4_b.w;

    }
    __syncthreads();



        for (int i = 0; i < BK; i ++){

            for (int j = 0; j < TM; j++){
                regA[j] = shared_memory_1[(threadIdx.y*4)+j][(i)];
            }
            
            float4 f4_rb = reinterpret_cast<const float4*>(&shared_memory_2[i][threadIdx.x*TN])[0];
            
            regB[0] = f4_rb.x;
            regB[1] = f4_rb.y;
            regB[2] = f4_rb.z;
            regB[3] = f4_rb.w;

            for(int j = 0; j < TM; j++){
                for (int t = 0; t < TN; t ++){
                    sum[j][t] += regA[j]*regB[t];
                }
            }
    }
    __syncthreads();
}

    //subimos nuestros resultados
    for (int i = 0; i < TM*TN; i++){
        int c_address = B_num_col* (row_ini_bloque + threadIdx.y*4 + (i/4)) + col_ini_bloque + i%4 + threadIdx.x*4;
        C[c_address] = sum[i/4][i%4];
    }


}

void report(const char* nombre, float* t, int n, double flops) {
        std::sort(t, t + n);
        printf("%-8s  min %.3f ms (%.0f GF/s)   mediana %.3f ms (%.0f GFLOP/s)\n",
            nombre, t[0], flops/(t[0]/1000.0)/1e9,
            t[n/2], flops/(t[n/2]/1000.0)/1e9);
    }

int main(){

    int A_num_fil = 2048;
    int A_num_col = 1024;
    int B_num_fil = 1024;
    int B_num_col = 2048;
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

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < A_num_fil; i++) {
        for (int j = 0; j < B_num_col; j++) {
            float sum = 0.0f;
            for (int k = 0; k < A_num_col; k++)
                sum += h_A[i*A_num_col + k] * h_B[k*B_num_col + j];
            h_C_2[i*B_num_col + j] = sum;
        }
    }
    for(int i = 0; i < 5; i ++){
        printf("%f\n",h_C_2[i]);
    }
    printf("Últimos: %f\n", h_C_2[N_C - 1]);
    printf("Últimos: %f\n", h_C_2[N_C - 2]);
    printf("Últimos: %f\n", h_C_2[N_C - 3]);
    printf("Últimos: %f\n", h_C_2[N_C - 4]);

    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("CPU: %.1f ms\n", cpu_ms);


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
        blocktiling_2d_float4transposed<<<blocks,threads>>>(
            A_num_fil, A_num_col, d_A,
            B_num_fil, B_num_col, d_B, d_C
        );
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
        blocktiling_2d_float4transposed<<<blocks,threads>>>(A_num_fil, A_num_col, d_A, B_num_fil, B_num_col, d_B, d_C);
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


    report("Float4 transposed + registrob", times, N_ITER, flops);
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