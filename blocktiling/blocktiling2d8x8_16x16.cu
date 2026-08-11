#include <cstdio>
#include <random>
#include <chrono>

constexpr int tile_size = 16;

using namespace std;

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<float> dist(-2.0, 2.0);

__global__ void tiled_matmul_2d(int A_num_fil, int A_num_col,float *A, int B_num_fil, int B_num_col,float *B, float *C){

    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;



    int threadIdx_r4_1 = ((threadIdx.x/4) + threadIdx.y*2);
    int threadIdxy4_1 = (threadIdx.x*4 - (threadIdx_r4_1%2)*16) % 16;
    int threadIdx_r4_2 = ((threadIdx.x/4) + threadIdx.y*2);
    int threadIdxy4_2 = (threadIdx.x*4 - (threadIdx_r4_2%2)*16 + 1) % 16;
    int threadIdx_r4_3 = ((threadIdx.x/4) + threadIdx.y*2);
    int threadIdxy4_3 = (threadIdx.x*4 - (threadIdx_r4_3%2)*16 + 2) % 16;
    int threadIdx_r4_4 = ((threadIdx.x/4) + threadIdx.y*2);
    int threadIdxy4_4 = (threadIdx.x*4 - (threadIdx_r4_4%2)*16 + 3) % 16;

    int row_pos1 = blockIdx.y * tile_size + threadIdx_r4_1;
    int row_pos2 = blockIdx.y * tile_size + threadIdx_r4_2;
    int row_pos3 = blockIdx.y * tile_size + threadIdx_r4_3;
    int row_pos4 = blockIdx.y * tile_size + threadIdx_r4_4;

    int col_pos1 = blockIdx.x * tile_size + threadIdxy4_1;
    int col_pos2 = blockIdx.x * tile_size + threadIdxy4_2;
    int col_pos3 = blockIdx.x * tile_size + threadIdxy4_3;
    int col_pos4 = blockIdx.x * tile_size + threadIdxy4_4;

    __shared__ float shared_memory_1[tile_size][tile_size];
    __shared__ float shared_memory_2[tile_size][tile_size];

    for (int i = 0; i < (A_num_col + tile_size - 1)/tile_size; i ++){
        shared_memory_1[threadIdx_r4_1][threadIdxy4_1] = (row_pos1 < A_num_fil && col_pos1 < A_num_col) ? A[row_pos1*A_num_col + threadIdxy4_1 + i*tile_size] : 0.0f;
        shared_memory_1[threadIdx_r4_2][threadIdxy4_2] = (row_pos2 < A_num_fil && col_pos2 < A_num_col) ? A[row_pos2*A_num_col + threadIdxy4_2 + i*tile_size] : 0.0f;
        shared_memory_1[threadIdx_r4_3][threadIdxy4_3] = (row_pos3 < A_num_fil && col_pos3 < A_num_col) ? A[row_pos3*A_num_col + threadIdxy4_3 + i*tile_size] : 0.0f;
        shared_memory_1[threadIdx_r4_4][threadIdxy4_4] = (row_pos4 < A_num_fil && col_pos4 < A_num_col) ? A[row_pos4*A_num_col + threadIdxy4_4 + i*tile_size] : 0.0f;

        
        shared_memory_2[threadIdx_r4_1][threadIdxy4_1] = (i*tile_size + threadIdx_r4_1 < B_num_fil && col_pos1 < B_num_col) ? B[(i*tile_size + threadIdx_r4_1)*B_num_col + col_pos1] : 0.0f;
        shared_memory_2[threadIdx_r4_2][threadIdxy4_2] = (i*tile_size + threadIdx_r4_2 < B_num_fil && col_pos2 < B_num_col) ? B[(i*tile_size + threadIdx_r4_2)*B_num_col + col_pos2] : 0.0f;
        shared_memory_2[threadIdx_r4_3][threadIdxy4_3] =(i*tile_size + threadIdx_r4_3 < B_num_fil && col_pos3 < B_num_col)? B[(i*tile_size + threadIdx_r4_3)*B_num_col + col_pos3]: 0.0f;
        shared_memory_2[threadIdx_r4_4][threadIdxy4_4] =(i*tile_size + threadIdx_r4_4 < B_num_fil && col_pos4 < B_num_col)? B[(i*tile_size + threadIdx_r4_4)*B_num_col + col_pos4]: 0.0f;

        __syncthreads();
        int local_col_pos = (threadIdx.x + (threadIdx.y%2)*8);
        int local_row_pos = (threadIdx.y/2) * 4;

        for (int j = 0; j < tile_size; j++){
            sum1 = sum1 + shared_memory_1[local_row_pos][j] * shared_memory_2[j][local_col_pos];
            sum2 = sum2 + shared_memory_1[local_row_pos + 1][j] * shared_memory_2[j][local_col_pos];
            sum3 = sum3 + shared_memory_1[local_row_pos + 2][j] * shared_memory_2[j][local_col_pos];
            sum4 = sum4 + shared_memory_1[local_row_pos + 3][j] * shared_memory_2[j][local_col_pos];
        }

        __syncthreads();

    }

    int local_col_pos = (threadIdx.x + (threadIdx.y%2)*8);
    int local_row_pos = (threadIdx.y/2) * 4;
    int row_pos = blockIdx.y * tile_size + local_row_pos;
    int col_pos = blockIdx.x * tile_size + local_col_pos;


    if (row_pos < A_num_fil && col_pos < B_num_col){
        C[row_pos*B_num_col + col_pos] = sum1;
    } 
    if (row_pos + 1 < A_num_fil && col_pos < B_num_col){
        C[(row_pos+1)*B_num_col + col_pos] = sum2;
    } 

    if (row_pos + 2 < A_num_fil && col_pos < B_num_col){
        C[(row_pos+2)*B_num_col + col_pos] = sum3;
    } 

    if (row_pos + 3 < A_num_fil && col_pos < B_num_col){
        C[(row_pos+3)*B_num_col + col_pos] = sum4;
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

    cudaMalloc(&d_A, bytes_AB);
    cudaMalloc(&d_B, bytes_AB);
    cudaMalloc(&d_C, bytes_C);

    cudaMemcpy(d_A, h_A, bytes_AB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_AB, cudaMemcpyHostToDevice);

    dim3 threads(8,8);
    dim3 blocks(
        (B_num_col + threads.x - 1)/tile_size,
        (A_num_fil + threads.y - 1)/tile_size
    );

    for (int i = 0; i < 3; i++)
        tiled_matmul_2d<<<blocks,threads>>>(
            A_num_fil, A_num_col, d_A,
            B_num_fil, B_num_col, d_B, d_C
        );

    cudaDeviceSynchronize();


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < 20; i++)
        tiled_matmul_2d<<<blocks,threads>>>(
            A_num_fil, A_num_col, d_A,
            B_num_fil, B_num_col, d_B, d_C
        );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    gpu_ms /= 20.0f;

    double gflops =
        2.0 * A_num_fil * B_num_col * A_num_col
        / (gpu_ms / 1000.0)
        / 1e9;

    printf("GPU: %.3f ms  (%.1f GFLOP/s)\n", gpu_ms, gflops);
    printf("Speedup: %.1fx\n", cpu_ms / gpu_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);

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