#include <cstdio>
#include <random>
#include <chrono>

using namespace std;

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<float> dist(-2.0, 2.0);

__global__ void matmul_2d(int A_num_fil, int A_num_col,float *A, int B_num_fil, int B_num_col,float *B, float *C){

    int row_pos = blockIdx.y * blockDim.y + threadIdx.y;
    int col_pos = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;

    if(row_pos < A_num_fil && col_pos < B_num_col){
        for (int i = 0; i < A_num_col; i ++){
            sum = sum + A[row_pos * A_num_col + i] * B[col_pos + i*B_num_col];
        }
        
        C[row_pos*B_num_col + col_pos] = sum;
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
        printf("%f\n",h_C[i]);
    }

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

    dim3 threads(16,16);
    dim3 blocks(
        (B_num_col + threads.x - 1)/threads.x,
        (A_num_fil + threads.y - 1)/threads.y
    );

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < 3; i++)
        matmul_2d<<<blocks,threads>>>(A_num_fil, A_num_col, d_A, B_num_fil, B_num_col, d_B, d_C);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < 20; i++)
        matmul_2d<<<blocks,threads>>>(A_num_fil, A_num_col, d_A, B_num_fil, B_num_col, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    gpu_ms /= 20.0f;

    double gflops = 2.0 * A_num_fil * B_num_col * A_num_col / (gpu_ms / 1000.0) / 1e9;
    printf("GPU: %.3f ms  (%.1f GFLOP/s)\n", gpu_ms, gflops);
    printf("Speedup: %.1fx\n", cpu_ms / gpu_ms);

    
    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    for(int i = 0; i < 5; i ++){
        printf("%f\n",h_C[i]);
    }

    return 0;

}