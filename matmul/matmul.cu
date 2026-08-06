%%writefile matmul.cu
#include <cstdio>
#include <random>
#include <chrono>

using namespace std;

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<float> dist(-2.0, 2.0);

__global__ void matmul(float *mat1, float *mat2, float *result, int M_filas_1, int K_col_1, int M_filas_2, int K_col_2, int N){

    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N){

        int fil_num_1 = i/K_col_2;
        int col_num_1 = i - fil_num_1 * K_col_2;
        float sum = 0.0f;

        for (int j = fil_num_1 * K_col_1; j < K_col_1*fil_num_1 + K_col_1; j++){
            
            sum = sum + mat1[j] * mat2[col_num_1 + ((j - fil_num_1*K_col_1)* (K_col_2))];

    }
        result[i] = sum;
    }
}

int main(){

    int M_fil_1 = 1024;
    int K_col_1 = 2048;
    int M_fil_2 = 2048;
    int K_col_2 = 1024;
    int N = M_fil_1 * K_col_2;

    size_t bytes_n = N * sizeof(float);
    size_t bytes_m1 = M_fil_1*K_col_1*sizeof(float);
    size_t bytes_m2 = M_fil_2*K_col_2*sizeof(float);

    float *h_mat1 = (float*)malloc(bytes_m1);
    float *h_mat2 = (float*)malloc(bytes_m2);
    float *h_result = (float*)malloc(bytes_n);
    float *h_C_cpu = (float*)malloc(bytes_n);
    
    
    for(int i = 0; i < M_fil_1*K_col_1; i++){
        h_mat1[i] = dist(gen);
    }

    for(int i = 0; i < M_fil_2*K_col_2; i++){
        h_mat2[i] = dist(gen);
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < M_fil_1; i++) {
        for (int j = 0; j < K_col_2; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K_col_1; k++)
                sum += h_mat1[i*K_col_1 + k] * h_mat2[k*K_col_2 + j];
            h_C_cpu[i*K_col_2 + j] = sum;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("CPU: %.1f ms\n", cpu_ms);

    float *d_mat1;
    float *d_mat2;
    float *d_result;

    cudaMalloc(&d_mat1, bytes_m1);
    cudaMalloc(&d_mat2, bytes_m2);
    cudaMalloc(&d_result, bytes_n);

    cudaMemcpy(d_mat1, h_mat1, bytes_m1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, h_mat2, bytes_m2, cudaMemcpyHostToDevice);

    
    int threads = 1024;
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < 3; i++)
        matmul<<<blocks,threads>>>(d_mat1, d_mat2, d_result, M_fil_1, K_col_1, M_fil_2, K_col_2, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < 20; i++)
        matmul<<<blocks,threads>>>(d_mat1, d_mat2, d_result, M_fil_1, K_col_1, M_fil_2, K_col_2, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    gpu_ms /= 20.0f;

    double gflops = 2.0 * M_fil_1 * K_col_2 * K_col_1 / (gpu_ms / 1000.0) / 1e9;
    printf("GPU: %.3f ms  (%.1f GFLOP/s)\n", gpu_ms, gflops);
    printf("Speedup: %.1fx\n", cpu_ms / gpu_ms);


    cudaMemcpy(h_result, d_result, bytes_n, cudaMemcpyDeviceToHost);

    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_result);

    return 0;
}