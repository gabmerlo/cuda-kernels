%%writefile blocktiling-2d.cu
#include <cstdio>
#include <random>
#include <chrono>


constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 32;


constexpr int TM = 4;
constexpr int TN = 4;


constexpr int num_threads = (BM*BN) / (TM*TN);


using namespace std;

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<float> dist(-2.0, 2.0);

__global__ void blocktiling_2d(int A_num_fil, int A_num_col,float *A, int B_num_fil, int B_num_col,float *B, float *C){



}

int main(){

    int A_num_fil = 256;
    int A_num_col = 512;
    int B_num_fil = 512;
    int B_num_col = 256;
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

    dim3 threads(num_threads);
    dim3 blocks(
        (B_num_col + BN - 1)/BN,
        (A_num_fil + BM - 1)/BM
    );

    for (int i = 0; i < 3; i++)
        blocktiling_2d<<<blocks,threads>>>(
            A_num_fil, A_num_col, d_A,
            B_num_fil, B_num_col, d_B, d_C
        );

    cudaDeviceSynchronize();


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < 20; i++)
        blocktiling_2d<<<blocks,threads>>>(
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