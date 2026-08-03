#include <cstdio>


__global__ void vectorAddition(float *A, float *B, float *C, int N){

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) C[i] = A[i] + B[i];
}

int main(){
    int N = 1000000;
    size_t bytes = N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < N; i ++){
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A;
    float *d_B;
    float *d_C;

    cudaMalloc(&d_A, N* sizeof(float));
    cudaMalloc(&d_B, N* sizeof(float));
    cudaMalloc(&d_C, N* sizeof(float));

    cudaMemcpy(d_A, h_A, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*sizeof(float), cudaMemcpyHostToDevice);

    int threads = 1024;
    int num_Blocks = (N + threads - 1) / threads;
    vectorAddition<<<num_Blocks, threads>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    for(int i = 0; i < N; i ++) printf("%f\n", h_C[i]);
    return 0;

}