#include <cstdio>

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

    int A_num_fil = 2000;
    int A_num_col = 4000;
    int B_num_fil = 4000;
    int B_num_col = 2000;
    int N_A_B = A_num_fil*A_num_col;
    int N_C = A_num_fil*B_num_col;

    size_t bytes_AB = N_A_B * sizeof(float);
    size_t bytes_C = N_C * sizeof(float);

    float *h_A = (float*)malloc(bytes_AB);
    float *h_B = (float*)malloc(bytes_AB);
    float *h_C = (float*)malloc(bytes_C);

    for (int i = 0; i < N_A_B; i ++){
        h_A[i] = 1.0;
        h_B[i] = 2.0;
    }
    
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

    matmul_2d<<<blocks,threads>>>(A_num_fil, A_num_col, d_A, B_num_fil, B_num_col, d_B, d_C);
    
    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    for(int i = 0; i < 5; i ++){
        printf("%f\n",h_C[i]);
    }

    return 0;

}