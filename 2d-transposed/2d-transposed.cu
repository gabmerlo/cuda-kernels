#include <cstdio>
constexpr int A_col = 40;
constexpr int B_col = 20;

__global__ void transposed(int A_fil,int A_col,float *mat1,int B_fil, int B_col,const float *mat2,float *result, int N){
    int row_pos = blockIdx.y * blockDim.y + threadIdx.y;
    int col_pos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row_pos < B_fil && col_pos < B_col){
        result[row_pos*B_col + col_pos] = mat1[col_pos*A_col + row_pos] + mat2[row_pos*B_col + col_pos];
    }
}

int main(){
    int N = 800;
    int A_fil = 20;
    int A_col = 40;
    int B_fil = 40;
    int B_col = 20;

    size_t bytes = N * sizeof(float);
    float *h_mat1 = (float*)malloc(bytes);
    float *h_mat2 = (float*)malloc(bytes);
    float *h_result = (float*)malloc(bytes);

    for (int i= 0; i < N; i++){
        h_mat1[i] = 1.0f;
        h_mat2[i] = 2.0;
    }

    float *d_mat1;
    float *d_mat2;
    float *d_result;

    cudaMalloc(&d_mat1, bytes);
    cudaMalloc(&d_mat2, bytes);
    cudaMalloc(&d_result, bytes);

    cudaMemcpy(d_mat1, h_mat1, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, h_mat2, bytes, cudaMemcpyHostToDevice);

    dim3 threads(16,16);
    dim3 blocks(
        (B_col + threads.x - 1)/threads.x,
        (B_fil + threads.y - 1)/threads.y
    );

    transposed<<<blocks,threads>>>(A_fil,A_col, d_mat1, B_fil, B_col, d_mat2, d_result, N);

    cudaMemcpy(h_result, d_result, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_result);

    for (int i = 0; i < 5; i ++){
        printf("%f\n",h_result[i]);
    }

    return 0;
        

}