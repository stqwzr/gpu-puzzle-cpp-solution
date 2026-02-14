#include <stdio.h>

__global__ void block(float *a, float *out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < size && j < size) {
        out[i * size + j] = a[i * size + j] + 10;
        printf("%f %d\n", out[i * size + j], i * size + j);
    }
}


int main() {
    const int SIZE = 5;
    float h_a[SIZE][SIZE] = {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
    float h_out[SIZE][SIZE] = {{0}};

    float *d_a, *d_out;
    cudaMalloc(&d_a, sizeof(float) * SIZE * SIZE);
    cudaMalloc(&d_out, sizeof(float) * SIZE * SIZE);

    cudaMemcpy(d_a, h_a, sizeof(float) * SIZE * SIZE, cudaMemcpyHostToDevice);

    block<<<dim3(2, 2), dim3(3, 3)>>>(d_a, d_out, SIZE);

    cudaError_t err = cudaGetLastError();
    printf("CUDA error: %s\n", cudaGetErrorString(err));

    cudaMemcpy(h_out, d_out, sizeof(float) * SIZE * SIZE, cudaMemcpyDeviceToHost);

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            printf("a[%d][%d] = %f\n", i, j, h_out[i][j]);
        }
    }
    cudaFree(d_a);
    cudaFree(d_out);
    return 0;
}
