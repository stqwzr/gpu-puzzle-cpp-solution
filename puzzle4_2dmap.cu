#include <stdio.h>

__global__ void map_kernel(float *out, float *a, int size) {
    int i = threadIdx.x;
    int j = threadIdx.y;

    if (i < size && j < size) {
        out[i * size + j] = a[i * size + j] + 10;
    }
}

int main() {
    const int SIZE = 2;
    float h_a[SIZE][SIZE] = {{0, 1}, {2, 3}};
    float h_out[SIZE * SIZE] = {0};

    float *d_a, *d_out;
    cudaMalloc(&d_a, sizeof(float) * SIZE * SIZE);
    cudaMalloc(&d_out, sizeof(float) * SIZE * SIZE);

    cudaMemcpy(d_a, h_a, sizeof(float) * SIZE * SIZE, cudaMemcpyHostToDevice);

    map_kernel<<<SIZE, dim3(2, 2)>>>(d_out, d_a, SIZE);

    cudaError_t err = cudaGetLastError();
    printf("CUDA error: %s\n", cudaGetErrorString(err));

    cudaMemcpy(h_out, d_out, sizeof(float) * SIZE * SIZE, cudaMemcpyDeviceToHost);

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            printf("a[%d][%d] = %f\n", i, j, h_out[i * SIZE + j]);
        }
    }
    cudaFree(d_a);
    cudaFree(d_out);
    return 0;
}
