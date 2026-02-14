#include <stdio.h>

__global__ void block(float *a, float *out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        out[i] = a[i] + 10;
        printf("%f %d\n", out[i], i);
    }
}


int main() {
    const int SIZE = 9;
    float h_a[SIZE] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    float h_out[SIZE] = {0};

    float *d_a, *d_out;
    cudaMalloc(&d_a, sizeof(float) * SIZE);
    cudaMalloc(&d_out, sizeof(float) * SIZE);

    cudaMemcpy(d_a, h_a, sizeof(float) * SIZE, cudaMemcpyHostToDevice);

    block<<<3, 4>>>(d_a, d_out, SIZE);

    cudaError_t err = cudaGetLastError();
    printf("CUDA error: %s\n", cudaGetErrorString(err));

    cudaMemcpy(h_out, d_out, sizeof(float) * SIZE, cudaMemcpyDeviceToHost);

    for (int i = 0; i < SIZE; i++) {
        printf("a[%d] = %f\n", i, h_out[i]);
    }
    cudaFree(d_a);
    cudaFree(d_out);
    return 0;
}
