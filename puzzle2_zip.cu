#include <stdio.h>

__global__ void zip_kernel(float* out, float* a, float* b, int size) {
    int i = threadIdx.x;
    if (i < size) {
        out[i] = a[i] + b[i];
    }
}

int main(int argc, char** argv) {
    const int SIZE = 4;
    float h_a[SIZE] = {1.0, 2.0, 3.0, 4.0};
    float h_b[SIZE] = {5.0, 6.0, 7.0, 8.0};
    float h_out[SIZE] = {0};

    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, SIZE * sizeof(float));
    cudaMalloc(&d_b, SIZE * sizeof(float));
    cudaMalloc(&d_out, SIZE * sizeof(float));

    cudaMemcpy(d_a, h_a, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    zip_kernel<<<1, 10>>>(d_out, d_a, d_b, SIZE);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    printf("CUDA error: %s\n", cudaGetErrorString(err));

    cudaMemcpy(h_out, d_out, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output: ");
    for (int i = 0; i < SIZE; i++) printf("%.1f ", h_out[i]);
    printf("\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return 0;
}