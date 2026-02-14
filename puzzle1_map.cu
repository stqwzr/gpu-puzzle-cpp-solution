#include <stdio.h>


__global__ void map_kernel(float* out, float* a, int size) {
    int i = threadIdx.x;

    if (i < size) {
        out[i] = a[i] + 10;
    }
}

int main() {

    const int SIZE = 4;
    float h_a[SIZE]   = {0.0, 1.0, 2.0, 3.0};
    float h_out[SIZE] = {0};

    float *d_a, *d_out;
    cudaMalloc(&d_a,   SIZE * sizeof(float));
    cudaMalloc(&d_out, SIZE * sizeof(float));

    cudaMemcpy(d_a, h_a, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    map_kernel<<<1, SIZE>>>(d_out, d_a, SIZE);

    cudaError_t err = cudaGetLastError();
    printf("CUDA error: %s\n", cudaGetErrorString(err));

    cudaMemcpy(h_out, d_out, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output: ");
    for (int i = 0; i < SIZE; i++) printf("%.1f ", h_out[i]);
    printf("\n");


    cudaFree(d_a);
    cudaFree(d_out);

    return 0;
}
