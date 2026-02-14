#include <stdio.h>

__global__ void dot_kernel(float *a, float *b, float *out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float shared[8];

    if (i < size) {
        shared[i] = a[i] * b[i];
    }

    __syncthreads();

    if (i == 0) {
        float acc = 0;
        for (int j = 0; j < size; j++) {
            acc += shared[j];
        }
        out[0] = acc;
    }
}

int main() {
    const int SIZE = 8;
    float h_a[SIZE] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    float h_b[SIZE] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    float h_out[SIZE];

    float *d_a, *d_b, *d_out;

    cudaMalloc((void**)&d_a, sizeof(float) * SIZE);
    cudaMalloc((void**)&d_b, sizeof(float) * SIZE);
    cudaMalloc((void**)&d_out, sizeof(float) * SIZE);

    cudaMemcpy(d_b, h_b, sizeof(float) * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, h_a, sizeof(float) * SIZE, cudaMemcpyHostToDevice);

    dot_kernel<<<1, SIZE>>>(d_a, d_b, d_out, SIZE);

    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("%f\n", h_out[0]);

    cudaFree(d_b);
    cudaFree(d_a);
    cudaFree(d_out);
    return 0;
}
