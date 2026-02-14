#include <stdio.h>

__global__ void sm_kernel(float *a, float *out, int size) {
    __shared__ float shared[4];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int local_i = threadIdx.x;

    if (i < size) {
        shared[local_i] = a[i];
    }

    __syncthreads();

    if (i < size) {
        out[i] = shared[local_i] + 10;
    }
}

int main() {
    const int SIZE = 8;
    float h_a[SIZE] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float h_out[SIZE] = {0.0};

    float *d_a, *d_out;

    cudaMalloc(&d_a, sizeof(float) * SIZE);
    cudaMalloc(&d_out, sizeof(float) * SIZE);

    cudaMemcpy(d_a, h_a, sizeof(float) * SIZE, cudaMemcpyHostToDevice);

    sm_kernel<<<2, 4>>>(d_a, d_out, SIZE);
    cudaError_t err = cudaGetLastError();
    printf("CUDA error: %s\n", cudaGetErrorString(err));

    cudaMemcpy(h_out, d_out, sizeof(float) * SIZE, cudaMemcpyDeviceToHost);

    for (int i = 0; i < SIZE; i++) printf("%.1f ", h_out[i]);

    cudaFree(d_a);
    cudaFree(d_out);
    return 0;
}
