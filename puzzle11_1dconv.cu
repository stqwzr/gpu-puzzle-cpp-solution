#include <stdio.h>

const int MAX_CONV = 4;
const int TPB = 8;
const int TPB_MAX_CONV = TPB + MAX_CONV;

__global__ void conv_kernel(float *input, float *kernel, float *output, int size, int kernel_size) {
    __shared__ float a_shared[TPB_MAX_CONV];
    __shared__ float kernel_shared[MAX_CONV];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int local_i = threadIdx.x;

    if (i < size) {
        a_shared[local_i] = input[i];
    }
    if (local_i < kernel_size - 1 && i + TPB < size) {
        a_shared[local_i + TPB] = input[i + TPB];
    }

    if (local_i < kernel_size) {
        kernel_shared[local_i] = kernel[local_i];
    }

    __syncthreads();

    if (i < size) {
        float acc = 0.0;
        for (int j = 0; j < kernel_size; j++) {
            if (i + j < size) {
                acc += a_shared[local_i + j] * kernel_shared[j];
            }
        }
        output[i] = acc;
    }
}

int main() {
    const int SIZE = 15;
    const int CONV = 4;
    float h_a[SIZE] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    float h_b[CONV] = {0, 1, 2, 3};
    float h_out[SIZE] = {0};

    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, sizeof(float) * SIZE);
    cudaMalloc(&d_b, sizeof(float) * CONV);
    cudaMalloc(&d_out, sizeof(float) * SIZE);

    cudaMemcpy(d_a, h_a, sizeof(float) * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * CONV, cudaMemcpyHostToDevice);

    conv_kernel<<<2, TPB>>>(d_a, d_b, d_out, SIZE, CONV);

    cudaError_t err = cudaGetLastError();
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, sizeof(float) * SIZE, cudaMemcpyDeviceToHost);

    printf("Output: ");
    for (int i = 0; i < SIZE; i++) printf("%.1f ", h_out[i]);
    printf("\n");


    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    return 0;
}