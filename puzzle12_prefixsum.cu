#include <stdio.h>

const int TPB = 8;
const int BLOCK_SIZE = 2;

__global__ void prefix_sum(float *input, float *output, int size) {
    __shared__ float shared[TPB];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int local_i = threadIdx.x;

    if (i < size) {
        shared[local_i] = input[i];
    }
    __syncthreads();

    for (int stride = 1; stride < TPB; stride = stride * 2) {
        if ((local_i + 1) % (stride * 2) == 0) {
            shared[local_i] += shared[local_i - stride];
        }
        __syncthreads();
    }

    if (local_i == TPB - 1) {
        output[blockIdx.x] = shared[local_i];
    }
}

int main() {
    const int SIZE = 16;
    float h_a[SIZE] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    float h_out[BLOCK_SIZE] = {0, 0};

    float *d_a, *d_out;
    cudaMalloc(&d_a, sizeof(float) * SIZE);
    cudaMalloc(&d_out, sizeof(float) * SIZE);

    cudaMemcpy(d_a, h_a, sizeof(float) * SIZE, cudaMemcpyHostToDevice);

    prefix_sum<<<BLOCK_SIZE, TPB>>>(d_a, d_out, SIZE);

    cudaError_t err = cudaGetLastError();
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, sizeof(float) * BLOCK_SIZE, cudaMemcpyDeviceToHost);

    printf("Output: ");
    for (int i = 0; i < BLOCK_SIZE; i++) printf("%.1f ", h_out[i]);
    printf("\n");

    cudaFree(d_a);
    cudaFree(d_out);
    return 0;
}
