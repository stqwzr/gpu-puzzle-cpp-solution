#include <stdio.h>

__global__ void broadcast(float *d_a, float *d_b, float *out, int size) {
    int i = threadIdx.x;
    int j = threadIdx.y;

    if (i < size && j < size) {
        out[i * size + j] = d_a[i] + d_b[j];
    }
}

int main(int argc, char **argv) {
    const int SIZE = 2;
    float h_a[1][SIZE] = {{1, 2}};
    float h_b[SIZE][1] = {{1}, {2}};
    float h_out[SIZE * SIZE] = {{0}};

    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, sizeof(float) * 1 * SIZE);
    cudaMalloc(&d_b, sizeof(float) * SIZE * 1);
    cudaMalloc(&d_out, sizeof(float) * SIZE * SIZE);

    cudaMemcpy(d_a, h_a, sizeof(float) * 1 * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * SIZE * 1, cudaMemcpyHostToDevice);

    broadcast<<<1, dim3(3, 3)>>>(d_a, d_b, d_out, SIZE);

    cudaError_t err = cudaGetLastError();
    printf("CUDA error: %s\n", cudaGetErrorString(err));

    cudaMemcpy(h_out, d_out, sizeof(float) * SIZE * SIZE, cudaMemcpyDeviceToHost);
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            printf("%f ", h_out[i * SIZE + j]);
        }
    }


    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}
