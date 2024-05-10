#include <stdio.h>
#include <cuda_runtime.h>

__global__ void sumArray(float *in, float *out, int N);

int main() {
    int N = 1024; // Size of the array
    size_t bytes = N * sizeof(float);

    float *h_in = (float*) malloc(bytes);
    float *h_out = (float*) malloc(sizeof(float)); // Output variable on host
    float *d_in, *d_out; // Device input and output pointers

    // Initialize input array with some values
    for(int i = 0; i < N; i++) {
        h_in[i] = 2.0f;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_in, bytes);
    cudaMalloc((void**)&d_out, sizeof(float));

    // Transfer data from host to device
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // Kernel launch
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    sumArray<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N);

    // Copy result back to host
    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    printf("Result: %f\n", *h_out);

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_out);

    // Free host memory
    free(h_in);
    free(h_out);

    return 0;
}

__global__ void sumArray(float *in, float *out, int N) {
    // Your implementation here
}
