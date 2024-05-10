#include <stdio.h>
#include <cuda_runtime.h>


__global__ void sumArray(float *in, float *out, int N);
__global__ void cumulative_sum(int n, float *array);
__global__ void block_cumulative_sum(int n, float *array);

int main() {
	int N = 14; // Size of the array
	size_t bytes = N * sizeof(float);

	float *h_in = (float*) malloc(bytes);
	float *h_out = (float*) malloc(sizeof(float));
	float *d_in, *d_out; // Device input and output pointers


	// Allocate device memory
	cudaMalloc((void**)&d_in, bytes);
	cudaMalloc((void**)&d_out, sizeof(float));

	// sum array
	{
		// Initialize input array with some values
		for(int i = 0; i < N; i++) {
			h_in[i] = 2.0f;
		}

		// Transfer data from host to device
		cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

		dim3 dimBlock(256, 1);
		dim3 dimGrid(1, 1);
		sumArray<<<dimGrid, dimBlock>>>(d_in, d_out, N);

		// Copy result back to host
		cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

		// Print result
		printf("Result: %f\n", *h_out);
	}

	// cummulative sum
	{
		// re-initialize
		for(int i = 0; i < N; i++) {
			h_in[i] = 2.0f;
		}
		cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
		dim3 dimBlock(256, 1);
		dim3 dimGrid(1, 1);
		cumulative_sum<<<dimGrid, dimBlock>>>(N, d_in);
		cudaMemcpy(h_in, d_in, bytes, cudaMemcpyDeviceToHost);

		// Print result
		for (int i = 0; i < N; i++) {
			printf("%f ", h_in[i]);
		}
	}

	// Free device memory
	cudaFree(d_in);
	cudaFree(d_out);

	// Free host memory
	free(h_in);
	free(h_out);

	return 0;
}

__global__ void sumArray(float *in, float *out, int N) {
	size_t threadId = threadIdx.x;
	// TODO: parallelize this

	if (threadId == 0) {
		(*out) = 0.0f;
		for (int i = 0; i < N; i++) {
			(*out) += in[i];
		}
	}
}


__global__ void cumulative_sum(int n, float *array) {
	// cumulative sum. think divide and conquer:
	//   suppose the first and last half are already summed,
	//   then we just need to add the last element of the first half
	//   to the each element of the second half

	// . is an element of the array
	// p is the previous sum
	// the indices are the thread id's

	// n = 9, sub_array_size == 1
	// [.][.]|[.][.]|[.][.]|[.][.][.
	//  p  0   p  1   p  2   p  3  p

	// n = 9, sub_array_size == 2
	// [.  .][.  .]|[.  .][.  .]|[.
	//     p  0  1      p  2  3   

	// n = 9, sub_array_size == 4
	// [. . . .][. . . .]|[.
	//        p  0 1 2 3  

	// n = 9, sub_array_size == 8
	// [. . . . . . . .][.
	//                p  0

	unsigned int threadId = threadIdx.x;
	unsigned int sub_array_size = 1;
	while (sub_array_size < n) {
		if (sub_array_size != 1) {
			// wait for the previous sum to be calculated
			__syncthreads();
		}

		// divide the array into sub arrays of size sub_array_size
		// this is the index of the two arrays we will add
		unsigned int array_pair = 2 * sub_array_size * (threadId / sub_array_size);
		// this is the index of the sum calculated last iteration
		unsigned int previous_sum = array_pair + sub_array_size - 1;
		// this is where this thread will add the previous result to
		unsigned int destination = array_pair + sub_array_size + threadId % sub_array_size;
		if (destination < n) {
			array[destination] += array[previous_sum];
		}
		sub_array_size *= 2;
	}
}

__global__ void block_cumulative_sum(int n, float *array) {
	// TODO: support array sizes larger than a block size
}
