#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

extern "C"
{
#include "nn.h"
}

// Should be a multiple of the warp size
#define BLOCK_SIZE 32
__device__ int d_error_flag = 0;

typedef struct DeviceNeuralNetworkPointers_
{
	size_t *d_sizes;
	size_t *d_weight_offsets;
	size_t *d_bias_offsets;
	ActivationFunction *d_activations;
	real *d_weights;
	real *d_biases;
} DeviceNeuralNetworkPointers;

typedef struct DeviceScratchSpacePointers_
{
	real *d_activations;
	real *d_d_weights;
	real *d_d_bias;
	real *d_d_activations;
} DeviceScratchSpacePointers;

typedef struct DeviceNeuralNetworkDataPointers_
{
	real *d_input;
	real *d_output;
} DeviceNeuralNetworkDataPointers;

typedef struct DeviceRmsPropPointers_
{
	real *d_ew;
	real *d_eb;
} DeviceRmsPropPointers;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Neural Network
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
NeuralNetwork *allocate_device_neural_network(NeuralNetwork *h_nn, DeviceNeuralNetworkPointers *d_nnp)
{
	NeuralNetwork *d_nn;
	cudaMalloc((void **)&d_nn, sizeof(NeuralNetwork));

	// Copy non-pointer data directly
	cudaMemcpy(d_nn, h_nn, sizeof(NeuralNetwork), cudaMemcpyHostToDevice);

	cudaMalloc((void **)&d_nnp->d_sizes, h_nn->n * sizeof(size_t));

	cudaMalloc((void **)&d_nnp->d_activations, h_nn->n * sizeof(ActivationFunction));
	size_t total_weights_size = get_weights_size(h_nn);
	size_t total_biases_size = get_biases_size(h_nn);

	cudaMalloc((void **)&d_nnp->d_weights, total_weights_size * sizeof(real));

	cudaMalloc((void **)&d_nnp->d_biases, total_biases_size * sizeof(real));

	cudaMemcpy(&(d_nn->sizes), &d_nnp->d_sizes, sizeof(size_t *), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_nn->activations), &d_nnp->d_activations, sizeof(ActivationFunction *), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_nn->weights), &d_nnp->d_weights, sizeof(real *), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_nn->biases), &d_nnp->d_biases, sizeof(real *), cudaMemcpyHostToDevice);

	return d_nn;
}

void copy_neural_network_to_device(NeuralNetwork *h_nn, NeuralNetwork *d_nn, DeviceNeuralNetworkPointers *d_nnp)
{
	size_t total_weights_size = get_weights_size(h_nn);
	size_t total_biases_size = get_biases_size(h_nn);

	cudaMemcpy(d_nnp->d_sizes, h_nn->sizes, h_nn->n * sizeof(size_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nnp->d_activations, h_nn->activations, h_nn->n * sizeof(ActivationFunction), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nnp->d_weights, h_nn->weights, total_weights_size * sizeof(real), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nnp->d_biases, h_nn->biases, total_biases_size * sizeof(real), cudaMemcpyHostToDevice);
}

void copy_neural_network_to_host(NeuralNetwork *d_nn, NeuralNetwork *h_nn, DeviceNeuralNetworkPointers *d_nnp)
{
	size_t total_weights_size = get_weights_size(h_nn);
	size_t total_biases_size = get_biases_size(h_nn);

	cudaMemcpy(h_nn->sizes, d_nnp->d_sizes, h_nn->n * sizeof(size_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_nn->activations, d_nnp->d_activations, h_nn->n * sizeof(ActivationFunction), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_nn->weights, d_nnp->d_weights, total_weights_size * sizeof(real), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_nn->biases, d_nnp->d_biases, total_biases_size * sizeof(real), cudaMemcpyDeviceToHost);
}

void free_device_neural_network(NeuralNetwork *d_nn, DeviceNeuralNetworkPointers *d_nnp)
{
	cudaFree(d_nnp->d_sizes);
	cudaFree(d_nnp->d_activations);
	cudaFree(d_nnp->d_weights);
	cudaFree(d_nnp->d_biases);
	cudaFree(d_nn);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Scratch space
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

ScratchSpace *allocate_device_scratch_space(NeuralNetwork *nn, ScratchSpace *h_scratch,
											DeviceScratchSpacePointers *d_ssp)
{
	ScratchSpace *d_scratch;
	cudaError_t err;

	err = cudaMalloc((void **)&d_scratch, sizeof(ScratchSpace));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device scratch space (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	size_t activations_size_bytes = h_scratch->n * h_scratch->activations_size * sizeof(real);
	size_t weights_size_bytes = h_scratch->n * get_weights_size(nn) * sizeof(real);
	size_t biases_size_bytes = h_scratch->n * get_biases_size(nn) * sizeof(real);

	printf("Allocating Scratch Space: activations %zu bytes, weights %zu bytes, biases %zu bytes\n",
		   activations_size_bytes, weights_size_bytes, biases_size_bytes);

	err = cudaMalloc((void **)&d_ssp->d_activations, activations_size_bytes);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device activations (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **)&d_ssp->d_d_activations, activations_size_bytes);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device double activations (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **)&d_ssp->d_d_weights, weights_size_bytes);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device weights (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **)&d_ssp->d_d_bias, biases_size_bytes);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device biases (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaMemcpy(&(d_scratch->d_activations), &d_ssp->d_d_activations, sizeof(real *), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_scratch->activations), &d_ssp->d_activations, sizeof(real *), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_scratch->d_weights), &d_ssp->d_d_weights, sizeof(real *), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_scratch->d_bias), &d_ssp->d_d_bias, sizeof(real *), cudaMemcpyHostToDevice);

	return d_scratch;
}

void copy_scratch_space_to_device(NeuralNetwork *nn, ScratchSpace *h_scratch, ScratchSpace *d_scratch,
								  DeviceScratchSpacePointers *d_ssp)
{
	size_t activations_size_bytes = h_scratch->n * h_scratch->activations_size * sizeof(real);
	size_t weights_size_bytes = h_scratch->n * get_weights_size(nn) * sizeof(real);
	size_t biases_size_bytes = h_scratch->n * get_biases_size(nn) * sizeof(real);

	cudaMemcpy(d_ssp->d_d_activations, h_scratch->d_activations, activations_size_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ssp->d_activations, h_scratch->activations, activations_size_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ssp->d_d_weights, h_scratch->d_weights, weights_size_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ssp->d_d_bias, h_scratch->d_bias, biases_size_bytes, cudaMemcpyHostToDevice);
}

void copy_scratch_space_to_host(NeuralNetwork *nn, ScratchSpace *d_scratch, ScratchSpace *h_scratch,
								DeviceScratchSpacePointers *d_ssp)
{
	size_t activations_size_bytes = h_scratch->n * h_scratch->activations_size * sizeof(real);
	size_t weights_size_bytes = h_scratch->n * get_weights_size(nn) * sizeof(real);
	size_t biases_size_bytes = h_scratch->n * get_biases_size(nn) * sizeof(real);

	cudaMemcpy(h_scratch->d_activations, d_ssp->d_d_activations, activations_size_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_scratch->activations, d_ssp->d_activations, activations_size_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_scratch->d_weights, d_ssp->d_d_weights, weights_size_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_scratch->d_bias, d_ssp->d_d_bias, biases_size_bytes, cudaMemcpyDeviceToHost);
}

void free_device_scratch_space(ScratchSpace *d_scratch, DeviceScratchSpacePointers *d_ssp)
{
	cudaFree(d_ssp->d_activations);
	cudaFree(d_ssp->d_d_weights);
	cudaFree(d_ssp->d_d_bias);
	cudaFree(d_scratch);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Data
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

NeuralNetworkData *allocate_device_neural_network_data(NeuralNetworkData *h_data,
													   DeviceNeuralNetworkDataPointers *d_dnp)
{
	NeuralNetworkData *d_data;
	cudaMalloc((void **)&d_data, sizeof(NeuralNetworkData));

	size_t input_size = h_data->n * h_data->input_dimension * sizeof(real);
	size_t output_size = h_data->n * h_data->output_dimension * sizeof(real);

	cudaMalloc((void **)&d_dnp->d_input, input_size);
	cudaMalloc((void **)&d_dnp->d_output, output_size);

	cudaMemcpy(&(d_data->input), &d_dnp->d_input, sizeof(real *), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_data->output), &d_dnp->d_output, sizeof(real *), cudaMemcpyHostToDevice);

	return d_data;
}

void copy_neural_network_data_to_device(NeuralNetworkData *h_data, NeuralNetworkData *d_data,
										DeviceNeuralNetworkDataPointers *d_dnp)
{
	size_t input_size = h_data->n * h_data->input_dimension * sizeof(real);
	size_t output_size = h_data->n * h_data->output_dimension * sizeof(real);

	cudaMemcpy(d_dnp->d_input, h_data->input, input_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dnp->d_output, h_data->output, output_size, cudaMemcpyHostToDevice);
}

void copy_neural_network_data_to_host(NeuralNetworkData *d_data, NeuralNetworkData *h_data,
									  DeviceNeuralNetworkDataPointers *d_dnp)
{
	size_t input_size = h_data->n * h_data->input_dimension * sizeof(real);
	size_t output_size = h_data->n * h_data->output_dimension * sizeof(real);

	cudaMemcpy(h_data->input, d_dnp->d_input, input_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_data->output, d_dnp->d_output, output_size, cudaMemcpyDeviceToHost);
}

void free_device_neural_network_data(NeuralNetworkData *d_data, DeviceNeuralNetworkDataPointers *d_dnp)
{

	cudaFree(d_dnp->d_input);
	cudaFree(d_dnp->d_output);
	cudaFree(d_data);
}

__global__ void gpu_forward_pass(NeuralNetwork *nn, NeuralNetworkData *data, ScratchSpace *scratch)
{
	size_t activation_index = threadIdx.x;
	size_t scratch_index = 0;
	size_t data_index = 0;

	real *current_output = scratch->activations + scratch_index * scratch->activations_size;
	real *current_biases = nn->biases;
	real *current_weights = nn->weights;
	real *current_inputs = data->input + data_index * nn->sizes[0];

	if (activation_index < nn->sizes[0])
	{
		current_output[activation_index] = current_inputs[activation_index];
	}

	current_inputs = current_output;
	current_output += nn->sizes[0];

	__syncthreads();

	for (size_t layer = 0; layer < nn->n - 1; layer++)
	{
		size_t result_size = nn->sizes[layer + 1]; // nrows
		size_t inputs_size = nn->sizes[layer];     // ncols

		if (activation_index < nn->sizes[layer + 1])
		{

			real activation = current_biases[activation_index];
			real *weight_row = current_weights + activation_index * inputs_size;
			for (size_t i = 0; i < inputs_size; i++)
			{
				activation += weight_row[i] * current_inputs[i];
			}

			switch (nn->activations[layer])
			{
			case RELU:
				if (activation < 0)
				{
					activation = 0;
				}
				break;
			case LEAKY_RELU:
				if (activation < 0)
				{
					activation = activation * LEAKY_RELU_SLOPE;
				}
				break;
			default:
				atomicExch(&d_error_flag, 1);
				return;
			}

			current_output[activation_index] = activation;
		}

		__syncthreads();

		current_biases += result_size;
		current_inputs += inputs_size;
		current_output += result_size;
		current_weights += inputs_size * result_size;
	}
}

int main()
{
	// extract this to common function
	size_t n = 1;
	size_t num_layers = 3;
	size_t sizes[] = {2, 4, 3};

	NeuralNetwork *nn = create_neural_network(num_layers, sizes);
	ScratchSpace *scratch = create_scratchspace(nn, n);
	NeuralNetworkData *data = create_random_input(sizes[0], sizes[num_layers - 1], n);

	for (size_t i = 0; i < nn->n; i++)
	{
		if (nn->sizes[i] > BLOCK_SIZE)
		{
			printf("Error: layer size is greater than block size\n");
			return EXIT_FAILURE;
		}
	}

	for (size_t i = 0; i < nn->n; i++)
	{
		nn->activations[i] = RELU;
	}
	for (size_t i = 0; i < sizes[0] * sizes[1] + sizes[1] * sizes[2]; i++)
	{
		nn->weights[i] = 0.1 * i;
	}
	for (size_t i = 0; i < 7; i++)
	{
		nn->biases[i] = 0.1 * (6 - i + 1);
	}
	data->input[0] = 0.1;
	data->input[1] = 0.2;
	data->output[0] = 1;
	data->output[1] = 2;
	data->output[2] = 3;

	DeviceNeuralNetworkPointers d_nnp;
	DeviceScratchSpacePointers d_ssp;
	DeviceNeuralNetworkDataPointers d_dnp;
	NeuralNetwork *d_nn = allocate_device_neural_network(nn, &d_nnp);
	ScratchSpace *d_scratch = allocate_device_scratch_space(nn, scratch, &d_ssp);
	NeuralNetworkData *d_data = allocate_device_neural_network_data(data, &d_dnp);

	cudaDeviceSynchronize();

	copy_neural_network_to_device(nn, d_nn, &d_nnp);
	copy_scratch_space_to_device(nn, scratch, d_scratch, &d_ssp);
	copy_neural_network_data_to_device(data, d_data, &d_dnp);

	/*
	for (size_t layer = 0; layer < nn->n - 1; layer++) {
		gpu_forward_pass_layer<<<dimGrid, dimBlock>>>(d_nn, d_data, d_scratch, layer);
		cudaDeviceSynchronize();
	}*/
	dim3 dimBlock(BLOCK_SIZE, 1);
	dim3 dimGrid(1, 1);
	gpu_forward_pass<<<dimGrid, dimBlock>>>(d_nn, d_data, d_scratch);

	int h_error_flag;
	cudaMemcpyFromSymbol(&h_error_flag, d_error_flag, sizeof(int));
	if (h_error_flag != 0)
	{
		// Handle the error appropriately
		printf("Error occurred in the CUDA kernel\n");
	}

	printf("CPU Forward Pass\n");
	forward_pass(nn, scratch, data, 0, 0);
	print_scratchspace(nn, scratch);

	copy_neural_network_data_to_host(d_data, data, &d_dnp);
	copy_scratch_space_to_host(nn, d_scratch, scratch, &d_ssp);
	copy_neural_network_to_host(d_nn, nn, &d_nnp);

	printf("====================================\n");
	printf("GPU Forward Pass\n");
	print_scratchspace(nn, scratch);

	free_device_neural_network_data(d_data, &d_dnp);
	free_device_scratch_space(d_scratch, &d_ssp);
	free_device_neural_network(d_nn, &d_nnp);

	free_neural_network_data(data);
	free_scratchspace(scratch);
	free_neural_network(nn);

	return EXIT_SUCCESS;
}
