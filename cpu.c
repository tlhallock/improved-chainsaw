
#include "nn.h"

/*
	Homework 1:
		Implement device backward pass
		Implement device rms prop
		What are the warp sizes, block sizes, and grid sizes available on your machine?
		How much shared memory/core memory is available?
		https://docs.google.com/spreadsheets/d/1mHY2-utLnAx_iYw_hz-u_QJqAJiXqg8hP-tarb-lfBI/edit#gid=0

	Homework 3:
		Implement multiple cores per activation
			https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/
			the link about matrix multiplication blocked correctly
		Optional: Time the differences
	
	Homework 2:
		Implement larger networks by using multiple blocks
		Investigate other models of synchronization
			cuda graphs
			cuda streams and events
			grid synchronization primitives
	
	Other Topics:
		nvidia's Nsight or visual profiler, occupancy, memory bandwidth, latency, and throughput
		cuda dynamic parallelism: kernel's launching other kernels
		cudaMallocManaged
		tiling, memory coalescing, texture memory
		debugging techniques
		distribute the work across multiple GPUs

		rms prop seems to have a bug (it is not converging)
		learn about cudann, cublas, thrust, (maybe openmp for cpu optimizations)
		split methods up (learn __host__, __device__, __global__, __shared__)
		other optimizers (adam, sgd, trust region)
		skip levels, dropout, batch normalization, recurrent, convolutional
		more activation functions
		paged/pinned memory
	
*/
void test_rms_prop()
{
	size_t num_layers = 4;
	size_t sizes[] = {2, 4, 4, 1};
	size_t n = 1;

	NeuralNetwork *nn = create_neural_network(num_layers, sizes);
	ScratchSpace *scratch = create_scratchspace(nn, 1);
	RmsProp *rmsprop = create_rms_props(nn);

	for (size_t epoch = 0; epoch < 1; epoch++)
	{
		NeuralNetworkData *data = create_random_input(sizes[0], sizes[num_layers - 1], n);
		print_data(data);

		real average_loss = rms_prop(nn, scratch, data, rmsprop);
		if (1 || epoch % 1000 == 0)
		{
			printf("Epoch %zu\n", epoch);
			printf("Average loss: %9.5f\n", average_loss);
			print_neural_network(nn);
		}

		free_neural_network_data(data);
	}

	printf("Final neural network\n");
	print_neural_network(nn);

	free_rms_props(rmsprop);
	free_scratchspace(scratch);
	free_neural_network(nn);
}

void simple_test()
{
	size_t n = 1;
	size_t num_layers = 3;
	size_t sizes[] = {2, 4, 3};

	NeuralNetwork *nn = create_neural_network(num_layers, sizes);
	ScratchSpace *scratch = create_scratchspace(nn, n);
	NeuralNetworkData *data = create_random_input(sizes[0], sizes[num_layers - 1], n);

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

	forward_pass(nn, scratch, data, 0, 0);
	real computed_loss = calculate_loss(nn, scratch, data, 0, 0);
	print_neural_network(nn);
	print_scratchspace(nn, scratch);

	printf("Loss: %9.5f\n", computed_loss);

	printf("Backward pass\n");
	for (size_t i = 0; i < n; i++)
	{
		backward_pass(nn, scratch, data, i, 0);
	}
	print_scratchspace(nn, scratch);

	free_neural_network_data(data);
	free_scratchspace(scratch);
	free_neural_network(nn);
}

int main(int argc, char **argv)
{
	// srand(time(NULL));
	srand(1776);

	simple_test();
	// test_rms_prop();

	return 0;
}