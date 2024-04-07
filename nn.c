#include "nn.h"

/*


L = (a_{2,0} - y_{0})^2 + (a_{2,1} - y_{1})^2

a_{2,i} = relu(b_{1,i} + w_{1,i,0} * a_{1,0} + w_{1,i,1} * a_{1, 1} )
a_{1,i} = relu(b_{0,i} + w_{0,i,0} * a_{0,0} + w_{0,i,1} * a_{0, 1} )
a_{0, i} = x_{i}

// Back prop


dL/da_{2,i} = 2 * (a_{2,i} - y_{i})

da_{2,0}/db_{1,0} = 1					(or 0)
da_{2,0}/db_{1,1} = 0					(or 1)

da_{2,i}/dw_{1,i,j} = a_{i,j}			(or 0)
	da_{2,0}/dw_{1,0,0} = a_{1,0}		(or 0)
	da_{2,0}/dw_{1,0,1} = a_{1,1}		(or 0)

dL/dw_{1,i,j} = dL/da_{2,i} * da_{2,i}/dw_{1,i,j} = dL/da_{2,i} * a_{i,j}
	dL/dw_{1,0,0} = dL/da_{2,0} * da_{2,0}/dw_{1,0,0} = dL/da_{2,0} * a_{1,0}

da_{1,i}/dw_{0,i,j} = a_{i,j}
da_{2,i}/da_{1,j} = w_{1,i,j}
	da_{2,0}/da_{1,1} = w_{1,0,1}
dL/da_{1,i} = dL/da_{2,0} * da_{2,0}/da_{1,i} + dL/da_{2,1} * da_{2,1}/da_{1,i} +
			= dL/da_{2,0} * w_{1,0,i} + dL/da_{2,1} * w_{1,1,i}


dL/dw_{0,i,j} = dL/da_{1,i} * da_{1,i}/dw_{0,i,j} = dL/da_{1,i} * a_{i,j}

dL/db_{0,i} = dL/da_{1,i} * da_{1,i}/db_{0,i} = dL/da_{1,i} * 1


*/

void box_muller(real *array, size_t n);
void accumulate(size_t *source, size_t n, size_t *dest); // not used...

void accumulate(size_t *source, size_t n, size_t *dest)
{
	dest[0] = source[0];
	for (size_t i = 1; i < n; i++)
	{
		dest[i] = dest[i - 1] + source[i];
	}
}

size_t get_weights_size(NeuralNetwork *nn)
{
	size_t weights_size = 0;
	for (size_t i = 0; i < nn->n - 1; i++)
	{
		weights_size += nn->sizes[i + 1] * nn->sizes[i];
	}
	return weights_size;
}

size_t get_biases_size(NeuralNetwork *nn)
{
	size_t biases_size = 0;
	for (size_t i = 1; i < nn->n; i++)
	{
		biases_size += nn->sizes[i];
	}
	return biases_size;
}

size_t get_activations_size(NeuralNetwork *nn)
{
	size_t activations_size = 0;
	for (size_t i = 0; i < nn->n; i++)
	{
		activations_size += nn->sizes[i];
	}
	return activations_size;
}

void print_neural_network(NeuralNetwork *nn)
{
	printf("Neural network:\n");
	printf("n: %d\n", nn->n);
	printf("sizes:\n");
	for (int i = 0; i < nn->n; i++)
	{
		printf("%d ", nn->sizes[i]);
	}
	printf("\n");
	printf("activations:\n");
	for (size_t i = 0; i < nn->n; i++)
	{
		print_activation_function(nn->activations[i]);
		printf(" ");
	}
	printf("\n");
	printf("weights:\n");

	real *current_weights = nn->weights;
	for (size_t layer = 0; layer < nn->n - 1; layer++)
	{
		printf("-----------------\n");
		size_t nrows = get_nrows(nn, layer);
		size_t ncols = get_ncols(nn, layer);

		for (size_t i = 0; i < nrows; i++)
		{
			for (size_t j = 0; j < ncols; j++)
			{
				printf("%9.5f ", sub(current_weights, ncols, i, j));
			}
			printf("\n");
		}
		current_weights += nrows * ncols;
	}
	printf("-----------------\n");
	printf("\n");
	printf("biases:\n");
	real *current_biases = nn->biases;
	for (size_t layer = 1; layer < nn->n; layer++)
	{
		size_t layer_size = nn->sizes[layer];
		for (size_t i = 0; i < layer_size; i++)
		{
			printf("%9.5f ", current_biases[i]);
		}
		printf("\n");
		current_biases += layer_size;
	}
	printf("\n");
}

void print_scratchspace(NeuralNetwork *nn, ScratchSpace *scratch)
{
	printf("Scratch space:\n");
	printf("n: %zu\n", scratch->n);
	printf("activations_size: %zu\n", scratch->activations_size);
	for (size_t data_point = 0; data_point < scratch->n; data_point++)
	{
		printf("Data point %zu:\n", data_point);
		real *current_activations = scratch->activations + data_point * scratch->activations_size;
		for (size_t layer = 0; layer < nn->n; layer++)
		{
			printf("\tLayer %zu activations: ", layer);
			for (size_t j = 0; j < nn->sizes[layer]; j++)
			{
				printf("%9.5f ", current_activations[j]);
			}
			printf("\n");
			current_activations += nn->sizes[layer];
		}

		real *current_d_weights = scratch->d_weights + data_point * get_weights_size(nn);
		for (size_t layer = 0; layer < nn->n - 1; layer++)
		{
			printf("\tLayer %zu d_weights:\n", layer);
			size_t nrows = get_nrows(nn, layer);
			size_t ncols = get_ncols(nn, layer);

			for (size_t i = 0; i < nrows; i++)
			{
				printf("\t\t");
				for (size_t j = 0; j < ncols; j++)
				{
					printf("%9.5f ", sub(current_d_weights, ncols, i, j));
				}
				printf("\n");
			}
			current_d_weights += nrows * ncols;
		}

		printf("\td_biases:\n");
		real *current_d_bias = scratch->d_bias + data_point * get_biases_size(nn);
		for (size_t layer = 1; layer < nn->n; layer++)
		{
			printf("\t\tLayer %zu: ", layer);
			for (size_t i = 0; i < nn->sizes[layer]; i++)
			{
				printf("%9.5f ", current_d_bias[i]);
			}
			printf("\n");
			current_d_bias += nn->sizes[layer];
		}

		printf("\td_activations:\n");
		real *current_d_activations = scratch->d_activations + data_point * scratch->activations_size;
		for (size_t layer = 0; layer < nn->n; layer++)
		{
			printf("\t\tLayer %zu: ", layer);
			for (size_t i = 0; i < nn->sizes[layer]; i++)
			{
				printf("%9.5f ", current_d_activations[i]);
			}
			printf("\n");
			current_d_activations += nn->sizes[layer];
		}
	}
}

void print_data(NeuralNetworkData *data)
{
	printf("Data:\n");
	printf("\tn: %zu\n", data->n);
	printf("\tinput_dimension: %zu\n", data->input_dimension);
	printf("\toutput_dimension: %zu\n", data->output_dimension);
	for (size_t i = 0; i < data->n; i++)
	{
		printf("\t\t% 5zu: ", i);
		for (size_t j = 0; j < data->input_dimension; j++)
		{
			printf("%9.5f ", data->input[i * data->input_dimension + j]);
		}
		printf(" -> ");
		for (size_t j = 0; j < data->output_dimension; j++)
		{
			printf("%9.5f ", data->output[i * data->output_dimension + j]);
		}
		printf("\n");
	}
}

void print_rms_prop(NeuralNetwork *nn, RmsProp *rmsprop)
{
	printf("RMSProp:\n");
	printf("\tbeta: %9.5f\n", rmsprop->beta);
	printf("\t eta: %9.5f\n", rmsprop->eta);
	printf("\t  ew: ");
	for (size_t i = 0; i < get_weights_size(nn); i++)
	{
		printf("%9.5f ", rmsprop->ew[i]);
	}
	printf("\n");
	printf("\t  eb: ");
	for (size_t i = 0; i < get_biases_size(nn); i++)
	{
		printf("%9.5f ", rmsprop->eb[i]);
	}
	printf("\n");
}

void print_activation_function(ActivationFunction activation)
{
	switch (activation)
	{
	case SIGMOID:
		printf("SIGMOID");
		break;
	case RELU:
		printf("RELU");
		break;
	case TANH:
		printf("TANH");
		break;
	case LEAKY_RELU:
		printf("LEAKY_RELU");
		break;
	case LINEAR:
		printf("LINEAR");
		break;
	default:
		printf("Unknown activation function: %d\n", activation);
		raise(SIGABRT);
		break;
	}
}

ScratchSpace *create_scratchspace(NeuralNetwork *nn, size_t n)
{
	ScratchSpace *scratch = malloc(sizeof(ScratchSpace));
	scratch->n = n;

	scratch->activations_size = get_activations_size(nn);
	scratch->activations = malloc(n * scratch->activations_size * sizeof(real));
	scratch->d_weights = malloc(n * get_weights_size(nn) * sizeof(real));
	scratch->d_bias = malloc(n * get_biases_size(nn) * sizeof(real));
	scratch->d_activations = malloc(n * scratch->activations_size * sizeof(real));

	memset(scratch->d_weights, 0, n * get_weights_size(nn) * sizeof(real));
	memset(scratch->d_bias, 0, n * get_biases_size(nn) * sizeof(real));

	// Because they are printed before being set...
	memset(scratch->d_activations, 0, n * scratch->activations_size * sizeof(real));
	memset(scratch->activations, 0, n * scratch->activations_size * sizeof(real));

	return scratch;
}

void free_scratchspace(ScratchSpace *scratch)
{
	free(scratch->d_weights);
	free(scratch->d_bias);
	free(scratch->d_activations);
	free(scratch->activations);
	free(scratch);
}

void box_muller(real *array, size_t n)
{
	for (size_t i = 0; i < n; i += 2)
	{
		real u1 = (real)rand() / (real)RAND_MAX;
		real u2 = (real)rand() / (real)RAND_MAX;
		real r = sqrt(-2 * log(u1));
		real theta = 2 * ((real)M_PI) * u2;
		array[i] = r * cos(theta);
		if (i + 1 < n)
		{
			array[i + 1] = r * sin(theta);
		}
	}
}

void initialize_parameters(NeuralNetwork *nn)
{
	// He initialization
	box_muller(nn->weights, get_weights_size(nn));
	real *current_weights = nn->weights;
	for (size_t layer = 0; layer < nn->n - 1; layer++)
	{
		size_t ncols = get_ncols(nn, layer);
		size_t weights_size = get_nrows(nn, layer) * ncols;
		real variance = 2.0 / ncols;

		for (size_t i = 0; i < weights_size; i++)
		{
			current_weights[i] *= variance;
		}

		current_weights += weights_size;
	}
	real *current_biases = nn->biases;
	for (size_t layer = 1; layer < nn->n; layer++)
	{
		for (size_t i = 0; i < nn->sizes[layer]; i++)
		{
			current_biases[i] = 0.1 * (2 * ((real)rand() / (real)RAND_MAX) - 1);
		}

		current_biases += nn->sizes[layer];
	}
}

NeuralNetwork *create_neural_network(size_t n, size_t *sizes)
{
	NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
	nn->n = n;
	nn->sizes = malloc(n * sizeof(size_t));
	memcpy(nn->sizes, sizes, n * sizeof(size_t));
	nn->activations = malloc(n * sizeof(ActivationFunction));
	for (size_t i = 0; i < n; i++)
	{
		nn->activations[i] = RELU;
	}
	nn->weight_offsets = malloc(n * sizeof(size_t));
	nn->bias_offsets = malloc(n * sizeof(size_t));

	size_t weights_size = 0;
	for (size_t i = 0; i < nn->n - 1; i++)
	{
		size_t current_weight_size = nn->sizes[i] * nn->sizes[i + 1];
		nn->weight_offsets[i] = weights_size;
		weights_size += current_weight_size;
	}
	nn->weights = malloc(weights_size * sizeof(real));

	size_t biases_size = 0;
	for (size_t i = 1; i < nn->n; i++)
	{
		biases_size += nn->sizes[i];
		nn->bias_offsets[i] = biases_size;
	}
	nn->biases = malloc(biases_size * sizeof(real));
	initialize_parameters(nn);
	return nn;
}

void forward_pass(NeuralNetwork *nn, ScratchSpace *scratch, NeuralNetworkData *data, size_t data_index,
				  size_t scratch_index)
{
	if (nn->sizes[0] != data->input_dimension)
	{
		printf("Input dimension mismatch: %d != %d\n", nn->sizes[0], data->input_dimension);
		raise(SIGABRT);
	}
	if (nn->sizes[nn->n - 1] != data->output_dimension)
	{
		printf("Output dimension mismatch: %d != %d\n", nn->sizes[nn->n - 1], data->output_dimension);
		raise(SIGABRT);
	}

	real *input = data->input + data_index * nn->sizes[0];

	real *current_biases = nn->biases;
	real *current_weights = nn->weights;
	real *current_inputs = scratch->activations + scratch_index * scratch->activations_size;
	real *current_activations = current_inputs + nn->sizes[0]; // rename to target or smh

	if (VERBOSE_FORWARD)
	{
		printf("Data point %zu\n", data_index);
		print_neural_network(nn);
		if (data_index != 0)
		{
			print_scratchspace(nn, scratch);
		}
	}

	memcpy(current_inputs, input, nn->sizes[0] * sizeof(real));

	for (size_t layer = 0; layer < nn->n - 1; layer++)
	{
		if (VERBOSE_FORWARD)
		{
			printf("====================================\n");
			printf("Layer %zu\n", layer);
		}

		size_t result_size = get_nrows(nn, layer); // output size
		size_t inputs_size = get_ncols(nn, layer); // input size

		memcpy(current_activations, current_biases, result_size * sizeof(real));

		for (size_t dest_ndx = 0; dest_ndx < result_size; dest_ndx++)
		{
			if (VERBOSE_FORWARD)
			{
				printf("\t\tcurrent_activations[%zu]: activation(%9.5f", dest_ndx, current_activations[dest_ndx]);
			}
			for (size_t src_ndx = 0; src_ndx < inputs_size; src_ndx++)
			{
				real weight = sub(current_weights, inputs_size, dest_ndx, src_ndx);
				if (VERBOSE_FORWARD)
				{
					printf(" + %9.5f * %9.5f", weight, current_inputs[src_ndx]);
				}
				current_activations[dest_ndx] += weight * current_inputs[src_ndx];
			}
			if (VERBOSE_FORWARD)
			{
				printf(") = activation(%9.5f)\n", dest_ndx, current_activations[dest_ndx]);
			}
		};

		switch (nn->activations[layer])
		{
		case SIGMOID:
			for (size_t i = 0; i < result_size; i++)
			{
				current_activations[i] = 1.0 / (1.0 + exp(-current_activations[i]));

				if (VERBOSE_FORWARD)
				{
					printf("%9.5f ", current_activations[i]);
				}
			}
			break;
		case RELU:
			for (size_t i = 0; i < result_size; i++)
			{
				if (current_activations[i] < 0)
				{
					current_activations[i] = 0;
				}
			}
			break;
		case TANH:
			for (size_t i = 0; i < result_size; i++)
			{
				current_activations[i] = tanh(current_activations[i]);
			}
			break;
		case LEAKY_RELU:
			for (size_t i = 0; i < result_size; i++)
			{
				if (current_activations[i] < 0)
				{
					current_activations[i] = current_activations[i] * LEAKY_RELU_SLOPE;
				}
			}
			break;
		case LINEAR:
			break;
		default:
			printf("Unknown activation function: %d\n", nn->activations[layer]);
			raise(SIGABRT);
			break;
		}

		if (VERBOSE_FORWARD)
		{
			printf("Applying activation functions: ");
			for (size_t i = 0; i < result_size; i++)
			{
				printf("%9.5f ", current_activations[i]);
			}
			printf("\n");
		}

		current_biases += result_size;
		current_inputs += inputs_size;
		current_activations += result_size;
		current_weights += inputs_size * result_size;

		if (VERBOSE_FORWARD)
		{
			printf("====================================\n");
		}
	}
}

void backward_pass(NeuralNetwork *nn, ScratchSpace *scratch, NeuralNetworkData *data, size_t data_index,
				   size_t scratch_index)
{
	// This could be calculated into the same region of the scratch space
	size_t weights_size = get_weights_size(nn);
	size_t biases_size = get_biases_size(nn);
	size_t last_layer_weights_size = data->output_dimension * nn->sizes[nn->n - 2];

	if (VERBOSE_BACKWARD)
	{
		printf("Data point %zu\n", data_index);
	}

	real *datapoint_activations = scratch->activations + scratch_index * scratch->activations_size;
	real *next_activations = datapoint_activations + scratch->activations_size - data->output_dimension;
	real *current_activations = next_activations - nn->sizes[nn->n - 2];
	real *datapoint_d_activations = scratch->d_activations + scratch_index * scratch->activations_size;
	real *next_d_activations = datapoint_d_activations + scratch->activations_size - data->output_dimension;
	real *current_d_activations = next_d_activations - nn->sizes[nn->n - 2];
	real *datapoint_d_weights = scratch->d_weights + scratch_index * weights_size;
	real *current_d_weights = datapoint_d_weights + weights_size - last_layer_weights_size;
	real *current_weights = nn->weights + weights_size - last_layer_weights_size;
	real *d_bias = scratch->d_bias + scratch_index * biases_size;

	real *actual_output = next_activations;
	real *expected_output = data->output + data_index * data->output_dimension;
	for (size_t i = 0; i < data->output_dimension; i++)
	{
		next_d_activations[i] = 2 * (actual_output[i] - expected_output[i]);
		if (VERBOSE_BACKWARD)
		{
			printf("d_activations[%zu]: 2 * (%9.5f - %9.5f) = %9.5f\n", i, actual_output[i], expected_output[i],
				   next_d_activations[i]);
		}
	}

	for (int layer = nn->n - 2; layer >= 0; layer--)
	{
		if (VERBOSE_BACKWARD)
		{
			printf("\tLayer %d\n", layer);
		}
		size_t layer_size = get_ncols(nn, layer);
		size_t next_layer_size = get_nrows(nn, layer);

		for (size_t col = 0; col < layer_size; col++)
		{
			real d_activation_function = 1.0;
			switch (nn->activations[layer])
			{
			case RELU:
				d_activation_function = current_activations[col] > 0 ? 1 : 0;
				break;
			case LEAKY_RELU:
				d_activation_function = current_activations[col] > 0 ? 1 : LEAKY_RELU_SLOPE;
				break;
			default:
				printf("Activation function %d not implemented\n", nn->activations[layer]);
				raise(SIGABRT);
				break;
			}

			// dL/da_{1,i} = \sum_j dL/da_{2,j} * da_{2,j}/da_{1,i} = \sum_j dL/da_{2,j} * w_{1,j,i}
			if (VERBOSE_BACKWARD)
			{
				printf("current_d_activations[%zu]: %9.5f * (", col, d_activation_function);
			}
			current_d_activations[col] = 0;
			for (size_t row = 0; row < next_layer_size; row++)
			{
				if (VERBOSE_BACKWARD)
				{
					if (row != 0)
					{
						printf(" + ");
					}

					printf("%9.5f * %9.5f", next_d_activations[row], sub(current_weights, layer_size, row, col));
				}
				current_d_activations[col] += sub(current_weights, layer_size, row, col) * next_d_activations[row];
			}
			current_d_activations[col] *= d_activation_function;
			if (VERBOSE_BACKWARD)
			{
				printf(") = %9.5f\n", current_d_activations[col]);

				printf("\td_weights [%zu, :]\n");
			}
			for (size_t row = 0; row < next_layer_size; row++)
			{
				real weight = sub(current_weights, layer_size, row, col);
				current_d_weights[row * layer_size + col] = current_d_activations[col] * weight;
				if (VERBOSE_BACKWARD)
				{
					printf("\t\t(%9.5f * %.04f) = %9.5f,\n", current_d_activations[col], weight,
						   current_d_weights[row * layer_size + col]);
				}
			}
			for (size_t row = 0; row < layer_size; row++)
			{
				d_bias[row] = current_d_activations[row];
			}
		}

		if (layer > 0)
		{
			// Could calculate this at the beginning of the loop...
			size_t previous_layer_size = nn->sizes[layer - 1];

			next_activations = current_activations;
			current_activations -= previous_layer_size;
			next_d_activations = current_d_activations;
			current_d_activations -= previous_layer_size;

			current_d_weights -= previous_layer_size * layer_size;
			current_weights -= previous_layer_size * layer_size;
		}
	}
}

RmsProp *create_rms_props(NeuralNetwork *nn)
{
	size_t weights_size = get_weights_size(nn);
	size_t biases_size = get_biases_size(nn);
	RmsProp *rms = malloc(sizeof(RmsProp));
	rms->beta = 0.9;
	rms->eta = 0.001;
	rms->ew = malloc(weights_size * sizeof(real));
	rms->eb = malloc(biases_size * sizeof(real));
	memset(rms->ew, 0, weights_size * sizeof(real));
	memset(rms->eb, 0, biases_size * sizeof(real));
	return rms;
}

void free_rms_props(RmsProp *rms)
{
	free(rms->ew);
	free(rms->eb);
	free(rms);
}

real rms_prop(NeuralNetwork *nn, ScratchSpace *scratch, NeuralNetworkData *data, RmsProp *rmsprop)
{
	size_t weights_size = get_weights_size(nn);
	size_t biases_size = get_biases_size(nn);
	real loss_sum = 0;

	// https://optimization.cbe.cornell.edu/index.php?title=RMSProp
	for (size_t i = 0; i < data->n; i++)
	{
		forward_pass(nn, scratch, data, i, 0);
		backward_pass(nn, scratch, data, i, 0);
		real computed_loss = calculate_loss(nn, scratch, data, i, 0);
		loss_sum += computed_loss;

		if (VERBOSE_RMS_PROP)
		{
			printf("Loss in iteration %d: %9.5f\n", i, computed_loss);
			print_scratchspace(nn, scratch);
			printf("\tWeight updates\n");
		}
		for (size_t j = 0; j < weights_size; j++)
		{
			real new_ew =
				rmsprop->beta * rmsprop->ew[j] + (1 - rmsprop->beta) * scratch->d_weights[j] * scratch->d_weights[j];
			if (VERBOSE_RMS_PROP)
			{
				printf("\t\tew[%zu]      = %9.5f = %9.5f * %9.5f + %9.5f * %9.5f^2\n", j, new_ew, rmsprop->beta,
					   rmsprop->ew[j], 1 - rmsprop->beta, scratch->d_weights[j]);
			}
			rmsprop->ew[j] = new_ew;

			real new_weight = nn->weights[j] - rmsprop->eta * scratch->d_weights[j] / sqrt(rmsprop->ew[j] + 1e-8);
			if (VERBOSE_RMS_PROP)
			{
				printf("\t\tweights[%zu] = %9.5f = %9.5f - %9.5f * %9.5f / sqrt(%9.5f + 1e-8)\n", j, new_weight,
					   nn->weights[j], rmsprop->eta, scratch->d_weights[j], rmsprop->ew[j]);
			}
			nn->weights[j] = new_weight;
		}

		if (VERBOSE_RMS_PROP)
		{
			printf("\tBias updates\n");
		}
		real *d_activations = scratch->d_activations + nn->sizes[0];
		for (size_t layer = 1; layer < nn->n; layer++)
		{
			size_t layer_size = nn->sizes[layer];
			for (size_t j = 0; j < layer_size; j++)
			{
				real d_bias = d_activations[j];
				// switch (nn->activations[layer])
				// {
				// case RELU:
				// case LEAKY_RELU:
				// default:
				// 	printf("Activation function %d not implemented\n", nn->activations[layer]);
				// 	raise(SIGABRT);
				// 	break;
				// }
				real new_eb = rmsprop->beta * rmsprop->eb[j] + (1 - rmsprop->beta) * d_bias * d_bias;
				if (VERBOSE_RMS_PROP)
				{
					printf("\t\teb[%zu] = %9.5f = %9.5f * %9.5f + %9.5f * %9.5f^2\n", j, new_eb, rmsprop->beta,
						   rmsprop->eb[j], 1 - rmsprop->beta, d_bias);
				}
				rmsprop->eb[j] = new_eb;

				real new_bias = nn->biases[j] - rmsprop->eta * d_bias / sqrt(rmsprop->eb[j] + 1e-8);
				if (VERBOSE_RMS_PROP)
				{
					printf("\t\tbiases[%zu] = %9.5f = %9.5f - %9.5f * %9.5f / sqrt(%9.5f + 1e-8)\n", j, new_bias,
						   nn->biases[j], rmsprop->eta, d_bias, rmsprop->eb[j]);
				}
				nn->biases[j] = new_bias;
			}
		}

		if (VERBOSE_RMS_PROP)
		{
			print_rms_prop(nn, rmsprop);
			print_neural_network(nn);
		}
	}
	return loss_sum / data->n;
}

real calculate_loss(NeuralNetwork *nn, ScratchSpace *scratch, NeuralNetworkData *data, size_t data_index,
					size_t scratch_index)
{
	real loss = 0;
	real *expected_output = data->output + data_index * data->output_dimension;
	real *actual_output =
		scratch->activations + (scratch_index + 1) * scratch->activations_size - data->output_dimension;

	if (VERBOSE_LOSS)
	{
		printf("Loss for data point %zu: ", data_index);
	}

	for (size_t i = 0; i < data->output_dimension; i++)
	{
		if (VERBOSE_LOSS)
		{
			if (i != 0)
			{
				printf("+ ");
			}
			printf("(%9.5f - %9.5f)^2 ", expected_output[i], actual_output[i]);
		}
		real dx = expected_output[i] - actual_output[i];
		loss += dx * dx;
	}
	if (VERBOSE_LOSS)
	{
		printf("= %9.5f\n", loss);
	}
	return loss;
}

void free_neural_network(NeuralNetwork *nn)
{
	free(nn->sizes);
	free(nn->weights);
	free(nn->biases);
	free(nn->activations);
	free(nn->weight_offsets);
	free(nn->bias_offsets);
	free(nn);
}

NeuralNetworkData *create_random_input(size_t input_dimension, size_t output_dimension, size_t n)
{
	NeuralNetworkData *data = malloc(sizeof(NeuralNetworkData));
	data->input = malloc(input_dimension * n * sizeof(real));
	data->output = malloc(output_dimension * n * sizeof(real));
	data->n = n;
	data->input_dimension = input_dimension;
	data->output_dimension = output_dimension;
	for (size_t i = 0; i < n; i++)
	{
		real value = 0;
		for (size_t j = 0; j < input_dimension; j++)
		{
			real noise = 2 * ((real)rand() / (real)RAND_MAX) - 1;
			data->input[i * input_dimension + j] = noise;
			value += j * noise * noise;
		}
		for (size_t j = 0; j < output_dimension; j++)
		{
			real noise = 2 * ((real)rand() / (real)RAND_MAX) - 1;
			data->output[i * output_dimension + j] = 1 + j * (value + noise);
		}
	}
	return data;
}

void free_neural_network_data(NeuralNetworkData *data)
{
	free(data->input);
	free(data->output);
	free(data);
}

size_t get_maximum_activations_size(NeuralNetwork *nn)
{
	size_t max = 0;
	for (size_t i = 0; i < nn->n; i++)
	{
		if (nn->sizes[i] > max)
		{
			max = nn->sizes[i];
		}
	}
	return max;
}
