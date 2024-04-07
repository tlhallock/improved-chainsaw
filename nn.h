
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define real float
#define VERBOSE_FORWARD 1
#define VERBOSE_BACKWARD 1
#define VERBOSE_RMS_PROP 1
#define VERBOSE_LOSS 1
#define LEAKY_RELU_SLOPE 0.01
// #define M_PI ((real) 3.14159265358979323846)

typedef enum ActivationFunction_
{
    // Add softmax
    SIGMOID,
    RELU,
    LEAKY_RELU,
    TANH,
    LINEAR
} ActivationFunction;

typedef struct NeuralNetwork_
{
    size_t n;
    // The first size is the input size
    size_t *sizes;
    // Thought I would need this, but I guess it isn't used
    size_t *weight_offsets;
    size_t *bias_offsets;
    ActivationFunction *activations;

    // The dimension of the i-th W matrix is sizes[i+1] * sizes[i]
    real *weights;
    real *biases;
} NeuralNetwork;

typedef struct ScratchSpace_
{
    // This doesn't actually have to have separate memory for each data point...
    size_t n;
    // Could be recomputed from sizes, but it's easier to store it
    size_t activations_size;
    // This also includes the initial input
    real *activations;

    real *d_weights;
    real *d_bias;
    real *d_activations;
} ScratchSpace;

typedef struct NeuralNetworkData_
{
    size_t n;
    size_t input_dimension;
    size_t output_dimension;
    real *input;
    real *output;
} NeuralNetworkData;

typedef struct RmsProp_
{
    real beta;
    real eta;
    real *ew;
    real *eb;
} RmsProp;

void print_activation_function(ActivationFunction activation);

size_t get_weights_size(NeuralNetwork *nn);
size_t get_biases_size(NeuralNetwork *nn);
size_t get_maximum_activations_size(NeuralNetwork *nn);

NeuralNetwork *create_neural_network(size_t n, size_t *sizes);
void free_neural_network(NeuralNetwork *nn);
void print_neural_network(NeuralNetwork *nn);

void forward_pass(NeuralNetwork *nn, ScratchSpace *scratch, NeuralNetworkData *data, size_t data_index,
                  size_t scratch_index);
void backward_pass(NeuralNetwork *nn, ScratchSpace *scratch, NeuralNetworkData *data, size_t data_index,
                   size_t scratch_index);

ScratchSpace *create_scratchspace(NeuralNetwork *nn, size_t n);
void free_scratchspace(ScratchSpace *scratch);
void print_scratchspace(NeuralNetwork *nn, ScratchSpace *scratch);

RmsProp *create_rms_props(NeuralNetwork *nn);
void free_rms_props(RmsProp *rms);
void print_rms_prop(NeuralNetwork *nn, RmsProp *rmsprop);
real rms_prop(NeuralNetwork *nn, ScratchSpace *scratch, NeuralNetworkData *data, RmsProp *rmsprop);

NeuralNetworkData *create_random_input(size_t input_dimension, size_t output_dimension, size_t n);
void free_neural_network_data(NeuralNetworkData *data);
void print_data(NeuralNetworkData *data);
real calculate_loss(NeuralNetwork *nn, ScratchSpace *scratch, NeuralNetworkData *data, size_t data_index,
                    size_t scratch_index);

static inline size_t get_nrows(NeuralNetwork *nn, size_t layer)
{
    return nn->sizes[layer + 1];
}

static inline size_t get_ncols(NeuralNetwork *nn, size_t layer)
{
    return nn->sizes[layer];
}

// Row major (used for weights)
static inline real sub(real *matrix, size_t ncols, size_t row, size_t col)
{
    return matrix[row * ncols + col];
}
