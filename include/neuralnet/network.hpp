#pragma once

#include <random>
#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <functional>
#include <filesystem>
#include <fstream>

#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xcsv.hpp"

#include "xtensor-blas/xlinalg.hpp"

using std::vector;
using xt::xarray;



// ~~~~ Basic Progress Bar ~~~~

/* Basic progress bar that takes a max value and provides and increment and display method.
 * 
 * At creation time a max value is specified. Calling IncrementBar increments the internal
 * value and displays the current bar.
 *
 * IncrementBar flushes the standard output.
*/
class ProgressBar
{
private:
	int max_value = 0;
	int current_value = 0;
			
public:
	ProgressBar(int max_value) {
		this->max_value = max_value;
	}

	void IncrementBar()
	{
		std::cout << "\r" << current_value << "/" << max_value;
		std::cout.flush();
		current_value++;
	}
};



// ~~~~ Activation Functions ~~~~

/*
 * Sigmoid compresses values to a range of [0, 1]. 
 * The compression creates an s like curve.
*/
namespace Sigmoid {
	/*
	 * Elementwise sigmoid operation on xtensor float array.
	 *
	 * Sigmoid is defined as 1 / (1 + e^-x).
	 *
	 * # Arguments
	 * `array` : Reference to xtensor array to create an xtensor expression from.
	 *
	 * # Returns
	 * An xtensor xexpression object. Xexpression objects are lazy evauluated, only
	 * being evaluated when assigned to a container type.
    */
	auto F(xarray<float>& array) {
		return 1.0 / (1.0 + xt::exp(-array));
	}
	
	/*
	 * Elementwise sigmoid derivative operation on xtensor float array.
	 *
	 * The sigmoid derivative is defined as S(x) * (1 - S(x)),
	 * where S is the sigmoid function.
	 *
	 * # Arguments
	 * `array` : Reference to xtensor array to create an xtensor expression from.
	 *
	 * # Returns
	 * An xtensor xexpression object. Xexpression objects are lazy evauluated, only
	 * being evaluated when assigned to a container type.
    */
	auto FPrime(xarray<float>& array) {
		return F(array) * (1 - F(array));
	}
};



// ~~~~ Network ~~~~

/*
 * Structure that holds input and target data as 2d arrays, for a neural network.
 *
 * inputs is a 2d array, corresponding to an array of images. Each image is greyscale
 * float encoded, with dimensions 28x28. It corresponds to the network input layer.
 *
 * targets is a 2d array, corresponding to an array of network output layer arrays. 
 * targets are the ground truth network response. It is the correct values for the 
 * network output layer for each set of inputs.
*/
struct NetworkData {
	xarray<float> inputs;
	xarray<float> targets;
};


/*
 * Holds the structure and state of a neural network as well as the
 * methods neccecary for initilizing and serializing/deserializing
 * the network.
 *
 * Biases is a 2d array, which consists of an array of biases for each
 * neuron, for each layer, excluding the starting layer.
 *
 * Weights is a 3d array, which consits of an array of weights for each
 * of the next layers nuerons connection to the previous layers neurons, 
 * for each neuron, for each layer.
 *
 * This way each array stores the next layers connection to the previous layer,
 * which makes inference fast, becouse the inputs can be quickly matrix multiplied,
 * without having to transpose the matrix.
 * 
 * I.E for the example network. 
 * (Assume every neuron is connected to every other neuron).
 * n0   n3   n5
 * n1   n4   n6
 * n2        n7
 *
 * The biases array would be 
 * { 
 *    { n3-bias, n4-bias },
 *    { n5-bias, n6-bias, n7-bias }
 * }
 * The shapes of the biasarrays would be {3, 2, 3}.
 *
 * The weights array would be:
 * {
 *     {
 *         { n0-n3-weight n1-n3-weight n2-n3-weight },
 *         { n0-n4-weight n1-n4-weight n2-n4-weight },
 *     },
 * 
 *     {
 * 	        ...
 *     }
 *		
 * }
 * The shapes would be { {2, 3}, {3, 2} }
*/
struct NeuralNetwork
{
	// Network state and properties.
	vector<int> layer_sizes;
	vector<xarray<float>> weights;
	vector<xarray<float>> biases;
	

	/*
	 * Initilization function for the neural network. 
	 * Initilizes biases to 0, and weights to a normal distribution
	 * with mean 0 and variance 1 / layer size.
	 * 
	 * # Arguments
	 * `layer_sizes` : The number of neurons in each layer
	 * of the neural network.
	*/
	NeuralNetwork(vector<int> layer_sizes)
	{
		this->layer_sizes = layer_sizes;

		// Construct shapes and arrays for the bias and weight arrays in the network.
		
		for (int i = 0; i < layer_sizes.size(); i++) {
			// Bias array shape for each layer is simply one bias for each neuron in the layer.
			// except for the input layer.
			if (i > 0) {
				xarray<float>::shape_type biases_layer_shape = {(size_t)layer_sizes[i]};
				biases.push_back(xt::zeros<float>(biases_layer_shape));
			}

			// Weights array shape for each layer is (layer size, next layer size).
			// Last layer has no weights, but has biases.
			if (i < layer_sizes.size() - 1) {
				vector<size_t> weights_layer_shape;
				weights_layer_shape.push_back(layer_sizes[i + 1]);
				weights_layer_shape.push_back(layer_sizes[i]);
				weights.push_back(xt::random::randn<float>(weights_layer_shape, 0.0, 1.0 / (float)layer_sizes[i]));
			}
		}
	}

	/* Alternate constructor that loads network from directory */
	NeuralNetwork(std::string dir_path) {
		// Read layer sizes from file.
		vector<int> layer_sizes;
		std::ifstream input_stream;
		input_stream.open(dir_path + "/layer-sizes.csv", std::ifstream::in);
		// Read comma delimited layer size array.
        for (std::string temp; std::getline(input_stream, temp, ',');) {
            layer_sizes.push_back(std::stoi(temp));
        }
		input_stream.close();
		
		// Create array using read structure.
		*this = NeuralNetwork(layer_sizes);
		
		// Read weight and bias arrays from files.
		for (int i = 0; i < layer_sizes.size() - 1; i++) {
			// Read weight array.
			input_stream.open(dir_path + "/layer-" + std::to_string(i) + "-weights.csv", std::ifstream::in);
			weights[i] = xt::load_csv<float>(input_stream);
			input_stream.close();

			// Read bias array.
			input_stream.open(dir_path + "/layer-" + std::to_string(i) + "-biases.csv", std::ifstream::in);
			biases[i] = xt::flatten(xt::load_csv<float>(input_stream));
			input_stream.close();
		}
	}

	/* Save a network to a series of array files */
	void Save(std::string dir_path) {
		// Save layer sizes to a file.
		std::ofstream output_stream;
		output_stream.open(dir_path + "/layer-sizes.csv", std::ofstream::out);
		for (int i = 0; i < layer_sizes.size(); i++) {
			output_stream << layer_sizes[i] << ",";
		}
		output_stream.close();

		// Save weight and bias arrays to files.
		for (int i = 0; i < layer_sizes.size() - 1; i++) {
			// Save weight array.
			output_stream.open(dir_path + "/layer-" + std::to_string(i) + "-weights.csv", std::ofstream::out);
			xt::dump_csv(output_stream, weights[i]);
			output_stream.close();
			
			// Save bias array.
			output_stream.open(dir_path + "/layer-" + std::to_string(i) + "-biases.csv", std::ofstream::out);
			xt::dump_csv(output_stream, xt::reshape_view(biases[i], {(size_t)1, (size_t)biases[i].shape(0)}));
			output_stream.close();
		}
	}
};


/*
 * Returns an array of network outputs for an array of network inputs.
 *
 * # Arguments
 * `network` : The neural network structure and state.
 *
 * `previous_layer` : An array of inputs to the network.
 *
 * # Invariants
 * Assumes that the input array is 1d.
*/
xarray<float> Inference(NeuralNetwork& network, xarray<float> previous_layer) {

	// Take an example network with layer sizes {3, 2, 3}.
	// n0   n3   n5
	// n1   n4   n6
	// n2        n7
	//
	// For the first layer operation.
	// Biases are {n3-b, n4-b}
	// 
	// Weights are:
	// n0-n3-w n1-n3-w n2-n3-w 
	// n0-n4-w n1-n4-w n2-n4-w
	//
	// The calculation for the output is (as a column vector):
	// Sigmoid(n0*n0-n3-w + n1*n1-n3-w + n2*n2-n3-w) + n3-b)
	// Sigmoid(n0*n0-n4-w + n1*n1-n4-w + n2*n2-n4-w) + n4-b)
	//
	// Which is just:
	// Sigmoid(matmul(layer_weights * prev_layer^T) + layer_biases)
    // 
	// Becouse the output of the calculation 
	// naturally gives a column vector, only the original
	// inputs needs to be transposed.
	previous_layer = xt::transpose(previous_layer);

	xarray<float> next_layer;
	for (int i = 0; i < network.layer_sizes.size() - 1; i++) {
		next_layer = xarray<float>::from_shape({(size_t)network.layer_sizes[i + 1]});
		next_layer = xt::linalg::dot(network.weights[i], previous_layer) + network.biases[i];
		next_layer = Sigmoid::F(next_layer);
		previous_layer = next_layer;
	}
	
	return next_layer;
}


/* Gives an accuracy rate of the network against a classification dataset.
 * Classifications are assumed to be single-class.
 *
 * # Arguments
 * `network` : A reference to the network state and parameters to use for testing.
 *
 * `testing_data` : A pair of inputs and ground truth outputs to use. The ground
 * truth outputs should be oneshot encoded.
 *
 * # Invariants
 * Assumes that the data corresponds to single-class classifications.
 *
 * # Returns
 * The accuracy of the network against the given single-class classification dataset.
 * Calculated as correct / total.
*/
float Test(NeuralNetwork& network, NetworkData testing_data) {
	int correct = 0;
		
	for (int i = 0; i < testing_data.inputs.shape(0); i++) {
		// Get the ith output inference and ith target array from the dataset.
		xarray<float> outputs = Inference(network, xt::view(testing_data.inputs, i, xt::all()));
		xarray<float> ground_truth = xt::view(testing_data.targets, i, xt::all());

		// Check if inference classification was correct.
		if (xt::argmax(outputs) == xt::argmax(ground_truth)) {
			correct++;
		}
	}

	// return correct / total.
	return (float)correct / (float)testing_data.inputs.shape(0);
}


/* Estimates the derivatives of all parameters in the network
 * in relation to the cost function using back propagation,
 * for a single set of inputs/targets.
 *
 * Currently only the mean squared error loss function is
 * supported / used.
 *
 * # Arguments
 * `network` : The network state and properties to apply 
 * back propagation over.
 *
 * `training_data` : The training data to construct a
 * derivative against.
 *
 * `weights_derivatives` : Location to store resulting
 * weight derivatives (adds to exisitng values).
 *
 * `biases_derivatives` : Location to store resulting
 * bias derivatives (adds to exisitng values).
 *
 * # Invariants
 * Inputs and targets should only be a one dimensional array.
*/
void BackPropagation(
	NeuralNetwork& network, NetworkData training_data,
	vector<xarray<float>>& weights_derivatives,
	vector<xarray<float>>& biases_derivatives
) {

	// Do an network inference on the inputs, as described in the
	// neural network inference function. The only difference being
	// is that the raw non-activation function values are also kept
	// for each layers neurons. Theese are the z values.
	vector<xarray<float>> z_arrays;
	vector<xarray<float>> network_activations = {training_data.inputs};
	xarray<float> previous_layer = xt::transpose(training_data.inputs);
	xarray<float> next_layer;
	for (int i = 0; i < network.layer_sizes.size() - 1; i++) {
		// Do inference without the activation function.
		next_layer = xarray<float>::from_shape({(size_t)network.layer_sizes[i + 1]});
		next_layer = xt::linalg::dot(network.weights[i], previous_layer) + network.biases[i];

		// Save the nueron values before and after the activation function.
		z_arrays.push_back(next_layer);
		next_layer = Sigmoid::F(next_layer);
		network_activations.push_back(next_layer);
		
		previous_layer = next_layer;
	}

	
	// Layers are indexed with a superscript.
	// Posiitons within an array/matrix within that
	// layer is indexed with a subscript
	// Z^L_j stands for non activated network output.
	// A^L_j stands for activated network output.
	// W^L_jk stands for a network weight.
	// B^L_j stands for a network bias.
	// A^(L-1)_j, W^L_jk, B^L_j computes Z^L_j.
	// Z^L_j then computes A^(L+1)_j

	// The network paramters are W and B, thus
	// it is neccecary to calculate dC/dW, dC/dB. For every W, B.
	// Using the chain rule.
	// Thus dC/DW^L_jk = dZ^L_j/DW^L_jk * dA^(L+1)_j/dZ^L_j * dC/dA^(L+1)_j

	// dC/dA^(L+1)_j is just the derivative of the MSE for every last layer activation.
	// which is simply (A^(L+1)_j - target) * 2.
    xarray<float> array_delC_delA = (network_activations[network_activations.size() - 1] - training_data.targets) * 2;

	// dA^(L+1)_j/dZ^L_j is just the derivative of the activation function.
    xarray<float> array_delA_delZ = Sigmoid::FPrime(z_arrays[z_arrays.size() - 1]);

    // dZ^L_j/Dw^L_jk is just A^L. This is becouse, without the activation function,
    // the weight and previous activation are simpliy multiplied.
    //
    // Note each entry in the array represents the derivative for all k indexes
    // of the weight.
    xarray<float> array_delZ_delW = network_activations[network_activations.size() - 2];

	// Update weights sample for the last layer.
    for (int i = 0; i < network.layer_sizes[network.layer_sizes.size() - 1]; i++) {
    	// Kth indexes of the weight, (dW^L_3k ect), represent the connect of one node in the next layer,
    	// to all of the nodes in the layer.
    	// So the weight derivative relies on a single neuron in the next layer, with all of the weights
    	// connecting that node to the previous layer.
		xt::view(weights_derivatives[weights_derivatives.size() - 1], i, xt::all()) 
			+= array_delZ_delW * array_delA_delZ[i] * array_delC_delA[i];
    }
    
	// dZ^L_j/dB^L_j is simply 1, so that term is removed from the chain rule.
    // Baises are a one dimensional array so it is simply multiplied.
    biases_derivatives[biases_derivatives.size() - 1] += array_delA_delZ * array_delC_delA;

    
	// Coninue backpropagation for the rest of the layers.
	// Save the previously calcualted activation derivatives for calculation.
	// The calculations are the same, except that each nueron in the previous layer,
	// influes the cost function on every connection to a node in the next layer,
	// so dC/dA^L_j invovles the sum of derivatives on the next layer.
	xarray<float> array_delC_delA_next = array_delC_delA;
    for (int layer_number = network.layer_sizes.size() - 2; layer_number >= 1; layer_number--) {
    	
    	// Create new, zeroed array to hold activation derivatives for the layer.
	    xarray<float> array_delC_delA = xt::zeros<float>({(size_t)network.layer_sizes[layer_number]});

    	// Loop through neurons in the layer.
        for (int i = 0; i < network.layer_sizes[layer_number]; i++) {
        	// The second part is the chain rule for dC/dZ^L_j.
        	// The first part is the network weights for a partiuclar neuron in the next
        	// layer feeding back to the previous layer.
        	auto weights_view = xt::view(network.weights[layer_number], xt::all(), i);
        	xarray<float> product = weights_view * array_delA_delZ * array_delC_delA_next;
			array_delC_delA[i] = xt::sum(product)(0);
	    }

		// Recalculate chain rule components in the same way as the last layer.
        array_delZ_delW = network_activations[layer_number - 1];
        array_delA_delZ = Sigmoid::FPrime(z_arrays[layer_number - 1]);

		// Apply calculations to sample derivatives, in the same way as the last layer.
        for (int i = 0; i < network.layer_sizes[layer_number]; i++) {
            xt::view(weights_derivatives[layer_number - 1], i, xt::all()) 
            	+= array_delZ_delW * array_delA_delZ[i] * array_delC_delA[i];
	    }

        biases_derivatives[layer_number - 1] += array_delA_delZ * array_delC_delA;

        // Save activation derivatives for calculation in the next loop.
        array_delC_delA_next = array_delC_delA;
    }
}


/* Applies backpropagation learning over many smaller batches.
 * This is roughly equivelent to taking many smaller, less accurate steps
 * towards the loss functions minima, instead of a large accurate one.
 *
 * # Arguments
 * `network` : The network to train, training is done inplace.
 *
 * `training_data` : Data to train from.
 *
 * `testing_data` : Data to test from.
 * 
 * `epochs` : The number of times to repeat training over the whole training
 * dataset.
 *
 * `batch_size` : The size of each mini batch.
 *
 * `learning_rate` : The rate at which the SGD algorithm should follow the negative derivative.
 * This can be though of as the "step size" in the negative direction of the derivative on 
 * the graph of the loss function towards the local minimum.
 *
 * # Invariants
 * Assumes the overall array size is divisible by the batch size.
 */
void StochasticGradientDescent(
    NeuralNetwork& network, NetworkData training_data, NetworkData testing_data,
    int epochs, float learning_rate, int batch_size
) {
	std::cout << "Accuracy " << Test(network, testing_data) << "\n" << std::endl;

	for (int epoch = 0; epoch < epochs; epoch++) {
		std::cout << "epoch: "	<< epoch + 1 << "/" << epochs << std::endl;
		
		// Randomize training data. Specify a shared seed so that they
		// are shuffled the same way.
		uint32_t seed = std::random_device{}();
		xt::random::seed(seed);
		xt::random::shuffle(training_data.inputs);
		xt::random::seed(seed);
		xt::random::shuffle(training_data.targets);
		
		// Split training data into mini batches.
		auto input_batches = xt::split(training_data.inputs, training_data.inputs.shape(0) / batch_size);
		auto target_batches = xt::split(training_data.targets, training_data.inputs.shape(0) / batch_size);
		
		// Initilize progress bar that keeps track of minibatch progress.
		ProgressBar progress_bar = ProgressBar(input_batches.size());

		// Loop through all minibatches and perform SGD on each one.
		for (int i = 0; i < input_batches.size(); i++) {

			// Batch based estimate of the derivative of the loss function 
			// at the networks current position. The batch estimate is a
			// simple sum of all sample estimates. 
			vector<xarray<float>> weights_derivatives;
			vector<xarray<float>> biases_derivatives;

			// Populate batch derivatives with zero initilized arrays.
			for (int j = 0; j < network.layer_sizes.size() - 1; j++) {
				weights_derivatives.push_back(
				    xt::zeros<float>({(size_t)network.layer_sizes[j + 1], (size_t)network.layer_sizes[j]})
				);
				biases_derivatives.push_back(
				    xt::zeros<float>({(size_t)network.layer_sizes[j + 1]})
				);
			}
			
			for (int j = 0; j < input_batches[0].shape(0); j++) {
				// Single sample based estimate of the derivative of the loss
				// function at the networks current position. The sample
				// estimate is added to the batch estimate (total).
				BackPropagation(
					network, 
					{
						xt::view(input_batches[i], j, xt::all()), 
						xt::view(target_batches[i], j, xt::all())
					}, 
					weights_derivatives, 
					biases_derivatives
				);
			}

			// Move the network parameters in the negative derivative direction estimate
			// calculated from the batch derivative, with a step size according to the learning rate.
			for (int j = 0; j < network.layer_sizes.size() - 1; j++) {
				network.weights[j] = network.weights[j] - learning_rate * weights_derivatives[j];
				network.biases[j] = network.biases[j] - learning_rate * biases_derivatives[j];
			}

			// Move the progress bar forward for each completed mini batch.
			progress_bar.IncrementBar();
			
		}

		// Display new accuracy after each epoch.
		std::cout << std::endl;
		std::cout << "Accuracy " << Test(network, testing_data) << std::endl;
		std::cout << std::endl;
	}
};

