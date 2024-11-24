#pragma once

#include <random>
#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <functional>
#include <filesystem>

#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xsort.hpp"

#include "xtensor-blas/xlinalg.hpp"

#include "utilities.hpp"

using std::vector;
using xt::xarray;

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
		xarray<float> outputs = Inference(network, xt::view(testing_data.inputs, i, xt::all()));
		xarray<float> ground_truth = xt::view(testing_data.targets, i, xt::all());

		if (xt::argmax(outputs) == xt::argmax(ground_truth)) {
			correct++;
		}
	}

	return (float)correct / (float)testing_data.inputs.shape(0);
}



void BackPropagation() {};


void StochasticGradientDescent() {};

