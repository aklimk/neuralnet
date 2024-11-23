#include <random>
#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <functional>
#include <filesystem>

#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"

#include "utilities.hpp"

using std::vector;
using xt::xarray;

// Holds the structure and state of a neural network as well as methods
// for initilization and serializing/deserializing the network.

/*
 * Holds the structure and state of a neural network as well as the
 * methods neccecary for initilizing and serializing/deserializing
 * the network.
 *
 * Biases is a 2d array, which consists of an array of biases for each
 * neuron, for each layer.
 *
 * Weights is a 3d array, which consits of an array of weights for each
 * neurons connection to the next layer, for each neuron, for each layer.
 *
 * I.E for the example network. 
 * (Assume every neuron is connected to every other neuron).
 * n0   n3   n5
 * n1   n4   n6
 * n2        n7
 *
 * The biases array would be { {n0-bias, n1-bias, ...}, {n3-bias, ...}, ... }
 * The shapes would be {3, 2, 3}.
 *
 * The weights array would be:
 * { { {n0-n3-weight, n0-n4-weight, ...}, {n1-n3-weight, n1-n4-weight, ...}, ... } } 
 * The shapes would be { {3, 2}, {2, 3} }
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
			xarray<float>::shape_type biases_layer_shape = {(size_t)layer_sizes[i]};
			biases.push_back(xt::zeros<float>(biases_layer_shape));

			// Weights array shape for each layer is (layer size, next layer size).
			// Last layer has no weights, but has biases.
			if (i < layer_sizes.size() - 1) {
				vector<size_t> weights_layer_shape;
				weights_layer_shape.push_back(layer_sizes[i]);
				weights_layer_shape.push_back(layer_sizes[i + 1]);
				weights.push_back(xt::random::randn<float>(weights_layer_shape, 0.0, 1.0 / (float)layer_sizes[i]));
			}
		}
	}
};

