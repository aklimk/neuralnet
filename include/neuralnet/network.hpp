#pragma once

#include <vector>
#include <string>

#include <xtensor/xarray.hpp>


namespace NeuralNet {
	using std::vector;
	using std::string;
	using xt::xarray;
	
	/// Structure that holds input and target data as 2d arrays, for a neural network.
	///
	/// inputs is a 2d array, corresponding to an array of images. Each image is greyscale
	/// float encoded, with dimensions 28x28. It corresponds to the network input layer.
	///
	/// targets is a 2d array, corresponding to an array of network output layer arrays. 
	/// targets are the ground truth network response. It is the correct values for the 
	/// network output layer for each set of inputs.
	struct NetworkData {
		xarray<float> inputs;
		xarray<float> targets;
	};


	/// Holds the structure and state of a neural network as well as the
	/// methods neccecary for initilizing and serializing/deserializing
	/// the network.
	///
	/// Biases is a 2d array, which consists of an array of biases for each
	/// neuron, for each layer, excluding the starting layer.
	///
	/// Weights is a 3d array, which consits of an array of weights for each
	/// of the next layers nuerons connection to the previous layers neurons, 
	/// for each neuron, for each layer.
	///
	/// This way each array stores the next layers connection to the previous layer,
	/// which makes inference fast, becouse the inputs can be quickly matrix multiplied,
	/// without having to transpose the matrix.
	/// 
	/// I.E for the example network. 
	/// (Assume every neuron is connected to every other neuron).
	/// n0   n3   n5
	/// n1   n4   n6
	/// n2        n7
	///
	/// The biases array would be 
	/// { 
	///    { n3-bias, n4-bias },
	///    { n5-bias, n6-bias, n7-bias }
	/// }
	/// The shapes of the biasarrays would be {3, 2, 3}.
	///
	/// The weights array would be:
	/// {
	///     {
	///         { n0-n3-weight n1-n3-weight n2-n3-weight },
	///         { n0-n4-weight n1-n4-weight n2-n4-weight },
	///     },
	/// 
	///     {
	/// 	        ...
	///     }
	///		
	/// }
	/// The shapes would be { {2, 3}, {3, 2} }
	struct Network
	{
		// Network state and properties.
		vector<int> layer_sizes;
		vector<xarray<float>> weights;
		vector<xarray<float>> biases;
	
		/// Initilization function for the neural network. 
		/// Initilizes biases to 0, and weights to a normal distribution
		/// with mean 0 and variance 1 / layer size.
		/// 
		/// # Arguments
		/// `layer_sizes` : The number of neurons in each layer
		/// of the neural network.
		Network(vector<int> layer_sizes);
	
		/// Alternate constructor that loads network from directory.
		Network(string dir_path);

		/// Save a network to a series of array files.
		void Save(string dir_path);
	};


	/// Returns an array of network outputs for an array of network inputs.
	///
	/// # Arguments
	/// `network` : The neural network structure and state.
	///
	/// `previous_layer` : An array of inputs to the network.
	///
	/// # Invariants
	/// Assumes that the input array is 1d.
	xarray<float> Inference(Network& network, xarray<float> previous_layer);


	/// Gives an accuracy rate of the network against a classification dataset.
	/// Classifications are assumed to be single-class.
	///
	/// # Arguments
	/// `network` : A reference to the network state and parameters to use for testing.
	///
	/// `testing_data` : A pair of inputs and ground truth outputs to use. The ground
	/// truth outputs should be oneshot encoded.
	///
	/// # Invariants
	/// Assumes that the data corresponds to single-class classifications.
	///
	/// # Returns
	/// The accuracy of the network against the given single-class classification dataset.
	/// Calculated as correct / total.
	float Test(Network& network, NetworkData testing_data);


	/// Estimates the derivatives of all parameters in the network
	/// in relation to the cost function using back propagation,
	/// for a single set of inputs/targets.
	///
	/// Currently only the mean squared error loss function is
	/// supported / used.
	///
	/// # Arguments
	/// `network` : The network state and properties to apply 
	/// back propagation over.
	///
	/// `training_data` : The training data to construct a
	/// derivative against.
	///
	/// `weights_derivatives` : Location to store resulting
	/// weight derivatives (adds to exisitng values).
	///
	/// `biases_derivatives` : Location to store resulting
	/// bias derivatives (adds to exisitng values).///*
	/// # Invariants
	/// Inputs and targets should only be a one dimensional array.
	void BackPropagation(
		Network& network, NetworkData training_data,
		vector<xarray<float>>& weights_derivatives,
		vector<xarray<float>>& biases_derivatives
	);


	/// Applies backpropagation learning over many smaller batches.
	/// This is roughly equivelent to taking many smaller, less accurate steps
	/// towards the loss functions minima, instead of a large accurate one.
	///
	/// # Arguments
	/// `network` : The network to train, training is done inplace.
	///
	/// `training_data` : Data to train from.
	///
	/// `testing_data` : Data to test from.
	/// 
	/// `epochs` : The number of times to repeat training over the whole training
	/// dataset.
	///
	/// `batch_size` : The size of each mini batch.
	///
	/// `learning_rate` : The rate at which the SGD algorithm should follow the negative derivative.
	/// This can be though of as the "step size" in the negative direction of the derivative on 
	/// the graph of the loss function towards the local minimum.
	///
	/// # Invariants
	/// Assumes the overall array size is divisible by the batch size.
	void StochasticGradientDescent(
	    Network& network, NetworkData training_data, NetworkData testing_data,
	    int epochs, float learning_rate, int batch_size
	);
}
