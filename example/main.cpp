#include <iostream>

#include "loadimages.hpp"
#include "network.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Path to MNIST data not specified. Please specify path as second argument." << std::endl;
        return 1;
    }
    
    std::string project_dir(argv[1]);
    
    // Create a neural network with appropriate structure.
    // Input imagse are 784 float values, output is a single-class 
    // classification of a handwritten digit from 0-9.
    NeuralNetwork network = NeuralNetwork({784, 64, 32, 10});

    // Load training and testing data.
    NetworkData training_data = LoadMnistDataPair(
        project_dir + "/train-images.idx3-ubyte",
        project_dir + "/train-labels.idx1-ubyte"
    );
    NetworkData testing_data = LoadMnistDataPair(
        project_dir + "/t10k-images.idx3-ubyte",
        project_dir + "/t10k-labels.idx1-ubyte"
    );

    // Perform stochastic gradient descent on the network.
    StochasticGradientDescent(
        network,
        training_data,
        testing_data,
        5, // Epochs.
        0.05, // Learning rate.
        20 // Batch size.
    );
}
