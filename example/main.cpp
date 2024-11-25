#include <iostream>
#include <string>

#include "loadimages.hpp"
#include "network.hpp"

// Loads MNIST data and performs stochastic gradient descent.
void TrainNetwork(NeuralNetwork& network, std::string mnist_path, int epochs, float learning_rate, int batch_size) {
    // Load training and testing data.
    NetworkData training_data = LoadMnistDataPair(
        mnist_path + "/train-images.idx3-ubyte",
        mnist_path + "/train-labels.idx1-ubyte"
    );
    NetworkData testing_data = LoadMnistDataPair(
        mnist_path + "/t10k-images.idx3-ubyte",
        mnist_path + "/t10k-labels.idx1-ubyte"
    );

    // Perform stochastic gradient descent on the network.
    StochasticGradientDescent(
        network,
        training_data,
        testing_data,
        epochs,
        learning_rate,
        batch_size
    );
}

// Performs a network inference on a single image, encoded in a
// one dimensional idx file.
void ClassifyIDXImage(NeuralNetwork& network, std::string image_path) {
    // Turn 2d array of ints into a flat array of floats.
    xarray<uint8_t> raw_data = ReadIDX(image_path);
    xarray<float> data = xarray<float>::from_shape(raw_data.shape());
    for (int i = 0; i < raw_data.shape(0); i++) {
        data(i) = (float)raw_data(i) / 255.0;
    }
    std::cout << "Number is: " << xt::argmax(Inference(network, data)) << std::endl;
}


int main(int argc, char** argv) {
    // Send an error message if no command line parameters where specified,
    // or if there is a parameter sent without an argument attached.
    if (argc < 3 || (argc % 2 == 0)) {
        std::cout << 
            "Use one or both of the following command line arguments. \n\n"
            "--epochs <epochs> --learning-rate <learning rate> -batch-size <batch size>\n"
            "Specify at any point before --train to set theese variables\n"
            "Default values are: epochs: 5, learning-rate: 0.05, batch-size: 20\n\n"
            "--layer-sizes <va1>,<val2>,<val3> (Do not add spaced between values)\n"
            "Specify before --train to set network layer structure\n"
            "NOTE: does not effect first or last layer!\n"
            "NOTE: do not add spaces between values!\n"
            "Defualt structure is (784), 128, 64, (10)\n"
            "I.E. --layer-sizes 128,64\n\n"
            "--train <path to mnist data folder> \n "
            "Specify a path to the mnist folder to train the network from. \n\n"
            "--image <path to image for inference> \n"
            "Specify a path to an image created with the python drawing pad in order to classify it.\n"
        << std::endl;
        return 1;
    }

    // Network properties.
    int epochs = 5;
    float learning_rate = 0.05;
    int batch_size = 20;
    std::vector<int> layer_sizes = { 784, 128, 64, 10 };
    NeuralNetwork network = NeuralNetwork(layer_sizes);

    // Parse command line arguments.
    for (int i = 1; i < argc - 1; i += 2) {
        std::string argument(argv[i]);
        std::string value(argv[i + 1]);

        // Train the network.
        if (argument == "--train") {
            network = NeuralNetwork(layer_sizes);
            TrainNetwork(network, value, epochs, learning_rate, batch_size);   
        }

        // Classify a single digit stroed in an idx file.
        else if (argument == "--image") {
            ClassifyIDXImage(network, value);
        }

        // Set variables.
        else if (argument == "--epochs") {
            epochs = std::stoi(value);
        }

        else if (argument == "--learning-rate") {
            learning_rate = std::stof(value);
        }

        else if (argument == "--batch-size") {
            batch_size = std::stoi(value);
        }

        else if (argument == "--layer-sizes") {
            layer_sizes.clear();
            layer_sizes.push_back(784);
            
            // Parse layer sizes comma-delimited array.
            std::istringstream stream(value);
            for (std::string temp; std::getline(stream, temp, ',');) {
                layer_sizes.push_back(std::stoi(temp));
            }

            layer_sizes.push_back(10);
        }
    }
}
