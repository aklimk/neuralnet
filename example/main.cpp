#include <iostream>
#include <string>

#include "loadimages.hpp"
#include "network.hpp"

// Loads MNIST data and performs stochastic gradient descent.
void TrainNetwork(NeuralNetwork& network, std::string mnist_path, std::string save_dir_path, int epochs, float learning_rate, int batch_size) {
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

    network.Save(save_dir_path);
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
    std::string error_message = 
        "Training Options:\n"
        "mnist_example [--epochs <epochs>] [--learning-rate <learning rate>] [--batch-size <batch size>]\n"
        "   [--layer-sizes <<val1>,<val2>,<val3>>]\n"
        "   [--load <network directory>]\n"
        "   [--save-dir <network directory>]\n"
        "   [--mnist-dir <path to mnist data directory>]\n"
        "   train\n\n"
        "Testing Options:\n"
        "mnist_example [--load <network directory>] [--image-path <path to image to classify> test]";


    // Network properties.
    int epochs = 5;
    float learning_rate = 0.05;
    int batch_size = 20;
    std::vector<int> layer_sizes = { 784, 128, 64, 10 };

    // Paths.
    std::string network_load_dir_path = "";
    std::string network_save_dir_path = "model";
    std::string mnist_data_dir_path = "mnist";
    std::string image_path = "image.idx";

    // Training or testing.
    int is_training = -1;

    // Parse command line arguments into variables.
    for (int i = 1; i < argc; i++) {
        std::string argument(argv[i]);
        std::string value;
        if (i < argc - 1) {
            value = std::string(argv[i + 1]);
            i++;
        }

        // Set network properties.

        if (argument == "--epochs") {
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

        
        // Set paths.
        else if (argument == "--load") {
            network_load_dir_path = value;
        }
        else if (argument == "--save-dir") {
            network_save_dir_path = value;
        }
        else if (argument == "--mnist-dir") {
            mnist_data_dir_path = value;
        }
        else if (argument == "--image-path") {
            image_path = value;
        }


        // Set operation type. 1 == train, 0 == test, -1 == neither.
        else if (argument == "test") {
            is_training = 0;
        }
        else if (argument == "train") {
            is_training = 1;
        }


        // Unkown argument.
        else {
            std::cout << error_message << std::endl;
            return 1;
        }
    }

    // Error if operation unspecified.
    if (is_training == -1) {
        std::cout << error_message << std::endl;
        return 1;
    }


    // Execute arguments.
    // Construct network with specified layout.
    NeuralNetwork network = NeuralNetwork(layer_sizes);

    // Always load model before training or testing, if the option was set.
    if (network_load_dir_path.size() != 0) {
        network = NeuralNetwork(network_load_dir_path);
    }

    if (is_training) {
        TrainNetwork(network, mnist_data_dir_path, network_save_dir_path, epochs, learning_rate, batch_size);
    }
    else {
        ClassifyIDXImage(network, image_path);
    }
}
