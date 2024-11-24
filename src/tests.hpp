#include <chrono>
#include <thread>

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"

#include "loadimages.hpp"
#include "neuralnetwork.hpp"

void TestUtilities() {
    xt::print_options::set_edge_items(20);
    xt::print_options::set_line_width(200);

    // Progress bar operation test.
    ProgressBar progress_bar = ProgressBar(100);
    for (int i = 0; i < 100; i++) {
        progress_bar.IncrementBar();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cout << "\n" << std::endl;

    // Sigmoid activation function tests.
    xt::xarray<float> testing = {1.0, 5.3, 9.3};
    std::cout << testing << std::endl;
    xt::xarray<float> r0 = Sigmoid::F(testing);
    std::cout << r0 << std::endl;
    xt::xarray<float> r1 = Sigmoid::FPrime(testing);
    std::cout << r1 << std::endl;
    std::cout << std::endl;

    // Image loading and IDX loading library tests.
    std::string project_dir = R"(C:/users/andre/downloads/neuralnet/mnist)";
    NetworkData training_data = LoadMnistDataPair(
        project_dir + "/train-images.idx3-ubyte",
        project_dir + "/train-labels.idx1-ubyte"
    );
    NetworkData testing_data = LoadMnistDataPair(
        project_dir + "/t10k-images.idx3-ubyte",
        project_dir + "/t10k-labels.idx1-ubyte"
    );
    std::cout << xt::adapt(training_data.inputs.shape()) << "\n" << xt::adapt(training_data.targets.shape()) << std::endl;
    std::cout << training_data.inputs << "\n\n" << training_data.targets << std::endl;
    std::cout << std::endl;
    std::cout << xt::adapt(testing_data.inputs.shape()) << "\n" << xt::adapt(testing_data.targets.shape()) << std::endl;
    std::cout << testing_data.inputs << "\n\n" << testing_data.targets << std::endl;
    std::cout << "\n\n\n" << std::endl;
}

void TestNetwork() {
    // Network initilization tests.
    NeuralNetwork network = NeuralNetwork({5, 5, 6});
    for (int i = 0; i < network.layer_sizes.size() - 1; i++) {
        std::cout << xt::adapt(network.biases[i].shape()) << std::endl;
        std::cout << network.biases[i] << std::endl;
        std::cout << "\n" << std::endl;
        
        std::cout << xt::adapt(network.weights[i].shape()) << std::endl;
        std::cout << network.weights[i] << std::endl;
        std::cout << "\n\n\n" << std::endl;
    }

    network = NeuralNetwork({784, 32, 16, 10});

    // Network inference tests.
    std::string project_dir = R"(C:/users/andre/downloads/neuralnet/mnist)";
    NetworkData testing_data = LoadMnistDataPair(
        project_dir + "/t10k-images.idx3-ubyte",
        project_dir + "/t10k-labels.idx1-ubyte"
    );
    xarray<float> output = Inference(network, xt::view(testing_data.inputs, 0, xt::all()));
    std::cout << xt::adapt(output.shape()) << std::endl;
    std::cout << output << std::endl;
    std::cout << "\n\n\n" << std::endl;

    // Network testing tests.
    std::cout << Test(network, testing_data) << std::endl;
    std::cout << "\n\n\n" << std::endl;
}