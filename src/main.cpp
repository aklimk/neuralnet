#include <chrono>
#include <thread>

#include "utilities.hpp"
#include "loadimages.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"

void TestUtilities() {
    xt::print_options::set_edge_items(100);
    xt::print_options::set_line_width(200);
    
    ProgressBar progress_bar = ProgressBar(100);
    for (int i = 0; i < 100; i++) {
        progress_bar.IncrementBar();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cout << "\n" << std::endl;

    xt::xarray<float> testing = {1.0, 5.3, 9.3};
    std::cout << testing << std::endl;
    xt::xarray<float> r0 = Sigmoid::Activation(testing);
    std::cout << r0 << std::endl;
    xt::xarray<float> r1 = Sigmoid::ActivationPrime(testing);
    std::cout << r1 << std::endl;
    std::cout << std::endl;

    std::string project_dir = R"(C:/users/andre/downloads/neuralnet/mnist)";
    NetworkData training_data = LoadMnistDataPair(
        project_dir + "/train-images.idx3-ubyte",
        project_dir + "/train-labels.idx1-ubyte"
    );
    NetworkData testing_data = LoadMnistDataPair(
        project_dir + "/t10k-images.idx3-ubyte",
        project_dir + "/t10k-labels.idx1-ubyte"
    );
    std::cout << training_data.inputs << "\n\n" << training_data.targets << std::endl;
    std::cout << std::endl;
    std::cout << testing_data.inputs << "\n\n" << testing_data.targets << std::endl;
}


int main() {
    TestUtilities();
}
