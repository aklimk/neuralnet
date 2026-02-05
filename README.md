# nueralnet
GitHub project that includes a neural network library with customizable layers, activations and minibatch based SGD trainer with customizable training hyperparameters.
Also includes an example program to create a datapipline and utilize the network library to decode MNIST handwritten digits.

# Features and Performance of the Network Library
Achieves an accuracy of 98.x% with a time-to-convergence of 00:xx:xx on the MNIST handwritten digit dataset.

Comparison of network accuracy on MNIST test set vs other techniques. Information 
courtesy of wikipedia. https://en.wikipedia.org/wiki/MNIST_database.
| Algorithm                          |  Training on Synthetic/Augmented Data?  | MNIST Accuracy |
| -------------                      |  -------------------------------------  | -------------  |
| Linear Classifier                  | No Training                             | 92.4           |
| Non-Linear Classifier              | No Training                             | 96.7           |
| Other Network (3-layer)            | no                                      | 98.4           |
| This Project  (4-layer)            | no                                      | 98.x           |
| Other Network (6-layer)            | yes                                     | 99.65          |
| CNN (13-layer)                     | no                                      | 99.75          |
| CNN (3 CNN Ensemble)               | yes                                     | 99.91          |

## List of Features
- Minibatch based SGD trainer with customizable batch sizes, epochs and learning rate.
    Achieves convergence xx times faster than a simpler non-batched trainer.
- Customizable number of layers and layer sizes.
- Customizable activation functions for newtork layers.
- Saving and loading of network.
- Initilization/Saving/Loading/Training and all customization done through library API that
    can be either statically or dynamically linked with applications.

# Building and Running
This project uses cmake to build both the network library and the MNIST example application.

Requires xtensor and xtensor-blas to be available as cmake compatable libraries.
The following sections describe different ways of making those libraries available to cmake.

## Using Vcpkg (Cross-Platform Instructions)
1. Install xtensor and xtensor-blas using vcpkg.</br>
```vcpkg install xtensor xtensor-blas```

3. Create a build directory and cd into it.</br>
```mkdir build && cd build```

5. Invoke cmake in release mode, and link to vcpkg installation.</br>
```cmake ../ -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake```

7. Build the project using the generated build files.</br>
(Make) ```make```</br>
(Visual Studio) ``` msbuild project.sln``` 


