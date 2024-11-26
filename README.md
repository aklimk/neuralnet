A simple sigmoid based neural network, using a MSE loss function, written in c++.
The network supports stochastic gradient descent, inference and serializing/deserializing.

Using the MNIST handwritten digits database, the network reaches an accuracy over 98%
on the 10k sample unseen sample set, using a 784->256->128->64->10 network layer structure.

The project includes a CLI interface example for training the network with custom parameters against the 
MNIST database, and an option to run the network against a single image for classification.

There is an included python file `drawing-pad.py` that creates a drawing surface in pygame
and saves it in the IDX format, which can be classified using the example CLI interface.


# Building

The program utilizes xtensor, xtl (by dependency) and xtensor-blas. 

These libraries can either be included through the automatic package finding utility in cmake (find_package), 
which is compatiable with many common package managers.

Or the library source files can be included in a custom directory and pointed to using the following cmake flags:

The procedure for building the project is the standard procedure for cmake.<br/>
I.E.<br/>
- Create a build directory.<br/>
- Run CMake from the build directory specify the path to the CMakeLists file.<br/>
- CMake will create the build files, simply run them to build the program.<br/>

## Example

1. Make a build directory: `mkdir build`
2. Change directory into it: `cd build`
3. Run cmake on the source root: `cmake ..`
4. Add flags for source library inclusion: `cmake .. -DXTL_SOURCE_PATH=../libs/xtl -DXTENSOR_SOURCE_PATH=../libs/xtensor -DXTENSOR_BLAS_SOURCE_PATH=../libs/xtensor-blas`


# Running

The project is split into the network library and the example MNIST program. The library can be used by copying the static library and include headers.

In order to use the example program, the MNIST handwritten digit dataset needs to be downloaded and pointed within the CLI.

