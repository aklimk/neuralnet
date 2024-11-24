A simple sigmoid based neural network, using a MSE loss function, written in c++.

# Building
The program utilizes xtensor, xtl (by dependency) and xtensor-blas. Theese libraries
will be automatically build from the central CMakeLists.

Building is done in CMake, with standard procedure.
I.E.
- Create a build directory.
- Run CMake from the build directory specify the path to the CMakeLists file.
- CMake will create the build files, simply run them to build the program.
