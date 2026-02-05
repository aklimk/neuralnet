# ~~~~ Project Configuration ~~~~
cmake_minimum_required(VERSION 3.21)
project(neuralnet LANGUAGES CXX)

# ~~~~ C++ Options ~~~~
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ~~~~ Dependency Management ~~~~
find_package(xtl CONFIG REQUIRED)
find_package(xtensor CONFIG REQUIRED)
find_package(xtensor-blas CONFIG REQUIRED)

# ~~~~ Neural Network Library ~~~~
add_subdirectory(src/)

# ~~~~ MNIST Example Program ~~~~
add_subdirectory(example/)
