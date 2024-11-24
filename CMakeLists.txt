cmake_minimum_required(VERSION 3.10)
project(neuralnet)

add_executable(neuralnet src/main.cpp src/utilities.hpp src/loadimages.hpp src/neuralnetwork.hpp)
add_subdirectory(libs/xtl)
add_subdirectory(libs/xtensor)
add_subdirectory(libs/xtensor-blas)

add_definitions(-DXTENSOR_USE_FLENS_BLAS)
target_link_libraries(neuralnet xtensor xtensor-blas)
