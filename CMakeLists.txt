cmake_minimum_required(VERSION 3.10)
project(neuralnet)

add_executable(neuralnet src/main.cpp src/utilities.hpp src/loadimages.hpp)
add_subdirectory(libs/xtl)
add_subdirectory(libs/xtensor)

target_link_libraries(neuralnet xtensor)
