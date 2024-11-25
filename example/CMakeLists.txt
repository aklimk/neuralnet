# List example binary sources.
add_executable(mnist_example main.cpp)

# Link example binary to neural network library.
target_link_libraries(mnist_example PRIVATE neuralnet)

# Link example binary to neural network include directory.
target_include_directories(mnist_example PRIVATE ../include/neuralnet)