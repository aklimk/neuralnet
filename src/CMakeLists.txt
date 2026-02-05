# List library sources.
add_library(neuralnet network.cpp)

# Link libray to dependencies.
target_link_libraries(neuralnet PUBLIC
    xtensor
    xtensor-blas
)
target_include_directories(neuralnet PUBLIC ../include/neuralnet/)
