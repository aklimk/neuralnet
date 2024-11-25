cmake_minimum_required(VERSION 3.10)
project(neuralnet)

# ~~~~ Dependencies ~~~~

# Give the option of using source or pre-build dependencies
set(XTL_SOURCE_PATH "" CACHE PATH "Path to xtl source directory.")
set(XTENSOR_SOURCE_PATH "" CACHE PATH "Path to xtensor source directory.")
set(XTENSOR_BLAS_SOURCE_PATH "" CACHE PATH "Path to xtensor-blas source directory.")

# Handle dependencies, using source if specified, otherwise
# using package version.
message("${XTL_SOURCE_PATH}")
message("${XTENSOR_SOURCE_PATH}")
message("${XTENSOR_BLAS_SOURCE_PATH}")

if(XTL_SOURCE_PATH)
	add_subdirectory("${XTL_SOURCE_PATH}")
else()
	find_package(xtl QUIET)
endif()

if(XTENSOR_SOURCE_PATH)
	add_subdirectory("${XTENSOR_SOURCE_PATH}")
else()
	find_package(xtensor QUIET)
endif()

if(XTENSOR_BLAS_SOURCE_PATH)
	add_subdirectory("${XTENSOR_BLAS_SOURCE_PATH}")
else()
	find_package(xtensor-blas QUIET)
endif()

# Use inbuild BLAS for xtensor-blas
add_definitions(-DXTENSOR_USE_FLENS_BLAS)




# ~~~~ Neural Network Library ~~~~

add_subdirectory(src/)



# ~~~~ Neural Network MNIST Handwritten Digits Example Program ~~~~

# Check if building the example was enabled.
set(BUILD_EXAMPLE "Build MNIST example program." ON)
if(BUILD_EXAMPLE)
	add_subdirectory(example/)
endif()



