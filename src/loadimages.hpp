#pragma once
#include <string>
#include <fstream>
#include <stdint.h>
#include <vector>

#include "xtensor/xarray.hpp"
#include "xtensor/xadapt.hpp"

using std::vector;
using std::string;
using xt::xarray;

/*
 * Structure that holds input and target data as 2d arrays, for a neural network.
 *
 * inputs is a 2d array, corresponding to an array of images. Each image is greyscale
 * float encoded, with dimensions 28x28. It corresponds to the network input layer.
 *
 * targets is a 2d array, corresponding to an array of network output layer arrays. 
 * targets are the ground truth network response. It is the correct values for the 
 * network output layer for each set of inputs.
*/
struct NetworkData {
	xarray<float> inputs;
	xarray<float> targets;
};


/*
 * Reads 16 bits from a file stream. 
*/
uint16_t Read16FromStream(std::ifstream& input_stream) {
	uint8_t temp16[2];
	input_stream.read((char*)temp16, 2);
	return (temp16[0] << 8) | temp16[1];
}

/*
 * Reads 32 bits from a file stream.
*/
uint32_t Read32FromStream(std::ifstream& input_stream) {
	uint8_t temp32[4];
	input_stream.read((char*)temp32, 4);
	return (temp32[0] << 24) | (temp32[1] << 16) | (temp32[2] << 8) | (temp32[3]);
}

/*
 * Reads array of byte values from IDX file. 
 *
 * IDX files store multidimensional arrays in a variety of data types 
 * in a simple contigious format. Only unsigned byte are considered here.
 *
 * # Invariants
 * Assumes the IDX file is encoded with the correct formatting.
 *
 * Assumes the array type is unsigned bytes.
 *
 * Assumes the array dimensions are >= 1.
 *
 * # Arguments
 * `file_path` : Path to the idx file.
 *
 * # Returns
 * An xarray with the shape specified in the IDX file.
*/
xarray<uint8_t> ReadIDX(string file_path)
{
	std::ifstream input_stream;
	input_stream.open(file_path, std::ifstream::in | std::fstream::binary);

	// File type indicator.
	uint16_t padding = Read16FromStream(input_stream);

	// Array data type, uint8 support only.
	uint8_t data_type = input_stream.get();

	// Number of dimensions.
	uint8_t dimension_count = input_stream.get();

	// Construct overall array shape and keep track of 
	// the total element count.
	int total_elements = 1;
	vector<uint16_t> dimension_sizes;
	for (int i = 0; i < dimension_count; i++) {
	    uint32_t dimension_size = Read32FromStream(input_stream);
	    dimension_sizes.push_back(dimension_size);
	    total_elements *= dimension_size;
	}

	// Read file data into an buffer.
	vector<uint8_t> buffer(total_elements, 0);
	input_stream.read((char*)&(buffer[0]), total_elements);
	
	// Initilize xarray with constructed shape and buffer.
	xarray<uint8_t> output = xt::adapt(buffer, dimension_sizes);

	return output;
}


/*
 * Loads images and targets from mnist data files for either training or testing.
 *
 * # Arguments
 * `images_path` : Path to the images mnist file.
 *
 * `targets_path` : Path to the targets mnsit file.
 *
 * # Returns
 * A NetworkData struct containing a 2d array of flattened images and their 
 * oneshot encoded targets.
*/
NetworkData LoadMnistDataPair(string images_path, string targets_path) {

	// Read unformatted values from IDX files.
	xarray<uint8_t> images_raw = ReadIDX(images_path);
	xarray<uint8_t> targets_raw = ReadIDX(targets_path);

	// Create formatted arrays.
	xarray<float>::shape_type images_shape = {images_raw.shape(0), 784};
	xarray<float> images(images_shape);
	
	xarray<float>::shape_type targets_shape = {targets_raw.shape(0), 10};
	xarray<float> targets(targets_shape);
	targets = xt::zeros<float>(targets.shape());

	// Loop through raw images and turn greyscale pixel values into 
	// normalized floats. Flatten rows and columns into single dimesnion.
	for (int i = 0; i < images_raw.shape(0); i++)
	{
		for (int j = 0; j < images_raw.shape(1); j++)
		{
			for (int k = 0; k < images_raw.shape(2); k++) {
				images(i, j * 28 + k) = (float)images_raw(i, j, k) / 255;
			}
		}
	}

	// Loop through targets and oneshot encode all of the numerical values.
	for (int i = 0; i < targets_raw.shape()[0]; i++)
	{
	    targets(i, targets_raw(i)) = 1.0;
	}

	return NetworkData {images, targets};
}

