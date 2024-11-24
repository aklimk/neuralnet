#pragma once
#include <iostream>
#include <math.h>
#include <string>
#include "xtensor/xarray.hpp"

using std::vector;
using xt::xarray;
using xt::xexpression;


// ~~~~ Basic Progress Bar ~~~~

/* Basic progress bar that takes a max value and provides and increment and display method.
 * 
 * At creation time a max value is specified. Calling IncrementBar increments the internal
 * value and displays the current bar.
 *
 * IncrementBar flushes the standard output.
*/
class ProgressBar
{
private:
	int max_value = 0;
	int current_value = 0;
			
public:
	ProgressBar(int max_value) {
		this->max_value = max_value;
	}

	void IncrementBar()
	{
		std::cout << "\r" << current_value << "/" << max_value;
		std::cout.flush();
		current_value++;
	}
};


// ~~~~ Activation Functions ~~~~

/*
 * Sigmoid compresses values to a range of [0, 1]. 
 * The compression creates an s like curve.
*/
namespace Sigmoid {
	/*
	 * Elementwise sigmoid operation on xtensor float array.
	 *
	 * Sigmoid is defined as 1 / (1 + e^-x).
	 *
	 * # Arguments
	 * `array` : Reference to xtensor array to create an xtensor expression from.
	 *
	 * # Returns
	 * An xtensor xexpression object. Xexpression objects are lazy evauluated, only
	 * being evaluated when assigned to a container type.
    */
	auto F(xarray<float>& array) {
		return 1.0 / (1.0 + xt::exp(-array));
	}
	
	/*
	 * Elementwise sigmoid derivative operation on xtensor float array.
	 *
	 * The sigmoid derivative is defined as S(x) * (1 - S(x)),
	 * where S is the sigmoid function.
	 *
	 * # Arguments
	 * `array` : Reference to xtensor array to create an xtensor expression from.
	 *
	 * # Returns
	 * An xtensor xexpression object. Xexpression objects are lazy evauluated, only
	 * being evaluated when assigned to a container type.
    */
	auto FPrime(xarray<float>& array) {
		return F(array) * (1 - F(array));
	}
};
