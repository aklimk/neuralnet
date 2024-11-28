#pragma once

#include <xtensor/xarray.hpp>

/*
 * Sigmoid compresses values to a range of [0, 1]. 
 * The compression creates an s like curve.
*/
namespace Sigmoid {
	using xt::xarray;
	
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
