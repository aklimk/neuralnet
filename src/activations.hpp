#pragma once

#include <xtensor/containers/xarray.hpp>

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
	auto F(const xarray<float>& array) {
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
	 * An materialized set of values. Avoiding lazy expressions allows for caching
	 * the result of F().
    */
	xarray<float> FPrime(const xarray<float>& array) {
	    auto f_array = xt::eval(F(array));
		return f_array * (1 - f_array);
	}
};
