/*
 * image.hpp
 *
 *  Created on: Aug 1, 2011
 *      Author: Eugen
 *      Email: eugen.torica@gmail.com
 */

#ifndef _VIENNACL_IMAGE_HPP_
#define _VIENNACL_IMAGE_HPP_

#include "viennacl/forwards.h"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/linalg/image_operations.hpp"

#include <iostream>

namespace viennacl {
/** @brief A proxy for scalar expressions (e.g. from inner vector products)
 *
 * assumption: dim(LHS) >= dim(RHS), where dim(scalar) = 0, dim(vector) = 1 and dim(matrix = 2)
 * @tparam LHS   The left hand side operand
 * @tparam RHS   The right hand side operand
 * @tparam OP    The operation tag
 */

/** @brief This class represents a single scalar value on the GPU and behaves mostly like a built-in scalar type like float or double.
 *
 * Since every read and write operation requires a CPU->GPU or GPU->CPU transfer, this type should be used with care.
 * The advantage of this type is that the GPU command queue can be filled without blocking read operations.
 *
 * @tparam TYPE  Either float or double. Checked at compile time.
 */
template<cl_channel_order CHANNEL_ORDER, cl_channel_type CHANNEL_TYPE>
class image {
public:

	/** @brief */
	image() :
			_width(0), _height(0) {
		viennacl::linalg::kernels::image<CHANNEL_ORDER, CHANNEL_TYPE>::init();
	}

	/** @brief */
	explicit image(int width, int height) {
		viennacl::linalg::kernels::image<CHANNEL_ORDER, CHANNEL_TYPE>::init();
		cl_image_format image_format;
		image_format.image_channel_data_type = CHANNEL_TYPE;
		image_format.image_channel_order = CHANNEL_ORDER;
		_pixels = viennacl::ocl::current_context().create_image2d(
				CL_MEM_READ_WRITE, &image_format, width, height);
	}

	/** @brief Returns the OpenCL handle */
	const viennacl::ocl::handle<cl_mem> & handle() const {
		return _pixels;
	}

	image<CHANNEL_ORDER, CHANNEL_TYPE> & operator -=(
			const image<CHANNEL_ORDER, CHANNEL_TYPE> & other) {
		viennacl::linalg::inplace_sub(*this, other);
		return *this;
	}

private:
	viennacl::ocl::handle<cl_mem> _pixels;
	unsigned int _width;
	unsigned int _height;

};

} //namespace viennacl

#endif /* _VIENNACL_IMAGE_HPP_ */
