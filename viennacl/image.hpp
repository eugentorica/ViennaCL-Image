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
#include "viennacl/linalg/kernels/scalar_kernels.h"

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
template<class IMAGETYPE,unsigned int ALIGNMENT>
class image {
public:
	/** @brief Returns the underlying host scalar type. */
	//typedef typename viennacl::tools::CHECK_IMAGE_TEMPLATE_ARGUMENT<TYPE>::ResultType value_type;

	/** @brief Allocates the memory for the scalar, but does not set it to zero. */
	image() :_width(0), _height(0) {
		//viennacl::linalg::kernels::image<TYPE, 1>::init();
		//val_ = viennacl::ocl::current_context().create_memory(CL_MEM_READ_WRITE, sizeof(TYPE));
	}
	/** @brief Allocates the memory for the scalar and sets it to the supplied value. */
	explicit image(int width, int height) {
		//viennacl::linalg::kernels::image < TYPE, 1 > ::init();
		cl_image_format image_format;
		image_format.image_channel_data_type = CL_UNORM_INT8;
		image_format.image_channel_order = CL_RGBA;
		_pixels= viennacl::ocl::current_context().create_image2d(CL_MEM_READ_WRITE,&image_format,width,height);
	}

	/** @brief Returns the OpenCL handle */
	const viennacl::ocl::handle<cl_mem> & handle() const {
		return _pixels;
	}

private:
	viennacl::ocl::handle<cl_mem> _pixels;
	unsigned int _width;
	unsigned int _height;

};

} //namespace viennacl

#endif /* _VIENNACL_IMAGE_HPP_ */
