/*
 * image_source.h
 *
 *  Created on: Aug 2, 2011
 *      Author: sanda
 */

#ifndef _VIENNACL_IMAGE_SOURCE_HPP_
#define _VIENNACL_IMAGE_SOURCE_HPP_

namespace viennacl {
namespace linalg {
namespace kernels {
const char * const image_inplace_sub = " \n"
		"__kernel void inplace_sub(\n"
		"          __global float * val1,\n"
		"          __global const float * val2) \n"
		"{ \n"
		"  if (get_global_id(0) == 0)\n"
		"    *val1 -= *val2;\n"
		"}\n"
		" \n"; //image_inplace_sub
}
}
}

#endif /* _VIENNACL_IMAGE_SOURCE_HPP_ */
