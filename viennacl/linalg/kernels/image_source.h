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

const char * const image_add = " \n"
		"__kernel void add(\n"
		"          read_only image2d_t src1,\n"
		"          read_only image2d_t src2,\n"
		"		   write_only image2d_t dst)\n"
		"{ \n"
		"  int2 coord = (int2)(get_global_id(0), get_global_id(1)); // y - coord  \n"
		"  //int width = get_image_width(src1); \n"
		"  //const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates"
		"  //                    CLK_ADDRESS_CLAMP | //Clamp to zeros"
		"  //                    CLK_FILTER_NEAREST; //Don't interpolate \n"
		"  //float4 src1_val = read_imagef(src1,sampler,coord); \n"
		"  //float4 src2_val = read_imagef(src2,sampler,coord); \n"
		"  //float4 val=read_imagef(src1, sampler, coord); \n"
		"  //val.w=0;\n"
		"  //write_imagef(dst,coord,val); \n"
		"}\n"
		" \n"; //image_add


const char * const image_sub =  " \n"
		"__kernel void sub(\n"
		"          read_only image2d_t src1,\n"
		"          read_only image2d_t src2,\n"
		"		   write_only image2d_t dst)\n"
		"{ \n"
		"  int2 coord = (int2)(get_global_id(0), get_global_id(1)); // y - coord  \n"
		"  //int width = get_image_width(src1); \n"
		"  //const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates"
		"  //                    CLK_ADDRESS_CLAMP | //Clamp to zeros"
		"  //                    CLK_FILTER_NEAREST; //Don't interpolate \n"
		"  //float4 src1_val = read_imagef(src1,sampler,coord); \n"
		"  //float4 src2_val = read_imagef(src2,sampler,coord); \n"
		"  //float4 val=read_imagef(src1, sampler, coord); \n"
		"  //val.w=0;\n"
		"  //write_imagef(dst,coord,val); \n"
		"}\n"
		" \n"; //image_add
}
}
}

#endif /* _VIENNACL_IMAGE_SOURCE_HPP_ */
