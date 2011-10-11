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

const char * const image_add =
		" \n"
				"__kernel void add(\n"
				"          read_only image2d_t src1,\n"
				"          read_only image2d_t src2,\n"
				"		   write_only image2d_t dst)\n"
				"{ \n"
        "  int w = min(get_image_width(src1), get_image_width(src2));\n"
        "  int h = min(get_image_height(src1), get_image_height(src2));\n"
        "  int coordW = get_global_id(0); \n"
        "  int coordH = get_global_id(1); \n"
        "  int2 coord = (int2)(coordW , coordH); // y - coord  \n"
				"  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates \n"
				"                      CLK_ADDRESS_REPEAT | \n"
				"                      CLK_FILTER_NEAREST; //Don't interpolate \n"
				"  float4 src1_val = read_imagef(src1,sampler,coord); \n"
				"  float4 src2_val = read_imagef(src2,sampler,coord); \n"
				"  float4 val= src1_val + src2_val; \n"
        "  if(coordW < w && coordH < h)\n"
        "     write_imagef(dst,coord,val); \n"
				"}\n"
				" \n"; //image_add

const char * const image_sub =
    " \n"
		"__kernel void sub(\n"
    "          read_only image2d_t src1,\n"
    "          read_only image2d_t src2,\n"
    "      write_only image2d_t dst)\n"
    "{ \n"
    "  int w = min(get_image_width(src1), get_image_width(src2));\n"
    "  int h = min(get_image_height(src1), get_image_height(src2));\n"
    "  int coordW = get_global_id(0); \n"
    "  int coordH = get_global_id(1); \n"
    "  int2 coord = (int2)(coordW , coordH); // y - coord  \n"
    "  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates \n"
    "                      CLK_ADDRESS_REPEAT | \n"
    "                      CLK_FILTER_NEAREST; //Don't interpolate \n"
    "  float4 src1_val = read_imagef(src1,sampler,coord); \n"
    "  float4 src2_val = read_imagef(src2,sampler,coord); \n"
    "  float4 val= src1_val - src2_val; \n"
    "  if(coordW < w && coordH < h)\n"
    "     write_imagef(dst,coord,val); \n"
    "}\n"
		" \n"; //image_sub

const char* const image_gaussian_filter =
    "__kernel void gaussian_filter(read_only image2d_t srcImg,write_only image2d_t dstImg, \n"
    "     constant float * kernelWeights, float kernelTotalWeight,unsigned int kernelSize)\n"
    "{\n"

    " int width = get_image_width(dstImg); \n"
    " int height = get_image_height(dstImg); \n"
    " int dimX = get_global_size(0); \n"
    " int dimY = get_global_size(1); \n"
    " int kernelEdgeSize = floor(sqrt((float)kernelSize));\n"

    " int portionX = ceil(width / (float)dimX);\n"
    " int portionY = ceil(height / (float)dimY);\n"
    " const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates \n"
    "                      CLK_ADDRESS_REPEAT | \n"
    "                      CLK_FILTER_NEAREST; //Don't interpolate \n"
    " for(int n = 0; n < portionY; n++)"
    " {\n"
    "   for(int m = 0; m < portionX; m++)"
    "   {\n"
    "     int xCoord = get_global_id(0) + m; \n"
    "     int yCoord = get_global_id(1) + n; \n"
    "     int2 outImageCoord = (int2) (xCoord, yCoord);\n"
    "     if (outImageCoord.x < width && outImageCoord.y < height)\n"
    "     {\n"
    "       int2 startImageCoord = (int2) (xCoord - kernelEdgeSize / 2, yCoord - kernelEdgeSize / 2);\n"
    "       int2 endImageCoord = (int2) (xCoord + kernelEdgeSize / 2 , yCoord + kernelEdgeSize / 2);\n"
    "       int weight = 0;\n"
    "       float4 outColor = (float4)(0, 0, 0, 0);\n"
    "       for(int y = startImageCoord.y; y <= endImageCoord.y; y++)\n"
    "       {\n"
    "         for(int x= startImageCoord.x; x <= endImageCoord.x; x++)\n"
    "         {\n"
    "           outColor += (read_imagef(srcImg, sampler, (int2)(x, y)) * (kernelWeights[weight] / (float)kernelTotalWeight));\n"
    "           weight += 1;\n"
    "         }\n"
    "       }\n"
    "       // Write the output value to image\n"
    "       write_imagef(dstImg, outImageCoord, outColor);\n"
    "     }\n"
    "   }\n"
    " }\n"
    "}\n";

}
}
}

#endif /* _VIENNACL_IMAGE_SOURCE_HPP_ */
