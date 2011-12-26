
#ifndef _VIENNACL_IMAGE_SOURCE_HPP_
#define _VIENNACL_IMAGE_SOURCE_HPP_

namespace viennacl {
namespace linalg {
namespace kernels {

const char * const image2d_float_add =
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
    " \n"; 

const char * const image2d_float_sub =
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
    " \n";

const char* const image2d_float_convolute =
    "__kernel void convolute(read_only image2d_t srcImg,write_only image2d_t dstImg, \n"
    "     constant float * kernelWeights, float kernelTotalWeight,unsigned int kernelWidth, unsigned int kernelHeight)\n"
    "{\n"

    " int width = get_image_width(dstImg); \n"
    " int height = get_image_height(dstImg); \n"
    " int dimX = get_global_size(0); \n"
    " int dimY = get_global_size(1); \n"
    
    " int portionX = ceil(width / (float)dimX);\n"
    " int portionY = ceil(height / (float)dimY);\n"
    " const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates \n"
    "                      CLK_ADDRESS_REPEAT | \n"
    "                      CLK_FILTER_NEAREST; //Don't interpolate \n"
    " int glXCoord = get_global_id(0); \n"
    " int glYCoord = get_global_id(1); \n"
    " for(int n = 0; n <= portionY; n++)"
    " {\n"
    "   for(int m = 0; m <= portionX; m++)"
    "   {\n"
    "     int xCoord = glXCoord * portionX + m; \n"
    "     int yCoord = glYCoord * portionY + n; \n"
    "     int2 outImageCoord = (int2) (xCoord, yCoord);\n"
    "     if (outImageCoord.x < width && outImageCoord.y < height)\n"
    "     {\n"
    "       int2 startImageCoord = (int2) (xCoord - kernelWidth / 2, yCoord - kernelWidth / 2);\n"
    "       int2 endImageCoord = (int2) (xCoord + kernelHeight / 2 , yCoord + kernelHeight / 2);\n"
    "       int kernelIndex = 0;\n"
    "       float4 outColor = (float4)(0, 0, 0, 0);\n"
    "       for(int y = startImageCoord.y; y <= endImageCoord.y; y++)\n"
    "       {\n"
    "         for(int x= startImageCoord.x; x <= endImageCoord.x; x++)\n"
    "         {\n"
    "           outColor += (read_imagef(srcImg, sampler, (int2)(x, y)) * (kernelWeights[kernelIndex] / (float)kernelTotalWeight) );\n"
    "           kernelIndex += 1;\n"
    "         }\n"
    "       }\n"

    "       // Write the output value to image\n"
    "       write_imagef(dstImg, outImageCoord, outColor );\n"
    "     }\n"
    "   }\n"
    " }\n"
    "}\n";


const char* const morphologicalOperation =
    "__kernel void filter2D(read_only image2d_t srcImg,write_only image2d_t dstImg, \n"
    "     constant float * kernelWeights, unsigned int kernelSize, int operationType,int nrIterations)\n"
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
    " int glXCoord = get_global_id(0); \n"
    " int glYCoord = get_global_id(1); \n"
    " for(int n = 0; n <= portionY; n++)"
    " {\n"
    "   for(int m = 0; m <= portionX; m++)"
    "   {\n"
    "     int xCoord = glXCoord * portionX + m; \n"
    "     int yCoord = glYCoord * portionY + n; \n"
    "     int2 outImageCoord = (int2) (xCoord, yCoord);\n"

    "     if (outImageCoord.x < width && outImageCoord.y < height)\n"
    "     {\n"
    "       int2 startImageCoord = (int2) (xCoord - kernelEdgeSize / 2, yCoord - kernelEdgeSize / 2);\n"
    "       int2 endImageCoord = (int2) (xCoord + kernelEdgeSize / 2 , yCoord + kernelEdgeSize / 2);\n"
    "       int kernelIndex = 0;\n"
    "       float4 outColor = (float4)(0, 0, 0, 0);\n"

    "       for(int y = startImageCoord.y; y <= endImageCoord.y; y++)\n"
    "       {\n"
    "         for(int x= startImageCoord.x; x <= endImageCoord.x; x++)\n"
    "         {\n"
    "           if(kernelWeights[kernelIndex] == 0) \n"
    "             continue; //Zero value means skip current pixel value\n "
    "           float4 srcColor = (read_imagef(srcImg, sampler, (int2)(x, y));\n"
    "           if(operationType == 0){  //erode min \n"
    "             outColor.x = min(srcColor.x, outColor.x); \n"
    "             outColor.y = min(srcColor.y, outColor.y); \n"
    "             outColor.z = min(srcColor.z, outColor.z); \n"
    "             outColor.w = min(srcColor.w, outColor.w); \n"
    "           }\n"
    "           if(operationType == 1){  //dilate max \n"
    "             outColor.x = max(srcColor.x, outColor.x); \n"
    "             outColor.y = max(srcColor.y, outColor.y); \n"
    "             outColor.z = max(srcColor.z, outColor.z); \n"
    "             outColor.w = max(srcColor.w, outColor.w); \n"
    "           }\n"
    "           kernelIndex += 1;\n"
    "         }\n"
    "       }\n"

    "       // Write the output value to image\n"
    "       write_imagef(dstImg, outImageCoord, outColor );\n"
    "     }\n"

    "   }\n"
    " }\n"
    "}\n";

}
}
}

#endif /* _VIENNACL_IMAGE_SOURCE_HPP_ */
