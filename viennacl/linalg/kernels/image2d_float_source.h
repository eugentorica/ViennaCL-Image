
#ifndef _VIENNACL_IMAGE_SOURCE_HPP_
#define _VIENNACL_IMAGE_SOURCE_HPP_

namespace viennacl {
namespace linalg {
namespace kernels {

const char * const image2d_float_add =
"__kernel void add( read_only image2d_t src1, read_only image2d_t src2, \n"
"		   write_only image2d_t dst)\n"
"{ \n"
"	int2 dstImgDim = get_image_width(src2);\n"
"	int width = min(get_image_width(src1), get_image_width(src2));\n"
"	int height = min(get_image_height(src1), get_image_height(src2));\n"
"	int portionX = ceil(width / (float)get_global_size(0));\n"
"	int portionY = ceil(height / (float)get_global_size(1));\n"
"	int globalXId = get_global_id(0);\n"
"	int globalYId = get_global_id(1);\n"
"	int startXPixel = globalXId * portionX;\n"
"	int endXPixel = (globalXId + 1) * portionX;\n"
"	int startYPixel = globalYId * portionY;\n"
"	int endYPixel = (globalYId + 1) * portionY;\n"
"	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates \n"
"				  CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST; //Don't interpolate \n"
"	for(int n = startYPixel; n < endYPixel; n++){\n"
"		for(int m = startXPixel; m < endXPixel; m++){\n"
"			int2 coord = (int2)(m , n); // y - coord  \n"
"			float4 src1_val = read_imagef(src1, sampler, coord); \n"
"			float4 src2_val = read_imagef(src2, sampler, coord); \n"
"			float4 val= src1_val + src2_val; \n"
"			// verify if destination coordinates are in range\n"
"			if( m < dstImgDim.x && n < dstImgDim.y)\n"
"				write_imagef(dst, coord, val); \n"
"		}\n"
"	}\n"
"}\n";

const char * const image2d_float_sub =
"__kernel void sub( read_only image2d_t src1, read_only image2d_t src2, \n"
"		   write_only image2d_t dst)\n"
"{\n"
"	int2 dstImgDim = get_image_width(src2);\n"
"	int width = min(get_image_width(src1), get_image_width(src2));\n"
"	int height = min(get_image_height(src1), get_image_height(src2));\n"
"	int portionX = ceil(width / (float)get_global_size(0));\n"
"	int portionY = ceil(height / (float)get_global_size(1));\n"
"	int globalXId = get_global_id(0);\n"
"	int globalYId = get_global_id(1);\n"
"	int startXPixel = globalXId * portionX;\n"
"	int endXPixel = (globalXId + 1) * portionX;\n"
"	int startYPixel = globalYId * portionY;\n"
"	int endYPixel = (globalYId + 1) * portionY;\n"
"	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates \n"
"					  CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST; //Don't interpolate \n"
"	for(int n = startYPixel; n < endYPixel; n++){\n"
"		for(int m = startXPixel; m < endXPixel; m++){\n"
"			int2 coord = (int2)(m , n); // y - coord  \n"
"			float4 src1_val = read_imagef(src1, sampler, coord); \n"
"			float4 src2_val = read_imagef(src2, sampler, coord); \n"
"			float4 val= (0, 0, 0, 0); \n"
"			val.x = max(src1_val.x - src2_val.x, src2_val.x - src1_val.x); \n"
"			val.y = max(src1_val.y - src2_val.y, src2_val.y - src1_val.y); \n"
"			val.z = max(src1_val.z - src2_val.z, src2_val.z - src1_val.z); \n"
"			val.w = max(src1_val.w - src2_val.w, src2_val.w - src1_val.w); \n"
"			// verify if destination coordinates are in range\n"
"			if( m < dstImgDim.x && n < dstImgDim.y)\n"
"				write_imagef(dst, coord, val); \n"
"		}\n"
"	}\n"
"}\n";

const char* const image2d_float_convolute =
    "__kernel void convolute(read_only image2d_t srcImg,write_only image2d_t dstImg, \n"
    "     constant float * kernelWeights, float kernelTotalWeight,unsigned int kernelWidth, unsigned int kernelHeight)\n"
    "{\n"
    " const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates \n"
    "                      CLK_ADDRESS_REPEAT | \n"
    "                      CLK_FILTER_NEAREST; //Don't interpolate \n"
    " const int halfKernelWidth = floor(kernelWidth / (float)2); \n"
    " const int halfKernelHeigh = floor(kernelHeight / (float)2); \n"
    " int width = get_image_width(dstImg); \n"
    " int height = get_image_height(dstImg); \n"
    " int dimX = get_global_size(0); \n"
    " int dimY = get_global_size(1); \n"

    " int portionX = ceil(width / (float)dimX);\n"
    " int portionY = ceil(height / (float)dimY);\n"
    " int globalXId = get_global_id(0); \n"
    " int globalYId = get_global_id(1); \n"

    " int startXPixel = globalXId * portionX; \n"
    " int endXPixel = (globalXId + 1) * portionX; \n"

    " int startYPixel = globalYId * portionY; \n"
    " int endYPixel = (globalYId + 1) * portionY; \n"

    " for(int n = startYPixel; n < endYPixel; n++)"
    " {\n"
    "   for(int m = startXPixel; m < endXPixel; m++)"
    "   {\n"
    "     int2 outImageCoord = (int2) (m, n); \n"
    "     if (outImageCoord.x < width && outImageCoord.y < height)\n"
    "     {\n"
    "       int2 startImageCoord = (int2) (m - halfKernelWidth, n - halfKernelHeigh);\n"
    "       int2 endImageCoord = (int2) (m + halfKernelWidth, n + halfKernelHeigh);\n"
    "       int kernelIndex = 0;\n"
    "       float4 outColor = (float4)(0, 0, 0, 0);\n"
    "       for(int y = startImageCoord.y; y <= endImageCoord.y; y++)\n"
    "       {\n"
    "         for(int x= startImageCoord.x; x <= endImageCoord.x; x++)\n"
    "         {\n"
    "           float4 val = read_imagef(srcImg, sampler, (int2)(x, y));\n"
    "           outColor += val * ( (float)kernelWeights[kernelIndex] / (float)kernelTotalWeight);\n"
    "           kernelIndex += 1;\n"
    "         }\n"
    "       }\n"

    "       write_imagef(dstImg, outImageCoord, outColor );\n"
    "     }\n"
    "   }\n"
    " }\n"
    "}\n";

const char* const image2d_float_grayscale =
    "__kernel void grayscale(read_only image2d_t srcImg,write_only image2d_t dstImg, \n"
    "     constant float * colorWeights )\n"
    "{\n"
    " const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates \n"
    "                      CLK_ADDRESS_REPEAT | \n"
    "                      CLK_FILTER_NEAREST; //Don't interpolate \n"
    " int width = get_image_width(dstImg); \n"
    " int height = get_image_height(dstImg); \n"

    " int portionX = ceil(width / (float)get_global_size(0));\n"
    " int portionY = ceil(height / (float)get_global_size(1));\n"

    " int globalXId = get_global_id(0); \n"
    " int globalYId = get_global_id(1); \n"

    " int startXPixel = globalXId * portionX; \n"
    " int endXPixel = (globalXId + 1) * portionX; \n"

    " int startYPixel = globalYId * portionY; \n"
    " int endYPixel = (globalYId + 1) * portionY; \n"
    " /*Remove odd rows and columns. Left rows and columns compress into a new image*/ \n"
    " for(int n = startYPixel; n < endYPixel; n++)"
    " {\n"
    "   for(int m = startXPixel; m < endXPixel; m++)"
    "   {\n"
    "     int2 outImageCoord = (int2) (m, n);\n"
    "     if (outImageCoord.x < width && outImageCoord.y < height)\n"
    "     {\n"
    "       float4 currentColor = (float4)read_imagef(srcImg, sampler, outImageCoord);\n"
    "       float dstColor = 0;\n"
    "       dstColor += currentColor.x * colorWeights[0]; \n"
    "       dstColor += currentColor.y * colorWeights[1]; \n"
    "       dstColor += currentColor.z * colorWeights[2]; \n"
    "       // Write the output value to image\n"
    "       write_imagef(dstImg, outImageCoord, dstColor );\n"
    "     }\n"
    "   }\n"
    " }\n"
    "}\n";

const char* const  image2d_float_pyrup =
    "__kernel void pyrup(read_only image2d_t srcImg,write_only image2d_t dstImg )\n"
    "{\n"
    " int2 srcImgDim = get_image_dim(srcImg); \n"
    " int2 portion = (int2)(ceil(srcImgDim.x / (float)get_global_size(0)) , \n"
    "                       ceil( srcImgDim.y / (float)get_global_size(1)) );\n"

    " const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates \n"
    "                      CLK_ADDRESS_REPEAT | \n"
    "                      CLK_FILTER_NEAREST; //Don't interpolate \n"
    " int globalXId = get_global_id(0); \n"
    " int globalYId = get_global_id(1); \n"
    " int2 startPixel =(int2)(globalXId * portion.x, globalYId * portion.y); \n"
    " int2 endPixel =(int2)((globalXId + 1) * portion.x, (globalYId + 1) * portion.y); \n"
    " /*Remove odd rows and columns. Left rows and columns compress into a new image*/ \n"
    " for(int n = startPixel.y; n < endPixel.y; n+=2)"
    " {\n"
    "  for(int m = startPixel.x; m < endPixel.x; m+=2)"
    "  {\n"
    "     int2 srcImageCoord = (int2) (m, n);\n"
    "     int2 outImageCoord = srcImageCoord / 2;\n"
    "     if (srcImageCoord.x < srcImgDim.x && srcImageCoord.y < srcImgDim.y)\n"
    "     {\n"
    "       float4 currentColor = (float4)read_imagef(srcImg, sampler, srcImageCoord);\n"
    "       // Write the output value to image\n"
    "       write_imagef(dstImg, outImageCoord, currentColor );\n"
    "     }\n"
    "   }\n"
    " }\n"
    "}\n";

const char* const  image2d_float_pyrdown =
    "__kernel void pyrdown(read_only image2d_t srcImg,write_only image2d_t dstImg )\n"
    "{\n"
    " const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates \n"
    "                      CLK_ADDRESS_REPEAT | \n"
    "                      CLK_FILTER_NEAREST; //Don't interpolate \n"

    " int2 srcImgDim = get_image_dim(srcImg); \n"
    " int2 dstImgDim = get_image_dim(dstImg); \n"

    " int2 portion = (int2)(ceil(srcImgDim.x / (float)get_global_size(0)) , \n"
    "                       ceil( srcImgDim.y / (float)get_global_size(1)) );\n"
    " int globalXId = get_global_id(0); \n"
    " int globalYId = get_global_id(1); \n"

    " int2 startPixel =(int2)(globalXId * portion.x, globalYId * portion.y); \n"
    " int2 endPixel =(int2)((globalXId + 1) * portion.x, (globalYId + 1) * portion.y); \n"
    
    " for(int n = startPixel.y; n < endPixel.y; n++)"
    " {\n"
    "  for(int m = startPixel.x; m < endPixel.x; m++)"
    "  {\n"
    "     int2 srcImageCoord = (int2) (m, n);\n"
    "     int2 outImageCoord = (int2) ( 2 * m, 2 * n);\n"
    "     if (srcImageCoord.x < srcImgDim.x && srcImageCoord.y < srcImgDim.y)\n"
    "     {\n"
    "       float4 currentColor = (float4)read_imagef(srcImg, sampler, srcImageCoord);\n"
    "       write_imagef(dstImg, outImageCoord, currentColor );\n"
    
    "       if (outImageCoord.x < dstImgDim.x - 1 && outImageCoord.y < dstImgDim.y - 1)\n"
    "       {\n"
    "         float4 zeroColor = (float4)0;\n"
    "         int2 neighbourCornerOutImageCoord = (int2) ( outImageCoord.x + 1, outImageCoord.y + 1 ) ;\n"
    "         write_imagef(dstImg, neighbourCornerOutImageCoord, zeroColor );\n"

    "         int2 neighbourRightOutImageCoord = (int2) ( outImageCoord.x + 1, outImageCoord.y ) ;\n"
    "         write_imagef(dstImg, neighbourRightOutImageCoord, zeroColor );\n"

    "         int2 neighbourBottomOutImageCoord = (int2) ( outImageCoord.x , outImageCoord.y + 1 ) ;\n"
    "         write_imagef(dstImg, neighbourBottomOutImageCoord, zeroColor );\n"
    "       }\n"
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


