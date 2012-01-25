#ifndef _VIENNACL_IMAGE_TOOLS_HPP_
#define _VIENNACL_IMAGE_TOOLS_HPP_


#include <string>
#include <fstream>
#include <sstream>
#include "viennacl/forwards.h"

#include "viennacl/ocl/backend.hpp"
#include <vector>
#include <math.h>

namespace viennacl
{
namespace tools
{
namespace traits
{

template <typename KERNEL_ELEMENT_TYPE>
std::vector<KERNEL_ELEMENT_TYPE> getGrayScaleCoefficients(cl_channel_order CHANNEL_ORDER )
{
    std::vector<KERNEL_ELEMENT_TYPE> result;
    switch (CHANNEL_ORDER)
    {
    case CL_RGB:
        result.push_back(0.3);
        result.push_back(0.59);
        result.push_back(0.11);
        break;
    case CL_RGBx:
    case CL_RGBA:
        result.push_back(0.3);
        result.push_back(0.59);
        result.push_back(0.11);
        result.push_back(0);
        break;
    case CL_BGRA:
        result.push_back(0.11);
        result.push_back(0.59);
        result.push_back(0.3);
        result.push_back(0);
        break;
    case CL_ARGB:
        result.push_back(0);
        result.push_back(0.3);
        result.push_back(0.59);
        result.push_back(0.11);
        break;

    default:
        throw "Unexpected cl_channel_order"+CHANNEL_ORDER;
    }
    return result;

}

//http://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm
template<typename KERNEL_ELEMENT_TYPE>
std::vector<KERNEL_ELEMENT_TYPE> getGaussianKernel(unsigned int kernelSize, double sigma)
{
    if (kernelSize % 2 != 1)
        throw "ViennaCL: kernelSize must be odd number";

    if (sigma <= 0)
        sigma =  0.3*(kernelSize/2 - 1) + 0.8;

    double sigmaSquare = sigma * sigma;
    double pi =  4.0 * atan(1.0);
    double coefficientSum = 0.0;

    std::vector<KERNEL_ELEMENT_TYPE> result(kernelSize*kernelSize);
    for (uint i = 0; i< kernelSize;i++)
    {
        for (uint j = 0; j < kernelSize; j++)
        {
            int x =  i - kernelSize / 2;
            int y =  j - kernelSize / 2;
            double coefficient =  (x * x + y * y) / (2 * sigmaSquare);
            result[ i * kernelSize + j] = exp( - 1 * coefficient)/( 2 * pi * sigmaSquare);
            coefficientSum += result[ i * kernelSize + j];
        }
    }

    typename std::vector<KERNEL_ELEMENT_TYPE>::iterator it;
    for (it = result.begin(); it!= result.end();it++)
    {
        *it = *it / coefficientSum;
    }

    return result;
}


//http://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm
//http://opencv.itseez.com/doc/tutorials/imgproc/pyramids/pyramids.html
std::vector<float> getPyramidKernel()
{
    /* OpenCV documentation -
     *                       |1  4   6  4  1|
     *                       |4 16  24 16  4|
     *               1/256 * |6 24  36 24  6|
     *                       |4 16  24 16  4|
     *                       |1  4   6  4  1|
     */
    static const float row1[]= {1,  4,   6,  4,  1};
    static const float row2[]= {4, 16,  24, 16,  4};
    static const float row3[]= {6, 24,  36, 24,  6};
    static std::vector<float> result;
    result.insert(result.end(),row1, row1 + sizeof row1 / sizeof row1[ 0 ]);
    result.insert(result.end(),row2, row2 + sizeof row2 / sizeof row2[ 0 ]);
    result.insert(result.end(),row3, row3 + sizeof row3 / sizeof row3[ 0 ]);
    result.insert(result.end(),row2, row2 + sizeof row2 / sizeof row2[ 0 ]);
    result.insert(result.end(),row1, row1 + sizeof row1 / sizeof row1[ 0 ]);
    
    return result;
}


}
}
}

#endif


