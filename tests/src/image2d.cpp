/*
 * image.cpp
 *
 *  Created on: Aug 1, 2011
 *      Author: sanda
 */

#ifndef IMAGE_CPP_
#define IMAGE_CPP_

//
// *** System
//
#include <iostream>

//
// *** Boost
//
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/vector.hpp>
//
// *** ViennaCL
//
#define VIENNACL_DEBUG_ALL
#define VIENNACL_HAVE_UBLAS 1
#include "viennacl/scalar.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/image2d.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/direct_solve.hpp"
#include "examples/tutorial/Random.hpp"

#include <vector>

#define cimg_display 0
#include "../../external/CImg.h"
//
// -------------------------------------------------------------
//

//
// -------------------------------------------------------------
//

#include <time.h>
#include <stdlib.h>

#define VIENNACL_DEBUG_ALL

using namespace boost::numeric;

cimg_library::CImg<unsigned char> vector_to_cimg(std::vector<unsigned char> v, uint width, uint height, uint spectrum)
{
    cimg_library::CImg<unsigned char> resultImg(width, height, 1, spectrum);

    for (uint k=0; k < height; k++)
        for (uint i=0; i < width;i++)
        {
            for (uint j=0; j < spectrum; j++)
            {
                resultImg(i,k,j) = v[ k * width * spectrum + i * spectrum + j];
            }
        }
    return resultImg;
}

std::vector<unsigned char>  cimg_to_vector(const cimg_library::CImg<unsigned char> &cimg, uint vector_spectrum)
{
    uint imgWidth = cimg.width();
    uint imgHeight = cimg.height();
    uint imgSpectrum = cimg.spectrum();
    std::vector<unsigned char> v(vector_spectrum * imgWidth * imgHeight);
    for (uint k=0; k < imgHeight; k++)
        for (uint i=0; i< imgWidth; i++)
        {
            for (uint j=0; j < vector_spectrum; j++)
            {
                if ( j <  imgSpectrum )
                    v[ k * imgWidth * vector_spectrum + i * vector_spectrum + j] = cimg(i,k,j);
                else
                    v[ k * imgWidth * vector_spectrum + i * vector_spectrum + j] = 0;
            }
        }
    return v;
}

const int pixelSize = 4;

int test_grayscale(viennacl::image2d<CL_RGBA, CL_UNORM_INT8> &image2)
{
    std::cout<<"--------------------------"<<std::endl;
    std::vector<unsigned char> v(( pixelSize+1) * image2.width() * image2.height());
    viennacl::image2d<CL_LUMINANCE, CL_UNORM_INT8> grayImg = image2.grayscale<float>();
    grayImg.fast_copy_cpu(v.begin());
    cimg_library::CImg<unsigned char> resultImg = vector_to_cimg(v, image2.width(), image2.height(), 1);
    resultImg.save("gray_milla2.bmp");

    return 0;
}

int test()
{
    int retval = EXIT_SUCCESS;
    std::cout << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << std::endl;

    cimg_library::CImg<unsigned char> srcImg("milla.bmp");

    int imgWidth = srcImg.width();
    int imgHeight = srcImg.height();
    
    std::vector<unsigned char> srcImg3(pixelSize * imgWidth * imgHeight);

    srand ( time(NULL) );

    uint vector_spectrum = 4;
    std::vector<unsigned char> srcImg2 = cimg_to_vector(srcImg, vector_spectrum);

    viennacl::image2d<CL_RGBA, CL_UNORM_INT8> image2(imgWidth, imgHeight, &(*srcImg2.begin()));
    std::vector<unsigned char> v((pixelSize+1) * imgWidth * imgHeight);
    for (std::vector<unsigned char>::iterator iter=v.begin(); iter < v.end();++iter)
        *iter=0;

    unsigned int kernelWitdth = 21;
    unsigned int kernelHeight = kernelWitdth;

    viennacl::image2d<CL_RGBA, CL_UNORM_INT8> gaussianFilteredImg  = image2.gausian_filter<float>(kernelWitdth,10);
    gaussianFilteredImg.fast_copy_cpu(v.begin());

    cimg_library::CImg<unsigned char> resultImg = vector_to_cimg(v, srcImg.width(), srcImg.height(), srcImg.spectrum()+1);
    resultImg.save("milla2.bmp");

    test_grayscale(image2);
    
    return retval;
}

int main()
{
    return test();
}

#endif /* IMAGE_CPP_ */
