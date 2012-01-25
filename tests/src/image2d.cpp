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

int test_image_subtraction()
{
    cimg_library::CImg<unsigned char> cImgSrcImg1("walle1.jpg");
    cimg_library::CImg<unsigned char> cImgSrcImg2("walle2.jpg");
    uint vector_spectrum = 4;

    std::vector<unsigned char> vector1 = cimg_to_vector(cImgSrcImg1, vector_spectrum);
    std::vector<unsigned char> vector2 = cimg_to_vector(cImgSrcImg2, vector_spectrum);

    viennacl::image2d<CL_RGBA, CL_UNORM_INT8> srcImg1(cImgSrcImg1.width(), cImgSrcImg1.height(), &(*vector1.begin()));
    viennacl::image2d<CL_RGBA, CL_UNORM_INT8> srcImg2(cImgSrcImg2.width(), cImgSrcImg2.height(), &(*vector2.begin()));

    viennacl::image2d<CL_RGBA, CL_UNORM_INT8> imgResult = srcImg1 - srcImg2;

    std::vector<unsigned char> v((pixelSize+1) * imgResult.width() * imgResult.height());
    imgResult.fast_copy_cpu(v.begin());
    cimg_library::CImg<unsigned char> resultImg = vector_to_cimg(v, imgResult.width(), imgResult.height(), pixelSize);
    resultImg.save("subtraction.bmp");
}

int test_gaussianBlur(viennacl::image2d<CL_RGBA, CL_UNORM_INT8> &image2)
{
    unsigned int kernelWitdth = 5;
    unsigned int kernelHeight = kernelWitdth;

    viennacl::image2d<CL_RGBA, CL_UNORM_INT8> gaussianFilteredImg  = image2.gaussian_filter<float>(kernelWitdth,10);
    std::vector<unsigned char> v((pixelSize+1) * gaussianFilteredImg.width() * gaussianFilteredImg.height());
    gaussianFilteredImg.fast_copy_cpu(v.begin());

    cimg_library::CImg<unsigned char> resultImg = vector_to_cimg(v, gaussianFilteredImg.width(), gaussianFilteredImg.height(), pixelSize);
    resultImg.save("milla2_gaussian_blur.bmp");

    return 0;
}

int test_grayscale(viennacl::image2d<CL_RGBA, CL_UNORM_INT8> &image2)
{
    std::cout<<std::endl<<"--------------------------"<<std::endl;
    viennacl::image2d<CL_LUMINANCE, CL_UNORM_INT8> grayImg = image2.grayscale<float>();

    std::vector<unsigned char> v(( pixelSize+1) * grayImg.width() * grayImg.height());
    grayImg.fast_copy_cpu(v.begin());

    cimg_library::CImg<unsigned char> resultImg = vector_to_cimg(v, image2.width(), image2.height(), 1);
    resultImg.save("gray_milla2.bmp");

    return 0;
}


int test_pyrup(viennacl::image2d<CL_RGBA, CL_UNORM_INT8> &image2)
{
    std::cout<<std::endl<<"--------------------------"<<std::endl;
    viennacl::image2d<CL_RGBA, CL_UNORM_INT8> pyrUpImg = image2.pyrUp();

    std::vector<unsigned char> v(( pixelSize+1) * pyrUpImg.width() * pyrUpImg.height());
    pyrUpImg.fast_copy_cpu(v.begin());
    cimg_library::CImg<unsigned char> resultImg = vector_to_cimg(v, pyrUpImg.width(), pyrUpImg.height(), pixelSize);
    std::cout<<"PyrUp Result Image size: "<<resultImg.width()<<"x"<<resultImg.height()<<std::endl;
    resultImg.save("pyrup_milla2.bmp");

    return 0;
}

int test_pyrdown(viennacl::image2d<CL_RGBA, CL_UNORM_INT8> &image2)
{

    std::vector<unsigned char> initialVector(( pixelSize+1) * image2.width() * image2.height());
    image2.fast_copy_cpu(initialVector.begin());
    cimg_library::CImg<unsigned char> initialImg = vector_to_cimg(initialVector, image2.width(), image2.height(), pixelSize);
    initialImg.save("image_afer_pyrup.bmp");


    std::cout<<std::endl<<"--------------------------"<<std::endl;
    viennacl::image2d<CL_RGBA, CL_UNORM_INT8> pyrDownImg = image2.pyrDown();

    std::vector<unsigned char> v(( pixelSize+1) * pyrDownImg.width() * pyrDownImg.height());
    pyrDownImg.fast_copy_cpu(v.begin());
    cimg_library::CImg<unsigned char> resultImg = vector_to_cimg(v, pyrDownImg.width(), pyrDownImg.height(), pixelSize);
    std::cout<<"PyrDown Result Image size: "<<resultImg.width()<<"x"<<resultImg.height()<<std::endl;
    resultImg.save("pyrdown_milla2.bmp");

    return 0;
}


//#define TestPyrUp
//#define TestPyrDown

#define TestGaussianBlur
#define TestGrayScale
//#define TestSubtraction

int test()
{
    int retval = EXIT_SUCCESS;
    std::cout << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << std::endl;

    cimg_library::CImg<unsigned char> srcImg("walle1.jpg");   
    uint vector_spectrum = 4;
    std::vector<unsigned char> srcImg2 = cimg_to_vector(srcImg, vector_spectrum);
    viennacl::image2d<CL_RGBA, CL_UNORM_INT8> image2(srcImg.width(), srcImg.height(), &(*srcImg2.begin()));

    std::vector<unsigned char> srcImg3(pixelSize * srcImg.width() * srcImg.height());
    srand ( time(NULL) );
    
#ifdef TestSubtraction
    test_image_subtraction();
#endif

#ifdef TestGaussianBlur
    test_gaussianBlur(image2);
#endif

#ifdef TestGrayScale
    test_grayscale(image2);
#endif


#ifdef TestPyrDown
    test_pyrdown(image2);
#endif

#ifdef TestPyrUp
    test_pyrup(image2);
#endif



    return retval;
}

int main()
{
    return test();
}

#endif /* IMAGE_CPP_ */
