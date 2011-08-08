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
#include <boost/numeric/ublas/io.hpp>

//
// *** ViennaCL
//
#define VIENNACL_DEBUG_ALL
#define VIENNACL_HAVE_UBLAS 1
#include "viennacl/scalar.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/image.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/direct_solve.hpp"
#include "examples/tutorial/Random.hpp"

//#define cimg_display 0
//#include "../lib/CImg.h"
//#include "../lib/SDKUtil/include/SDKCommon.hpp"
//#include "../lib/SDKUtil/include/SDKApplication.hpp"
//#include "../lib/SDKUtil/include/SDKFile.hpp"
#include "../lib/SDKUtil/SDKBitMap.cpp"


//
// -------------------------------------------------------------
//

//
// -------------------------------------------------------------
//
#define VIENNACL_DEBUG_ALL

using namespace std;

int test()
{
	int retval = EXIT_SUCCESS;
	std::cout << std::endl;
	std::cout << "----------------------------------------------" << std::endl;
	std::cout << "----------------------------------------------" << std::endl;
	std::cout << "## Test :: Image" << std::endl;
	std::cout << "----------------------------------------------" << std::endl;
	std::cout << "----------------------------------------------" << std::endl;
	std::cout << std::endl;
	viennacl::image<CL_RGBA,CL_UNORM_INT8> image;
	// --------------------------------------------------------------------------
	std::cout << std::endl;
	std::cout << "----------------------------------------------" << std::endl;
	std::cout << std::endl;

	std::cout << std::endl;
	std::cout << "----------------------------------------------" << std::endl;
	std::cout << "----------------------------------------------" << std::endl;
	std::cout << "## Test :: Image 256x256" << std::endl;
	std::cout << "----------------------------------------------" << std::endl;
	std::cout << "----------------------------------------------" << std::endl;
	//cimg_library::CImg<unsigned char> img("/home/sanda/Desktop/CImg-1.5.0_beta/examples/img/lena.pgm");
	//img.save("/home/sanda/Desktop/CImg-1.5.0_beta/examples/img/lena2.pgm");
    streamsdk::SDKBitMap img;
	img.load("/home/sanda/Desktop/CImg-1.5.0_beta/examples/img/lena2.pgm");
	streamsdk::uchar4* d=  img.getPixels();
	cl_uchar4* f;

	viennacl::image<CL_RGBA,CL_UNORM_INT8> image2(8,8);
	viennacl::image<CL_RGBA,CL_UNORM_INT8> image3(8,8);
	viennacl::image<CL_RGBA,CL_UNORM_INT8> image4 = image2 + image3;
	std::cout << std::endl;
	std::cout << "----------------------------------------------" << std::endl;
	std::cout << std::endl;

	return retval;
}

int main()
{
	test();
}

#endif /* IMAGE_CPP_ */
