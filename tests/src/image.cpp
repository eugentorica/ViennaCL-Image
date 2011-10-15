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
#include "viennacl/image.hpp"
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

int test()
{
  int retval = EXIT_SUCCESS;
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;

  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Test :: Image 256x256" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;

  cimg_library::CImg<unsigned char> srcImg("milla.bmp");
  cimg_library::CImg<unsigned char> resultImg(srcImg.width(), srcImg.height(), srcImg.depth(), srcImg.spectrum());

  int imgWidth = srcImg.width();
  int imgHeight = srcImg.height();
  const int pixelSize = 3;
  int c = 0;
  std::vector<unsigned char> srcImg2(pixelSize * imgWidth * imgHeight);

  std::vector<unsigned char> srcImg3(pixelSize * imgWidth * imgHeight);

  srand ( time(NULL) );

  /* generate secret number: */

  for(int j=0; j< pixelSize; j++)
  {
    for(int k=0; k<imgHeight; k++)
      for(int i=0; i< imgWidth;i++)
      {
          //srcImg2[k*imgWidth*pixelSize +i*pixelSize + j] = srcImg(i,k,j);
          srcImg2[j * imgWidth * imgHeight + k*imgWidth + i] = srcImg(i,k,j);
          //srcImg2[i] = rand() % 255 + 1;
          srcImg3[i] = 120;
      }
  }

  viennacl::image<CL_RGBA, CL_UNORM_INT8> image2(imgWidth, imgHeight, &(*srcImg2.begin()));
  std::vector<unsigned char> v((pixelSize+1) * imgWidth * imgHeight);
  for(std::vector<unsigned char>::iterator iter=v.begin(); iter < v.end();++iter)
    *iter=0;

  //float myfloats[] = {1,2,1,2,4,2,1,2,1};
  float myfloats[] = {0,0,0,0,1,0,0,0,0};
  std::vector<float>kernel(myfloats, myfloats + sizeof(myfloats) / sizeof(float) );
  viennacl::image<CL_RGBA, CL_UNORM_INT8> gaussianFilteredImg  = image2.gausian_filter<std::vector<float>,float>(kernel);
  gaussianFilteredImg.fast_copy_cpu(v.begin());

  for(int j=0; j< pixelSize; j++)
  {
    for(int k=0; k<imgHeight; k++)
      for(int i=0; i< imgWidth;i++)
      {
        //resultImg(i,k,j)=v[k*imgWidth*pixelSize +i*pixelSize + j];
        resultImg(i,k,j)=v[j * imgWidth * imgHeight + k*imgWidth + i];
      }
  }

  resultImg.save("milla2.bmp");


/*
  std::cout << "Vector Result" << std::endl;
  c = 0;
  for(std::vector<unsigned char>::iterator iter=v.begin(); iter < v.end();++iter)
  {

     std::cout << ((int)*iter ) << " ";
     if (++c % 4 == 0)
     {
        std::cout<<std::endl;
        c = 0;
      }
   }
   std::cout<<" End Vector" << std::endl;
*/
  return retval;
}

int main()
{
  test();
}

#endif /* IMAGE_CPP_ */
