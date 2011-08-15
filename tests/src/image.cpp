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
//
// -------------------------------------------------------------
//

//
// -------------------------------------------------------------
//
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

  const int size = 2;
  const int pixelSize = 4;
  int c = 0;
  unsigned char* srcImg2 = new unsigned char[pixelSize * size * size];
  unsigned char* srcImg3 = new unsigned char[pixelSize * size * size];

  for(int i=0; i< pixelSize* size * size;i++)
  {
    srcImg2[i] = 124;
    srcImg3[i] = 120;
  }

  std::vector<unsigned char> v(pixelSize * size * size);
  for(std::vector<unsigned char>::iterator iter=v.begin(); iter < v.end();++iter)
    *iter=0;


  viennacl::image<CL_RGBA, CL_UNORM_INT8> image2(size, size, (void*)srcImg2);
  viennacl::image<CL_RGBA, CL_UNORM_INT8> image3(size, size, (void*)srcImg3);

  (image2+image3).fast_copy_cpu(v.begin());

  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;

  std::cout << "Vector Init" << std::endl;
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

  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;

  (image2 - image3).fast_copy_cpu(v.begin());
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

  delete[] srcImg2;
  delete[] srcImg3;

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
