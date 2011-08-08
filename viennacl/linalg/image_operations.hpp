#ifndef _VIENNACL_IMAGE_OPERATIONS_HPP_
#define _VIENNACL_IMAGE_OPERATIONS_HPP_


#include "viennacl/forwards.h"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/handle.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/tools/tools.hpp"

#include "viennacl/linalg/kernels/image_kernels.h"
#include "viennacl/image.hpp"

namespace viennacl
{
  namespace linalg
  {

 template<cl_channel_order CHANNEL_ORDER, cl_channel_type CHANNEL_TYPE>
 void add(const viennacl::image<CHANNEL_ORDER, CHANNEL_TYPE> & img1,const viennacl::image<CHANNEL_ORDER, CHANNEL_TYPE> & img2, viennacl::image<CHANNEL_ORDER, CHANNEL_TYPE> & img3)
 {
  //assert(img1.size() == img2.size());
   //unsigned int size = std::min(img1.internal_size(), img2.internal_size());

  //unsigned int size = 0;
   viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::image<CHANNEL_ORDER, CHANNEL_TYPE>::program_name(), "add");

   viennacl::ocl::enqueue(k(img1, img2, img3));
 }


 template<cl_channel_order CHANNEL_ORDER, cl_channel_type CHANNEL_TYPE>
 void sub(const viennacl::image<CHANNEL_ORDER, CHANNEL_TYPE> & img1,const viennacl::image<CHANNEL_ORDER, CHANNEL_TYPE> & img2, viennacl::image<CHANNEL_ORDER, CHANNEL_TYPE> & img3)
    {
	  //assert(img1.size() == img2.size());
      //unsigned int size = std::min(img1.internal_size(), img2.internal_size());

	  //unsigned int size = 0;
      viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::image<CHANNEL_ORDER, CHANNEL_TYPE>::program_name(), "sub");

      viennacl::ocl::enqueue(k(img1, img2, img3));
    }

  }
}

#endif
