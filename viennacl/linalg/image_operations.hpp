#ifndef _VIENNACL_IMAGE_OPERATIONS_HPP_
#define _VIENNACL_IMAGE_OPERATIONS_HPP_


#include "viennacl/forwards.h"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/handle.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/kernels/image_kernels.h"

 template<cl_channel_order CHANNEL_ORDER, cl_channel_type CHANNEL_TYPE>
 void inplace_sub(viennacl::image<CHANNEL_ORDER, CHANNEL_TYPE> & img1,
                     const viennacl::image<CHANNEL_ORDER, CHANNEL_TYPE> & img2)
    {
      assert(img1.size() == img2.size());
      unsigned int size = std::min(img1.internal_size(), img2.internal_size());
      viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::image<CHANNEL_ORDER, CHANNEL_TYPE>::program_name(), "inplace_sub");

      viennacl::ocl::enqueue(k(img1, img2, size));
    }

#endif
