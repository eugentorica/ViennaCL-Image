#ifndef _VIENNACL_IMAGE_OPERATIONS_HPP_
#define _VIENNACL_IMAGE_OPERATIONS_HPP_

#include "viennacl/forwards.h"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/handle.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/tools/tools.hpp"

#include "viennacl/linalg/kernels/image2d_float_kernels.h"
#include "viennacl/image2d.hpp"

namespace viennacl
{
namespace linalg
{
const int global_group_size = 128;
const int local_group_size = 16;

template<cl_channel_order CHANNEL_ORDER, cl_channel_type CHANNEL_TYPE>
void add(const viennacl::image2d<CHANNEL_ORDER, CHANNEL_TYPE> & img1
         ,const viennacl::image2d<CHANNEL_ORDER, CHANNEL_TYPE> & img2
         , viennacl::image2d<CHANNEL_ORDER, CHANNEL_TYPE> & img3)
{
    viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(
                                    viennacl::linalg::kernels::image2d<CHANNEL_ORDER, CHANNEL_TYPE>::program_name(), "add");
    k.global_work_size(0, global_group_size);
    k.global_work_size(1, global_group_size);
    k.local_work_size(0, local_group_size);
    k.local_work_size(1, local_group_size);
    viennacl::ocl::enqueue(k(img1, img2, img3));
}

template<cl_channel_order CHANNEL_ORDER, cl_channel_type CHANNEL_TYPE>
void sub(const viennacl::image2d<CHANNEL_ORDER, CHANNEL_TYPE> & img1
         ,const viennacl::image2d<CHANNEL_ORDER, CHANNEL_TYPE> & img2
         , viennacl::image2d<CHANNEL_ORDER, CHANNEL_TYPE> & img3)
{
    viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(
                                    viennacl::linalg::kernels::image2d<CHANNEL_ORDER, CHANNEL_TYPE>::program_name(), "sub");
    k.global_work_size(0, global_group_size);
    k.global_work_size(1, global_group_size);
    k.local_work_size(0, local_group_size);
    k.local_work_size(1, local_group_size);
    viennacl::ocl::enqueue(k(img1, img2, img3));
}

template<cl_channel_order CHANNEL_ORDER, cl_channel_type CHANNEL_TYPE>
void convolute(const viennacl::image2d<CHANNEL_ORDER, CHANNEL_TYPE> & imgSrc,
               viennacl::image2d<CHANNEL_ORDER, CHANNEL_TYPE> & imgDst,
               const viennacl::vector<float> & kernel,float kernelTotalWeight,unsigned int kernelWidth, unsigned int kernelHeight)
{
    viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::image2d<CHANNEL_ORDER, CHANNEL_TYPE>::program_name()
                                , "convolute");
    k.global_work_size(0, global_group_size);
    k.global_work_size(1, global_group_size);
    k.local_work_size(0, local_group_size);
    k.local_work_size(1, local_group_size);
    viennacl::ocl::enqueue(k(imgSrc, imgDst, kernel, kernelTotalWeight, kernelWidth, kernelHeight));
}

template<cl_channel_order CHANNEL_ORDER, cl_channel_type CHANNEL_TYPE,typename KERNEL_TYPE>
void grayscale(const viennacl::image2d<CHANNEL_ORDER, CHANNEL_TYPE> & imgSrc,
               viennacl::image2d<CL_LUMINANCE, CHANNEL_TYPE> & imgDst,
               const viennacl::vector<KERNEL_TYPE> & kernel)
{
    viennacl::ocl::kernel & k = viennacl::ocl::get_kernel(viennacl::linalg::kernels::image2d<CHANNEL_ORDER, CHANNEL_TYPE>::program_name()
                                , "grayscale");
    k.global_work_size(0, global_group_size);
    k.global_work_size(1, global_group_size);
    k.local_work_size(0, local_group_size);
    k.local_work_size(1, local_group_size);
    viennacl::ocl::enqueue(k(imgSrc, imgDst, kernel));
}

}
}

#endif
