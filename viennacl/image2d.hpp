/*
 * image.hpp
 *
 *  Created on: Aug 1, 2011
 *      Author: Eugen
 *      Email: eugen.torica@gmail.com
 */

#ifndef _VIENNACL_IMAGE_HPP_
#define _VIENNACL_IMAGE_HPP_

#include "viennacl/forwards.h"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/linalg/image2d_operations.hpp"
#include "viennacl/tools/image_tools.hpp"

#include <iostream>

namespace viennacl {
/** @brief A proxy for scalar expressions (e.g. from inner vector products)
 *
 * assumption: dim(LHS) >= dim(RHS), where dim(scalar) = 0, dim(vector) = 1 and dim(matrix = 2)
 * @tparam LHS   The left hand side operand
 * @tparam RHS   The right hand side operand
 * @tparam OP    The operation tag
 */

/** @brief This class represents a single scalar value on the GPU and behaves mostly like a built-in scalar type like float or double.
 *
 * Since every read and write operation requires a CPU->GPU or GPU->CPU transfer, this type should be used with care.
 * The advantage of this type is that the GPU command queue can be filled without blocking read operations.
 *
 * @tparam TYPE  Either float or double. Checked at compile time.
 */
template<cl_channel_order CHANNEL_ORDER, cl_channel_type CHANNEL_TYPE>
class image2d {
public:

    /** @brief */
    /*
    image2d() :
            _width(0), _height(0) {
        viennacl::linalg::kernels::image2d<CHANNEL_ORDER, CHANNEL_TYPE>::init();
    }
    */

    /** @brief */
    explicit image2d(int width, int height,void* ptr=NULL) : _width(width),_height(height) {
        std::cout<<"RightBefore kernel init"<<std::endl;
        viennacl::linalg::kernels::image2d<CHANNEL_ORDER, CHANNEL_TYPE>::init();
        std::cout<<"RightAfter kernel init"<<std::endl;

        cl_image_format image_format;
        image_format.image_channel_data_type = CHANNEL_TYPE;
        image_format.image_channel_order = CHANNEL_ORDER;
        std::cout<<"RightBefore create image"<<std::endl;
        _pixels = viennacl::ocl::current_context().create_image2d(CL_MEM_READ_WRITE, &image_format, width, height,ptr);
    }

    explicit image2d(cl_mem existing_mem, int width, int height):_width(width),_height(height),_pixels(existing_mem)
    {
        _pixels.inc();
    }

    /** @brief Returns the OpenCL handle */
    const viennacl::ocl::handle<cl_mem> & handle() const {
        return _pixels;
    }

    image2d<CHANNEL_ORDER, CHANNEL_TYPE> operator + (const image2d<CHANNEL_ORDER, CHANNEL_TYPE> & other) const
    {
        image2d<CHANNEL_ORDER, CHANNEL_TYPE> result(_width,_height);
        viennacl::linalg::add(*this, other,result);
        return result;
    }

    image2d<CHANNEL_ORDER, CHANNEL_TYPE> operator - (const image2d<CHANNEL_ORDER, CHANNEL_TYPE> & other) const
    {
        image2d<CHANNEL_ORDER, CHANNEL_TYPE> result(_width,_height);
        viennacl::linalg::sub(*this, other,result);
        return result;
    }

    template <typename KERNEL_ELEMENT_TYPE>
    image2d<CHANNEL_ORDER, CHANNEL_TYPE> gausian_filter(unsigned int kernelSize, double sigma) const
    {
        float kernelTotalWeight;
        image2d<CHANNEL_ORDER, CHANNEL_TYPE> result(_width, _height);
        std::vector<KERNEL_ELEMENT_TYPE> kernel = viennacl::tools::traits::getGaussianKernel<KERNEL_ELEMENT_TYPE>(kernelSize, sigma);
        kernelTotalWeight = 1;

        // TODO: Find fast way to transfer CPU kernel to GPU one
        vector<KERNEL_ELEMENT_TYPE> gpu_kernel(kernel.size());
        for (unsigned int gpuIndex = 0 ;gpuIndex < kernel.size(); gpuIndex++)
        {
            gpu_kernel[gpuIndex] = kernel[gpuIndex];
        }

        viennacl::linalg::convolute(*this, result, gpu_kernel, kernelTotalWeight, kernelSize, kernelSize);
        return result;
    }

    template <typename KERNEL_ELEMENT_TYPE>
    image2d<CL_LUMINANCE, CHANNEL_TYPE> grayscale() const
    {
        std::cout<<std::endl<<"Grayscale"<<std::endl;
        std::vector<KERNEL_ELEMENT_TYPE> kernel = viennacl::tools::traits::getGrayScaleCoefficients<KERNEL_ELEMENT_TYPE>(CHANNEL_ORDER);

        // TODO: Find fast way to transfer CPU kernel to GPU one
        viennacl::vector<KERNEL_ELEMENT_TYPE> gpu_kernel(kernel.size());
        for (unsigned int gpuIndex = 0 ;gpuIndex < kernel.size(); gpuIndex++)
        {
            gpu_kernel[gpuIndex] = kernel[gpuIndex];
        }

        std::cout<<std::endl<<"Invoking grayscale"<<std::endl;
        image2d<CL_LUMINANCE, CHANNEL_TYPE> result(_width, _height);
        viennacl::linalg::grayscale(*this, result, gpu_kernel);

        return result;
    }

      //from gpu to cpu. Type assumption: cpu_vec lies in a linear memory chunk
    /** @brief STL-like transfer of a GPU vector to the CPU. The cpu type is assumed to reside in a linear piece of memory, such as e.g. for std::vector.
    *
    * This method is faster than the plain copy() function, because entries are
    * directly written to the cpu vector, starting with &(*cpu.begin()) However,
    * keep in mind that the cpu type MUST represent a linear piece of
    * memory, otherwise you will run into undefined behavior.
    *
    * @param gpu_begin  GPU iterator pointing to the beginning of the gpu vector (STL-like)
    * @param gpu_end    GPU iterator pointing to the end of the vector (STL-like)
    * @param cpu_begin  Output iterator for the cpu vector. The cpu vector must be at least as long as the gpu vector!
    */
    template <typename CPU_ITERATOR>
    void fast_copy_cpu(CPU_ITERATOR cpu_begin)
    {
        size_t origin[3]={0,0,0};
        size_t region[3]={_width,_height,1};

        cl_int err = clEnqueueReadImage(viennacl::ocl::get_queue().handle(),
                                        _pixels, CL_TRUE, origin,region,
                                        0,0,&(*cpu_begin),0,NULL,NULL);

        VIENNACL_ERR_CHECK(err);
        viennacl::ocl::get_queue().finish();
    }
    
    unsigned int width()
    {
      return _width;
    }
    
    unsigned int height()
    {
      return _height;
    }
    

private:
    
    viennacl::ocl::handle<cl_mem> _pixels;
    unsigned int _width;
    unsigned int _height;

};

} //namespace viennacl

#endif /* _VIENNACL_IMAGE_HPP_ */

