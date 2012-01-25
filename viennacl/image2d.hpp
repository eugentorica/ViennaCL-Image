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

#include "viennacl/vector.hpp"

#include <math.h>

#include <iostream>

namespace viennacl {

template<cl_channel_order CHANNEL_ORDER, cl_channel_type CHANNEL_TYPE>
class image2d {
public:

    /** @brief */
    /*
    image2d() :
            _width(0), _height(0) {
        viennacl::linalg::kernels::image2d<CHANNEL_ORDER, CHANNEL_TYPE>::init();
    }

    explicit image2d(cl_mem existing_mem, int width, int height): _pixels(existing_mem), _width(width), _height(height)
    {
        _pixels.inc();
    }
    */

    /** @brief */
    image2d(unsigned int width, unsigned int height,void* ptr=NULL) : _width(width),_height(height) {
        viennacl::linalg::kernels::image2d<CHANNEL_ORDER, CHANNEL_TYPE>::init();
        cl_image_format image_format;
        image_format.image_channel_data_type = CHANNEL_TYPE;
        image_format.image_channel_order = CHANNEL_ORDER;
        
        _pixels = viennacl::ocl::current_context().create_image2d(CL_MEM_READ_WRITE, &image_format, width, height,ptr);
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
    image2d<CHANNEL_ORDER, CHANNEL_TYPE> gaussian_filter(unsigned int kernelSize, double sigma) const
    {
        image2d<CHANNEL_ORDER, CHANNEL_TYPE> result(_width, _height);
        std::vector<KERNEL_ELEMENT_TYPE> kernel = viennacl::tools::traits::getGaussianKernel<KERNEL_ELEMENT_TYPE>(kernelSize, sigma);
        float kernelTotalWeight = 0;

        vector<KERNEL_ELEMENT_TYPE> gpu_kernel(kernel.size());
        viennacl::fast_copy(kernel, gpu_kernel);
        for (unsigned int gpuIndex = 0 ;gpuIndex < kernel.size(); gpuIndex++)
            kernelTotalWeight += kernel[gpuIndex];

        viennacl::linalg::convolute(*this, result, gpu_kernel, kernelTotalWeight, kernelSize, kernelSize);
        return result;
    }

    template <typename KERNEL_ELEMENT_TYPE>
    image2d<CL_LUMINANCE, CHANNEL_TYPE> grayscale() const
    {
        std::vector<KERNEL_ELEMENT_TYPE> kernel = viennacl::tools::traits::getGrayScaleCoefficients<KERNEL_ELEMENT_TYPE>(CHANNEL_ORDER);

        viennacl::vector<KERNEL_ELEMENT_TYPE> gpu_kernel(kernel.size());
        viennacl::fast_copy(kernel, gpu_kernel);

        image2d<CL_LUMINANCE, CHANNEL_TYPE> result(_width, _height);
        viennacl::linalg::grayscale(*this, result, gpu_kernel);

        return result;
    }

    image2d<CHANNEL_ORDER, CHANNEL_TYPE> pyrUp() const
    {
        std::vector<float> kernel = viennacl::tools::traits::getPyramidKernel();
        float kernelTotalWeight = 0;
        unsigned int kernelSize = (unsigned int)sqrt(kernel.size());
        viennacl::vector<float> gpu_kernel(kernel.size());
        viennacl::fast_copy(kernel, gpu_kernel);

        for (unsigned int gpuIndex = 0 ;gpuIndex < kernel.size(); gpuIndex++)
            kernelTotalWeight += kernel[gpuIndex];

        image2d<CHANNEL_ORDER, CHANNEL_TYPE> result(floor(_width / 2), floor(_height / 2));
        viennacl::linalg::pyrUp(*this, result, gpu_kernel, kernelTotalWeight, kernelSize);

        return result;
    }

    image2d<CHANNEL_ORDER, CHANNEL_TYPE> pyrDown() const
    {
        std::vector<float> kernel = viennacl::tools::traits::getPyramidKernel();

        unsigned int kernelSize = (unsigned int)sqrt(kernel.size());
        viennacl::vector<float> gpu_kernel(kernel.size());
        viennacl::fast_copy(kernel, gpu_kernel);

        float kernelTotalWeight = 0;
        for (unsigned int gpuIndex = 0 ;gpuIndex < kernel.size(); gpuIndex++)
            kernelTotalWeight += kernel[gpuIndex];

        image2d<CHANNEL_ORDER, CHANNEL_TYPE> result( 2 * _width, 2 * _height);

        // Divide kernelTotalWeight by 4 as surface of the image increased 4 times
        // and this factor will keep image brightness the same as in original image
        viennacl::linalg::pyrDown(*this, result, gpu_kernel, kernelTotalWeight / 4 , kernelSize);
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
    /*
        image2d<CHANNEL_ORDER, CHANNEL_TYPE> clone() const
        {
            image2d<CHANNEL_ORDER, CHANNEL_TYPE> result(_width, _height,_pixels);
            return result;
        }
    */

    unsigned int width() const
    {
        return _width;
    }

    unsigned int height() const
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

