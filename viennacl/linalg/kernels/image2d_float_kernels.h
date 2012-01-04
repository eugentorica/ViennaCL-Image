#ifndef _VIENNACL_IMAGE_KERNELS_HPP_
#define _VIENNACL_IMAGE_KERNELS_HPP_
#include "viennacl/tools/tools.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"

#include "viennacl/linalg/kernels/image2d_float_source.h"

#include <CL/cl.hpp>
namespace viennacl {
namespace linalg {
namespace kernels {


struct image_float {

    static void init(const std::string& prog_name) {
        static std::map<cl_context, bool> init_done;
        viennacl::ocl::context & context_ = viennacl::ocl::current_context();
        if (!init_done[context_.handle()]) {
            std::string source;
            source.append(image2d_float_add);
            source.append(image2d_float_sub);
            source.append(image2d_float_convolute);
	    source.append(image2d_float_grayscale);


#ifdef VIENNACL_BUILD_INFO
            std::cout << "Creating program " << prog_name << std::endl;
#endif

            context_.add_program(source, prog_name);
            viennacl::ocl::program & prog_ = context_.get_program(prog_name);
            prog_.add_kernel("add");
            prog_.add_kernel("sub");
            prog_.add_kernel("convolute");
            prog_.add_kernel("grayscale");
            init_done[context_.handle()] = true;
        }

    }
};

template<cl_channel_order CHANNEL_ORDER,cl_channel_type CHANNEL_TYPE>
struct image2d;

template <>
struct image2d<CL_RGBA,CL_SNORM_INT8> {

    static std::string program_name()
    {
        return "CL_RGBA_CL_SNORM_INT8";
    }

    static void init()
    {
        std::string prog_name = program_name();
        image_float::init(prog_name);
    }

};

template <>
struct image2d<CL_RGBA,CL_SNORM_INT16> {

    static std::string program_name()
    {
        return "CL_RGBA_CL_SNORM_INT16";
    }

    static void init()
    {
        std::string prog_name = program_name();
        image_float::init(prog_name);
    }

};

template <>
struct image2d<CL_RGBA,CL_UNORM_INT8> {

    static std::string program_name()
    {
        return "CL_RGBA_CL_UNORM_INT8";
    }

    static void init()
    {
        std::string prog_name = program_name();
        image_float::init(prog_name);
    }

};

template <>
struct image2d<CL_RGBA,CL_UNORM_INT16> {

    static std::string program_name()
    {
        return "CL_RGBA_CL_UNORM_INT16";
    }

    static void init()
    {
        std::string prog_name = program_name();
        image_float::init(prog_name);
    }
};

template <>
struct image2d<CL_RGBA,CL_FLOAT> {

    static std::string program_name()
    {
        return "CL_RGBA_CL_FLOAT";
    }

    static void init()
    {
        std::string prog_name = program_name();
        image_float::init(prog_name);
    }
};

template <>
struct image2d<CL_RGBA,CL_HALF_FLOAT> {

    static std::string program_name()
    {
        return "CL_RGBA_CL_HALF_FLOAT";
    }

    static void init()
    {
        std::string prog_name = program_name();
        image_float::init(prog_name);
    }
};

template <>
struct image2d<CL_LUMINANCE,CL_UNORM_INT8> {

    static std::string program_name()
    {
        return "CL_LUMINANCE_CL_UNORM_INT8";
    }

    static void init()
    {
        std::string prog_name = program_name();
        image_float::init(prog_name);
    }
};

template <>
struct image2d<CL_RGB,CL_SNORM_INT8> {

    static std::string program_name()
    {
        return "CL_RGB_CL_SNORM_INT8";
    }

    static void init()
    {
        std::string prog_name = program_name();
        image_float::init(prog_name);
    }
};


} //namespace kernels
} //namespace linalg
} //namespace viennacl
#endif
