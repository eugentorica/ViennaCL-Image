#ifndef _VIENNACL_IMAGE_KERNELS_HPP_
#define _VIENNACL_IMAGE_KERNELS_HPP_
#include "viennacl/tools/tools.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"
#include "viennacl/linalg/kernels/image_source.h"
namespace viennacl {
namespace linalg {
namespace kernels {

template<cl_channel_order CHANNEL_ORDER,cl_channel_type CHANNEL_TYPE>
struct image;

template <>
struct image<CL_RGBA,CL_UNORM_INT8> {

	static std::string program_name()
	{
		return "CL_RGBA_CL_UNORM_INT8";
	}

	static void init() {
		static std::map<cl_context, bool> init_done;
		viennacl::ocl::context & context_ = viennacl::ocl::current_context();
		if (!init_done[context_.handle()]) {
			std::string source;
			source.append(image_add);
			source.append(image_sub);

			std::string prog_name = program_name();

			#ifdef VIENNACL_BUILD_INFO
			std::cout << "Creating program " << prog_name << std::endl;
        	#endif

			context_.add_program(source, prog_name);
			viennacl::ocl::program & prog_ = context_.get_program(prog_name);
			prog_.add_kernel("add");
			prog_.add_kernel("sub");
			init_done[context_.handle()] = true;
		}

	}
};

} //namespace kernels
} //namespace linalg
} //namespace viennacl
#endif
