
#ifndef OCL_HPP
#define OCL_HPP


#include <CL/cl.hpp>

namespace hpcg_cl {
  void InitOpenCL(void);
  cl_command_queue getCommandQueue(void);
  cl_context getContext(void);
  cl_device_id getDeviceId(void);
  void ReleaseOpenCL(void);
}

#endif

