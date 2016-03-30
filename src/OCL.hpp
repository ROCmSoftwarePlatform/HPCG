
#ifndef OCL_HPP
#define OCL_HPP


#include <CL/cl.hpp>

namespace HPCG_OCL {

class OCL {
public:
  static OCL *getOpenCL(void);
  cl_context getContext(void);
  cl_device_id getDeviceId(void);
  cl_program getProgram(void);
  cl_command_queue getCommandQueue(void);
private:
  OCL();
  ~OCL();
  void create();
  void ReleaseOpenCL();
  void BuildProgram();
  static OCL *self;
  cl_platform_id *platform;
  cl_device_id *device;
  cl_context context;
  cl_program program;
  cl_command_queue command_queue;
};

}

#endif

