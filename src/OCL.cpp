
#include <iostream>
#include <cassert>
#include "OCL.hpp"

namespace HPCG_OCL {

OCL *OCL::self = NULL;

OCL *OCL::getOpenCL(void) {
  if (NULL == self) {
    self = new OCL();
  }
  return self;
}

OCL::OCL() {
  platform = NULL;
  device = NULL;
  context = 0;
  program = 0;
  command_queue = 0;
  create();
  BuildProgram();
}

OCL::~OCL() {
  ReleaseOpenCL();
  self = NULL;
  platform = NULL;
  device = NULL;
  context = 0;
  program = 0;
}

void OCL::create() {
  if (NULL != platform) {
    return;
  }

  cl_uint ret_num_of_platforms = 0;
  cl_int err = clGetPlatformIDs(0, NULL, &ret_num_of_platforms);
  assert(err == CL_SUCCESS && "clGetPlatformIDs\n");
  if (err != CL_SUCCESS || 0 == ret_num_of_platforms) {
    std::cout << "ERROR: Getting platforms!" << std::endl;
    return;
  }

  platform = (cl_platform_id *)malloc(ret_num_of_platforms * sizeof(cl_platform_id));
  err = clGetPlatformIDs(ret_num_of_platforms, platform, 0);
  assert(err == CL_SUCCESS && "clGetPlatformIDs\n");

  cl_uint ret_num_of_devices = 0;
  err = clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_GPU, 0, NULL, &ret_num_of_devices);

  assert(err == CL_SUCCESS && "clGetDeviceIds failed\n");

  device = (cl_device_id *)malloc(ret_num_of_devices * sizeof(cl_device_id));
  err = clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_GPU , ret_num_of_devices, device, 0);
  assert(err == CL_SUCCESS && "clGetDeviceIds failed\n");

  cl_context_properties property[] = {CL_CONTEXT_PLATFORM,
                                      (cl_context_properties)(platform[0]),
                                      0
                                     };

  context = clCreateContext(property, ret_num_of_devices, &device[0], NULL, NULL, &err);
  assert(err == CL_SUCCESS && "clCreateContext failed\n");

  command_queue = clCreateCommandQueue(context, device[0], 0, &err);
  assert(err == CL_SUCCESS && "clCreateCommandQueue failed \n");
  return;
}


void OCL::BuildProgram(void) {
  // get size of kernel source
  FILE *programHandle = fopen("..//..//src//kernel.cl", "r");
  if (NULL == programHandle) {
    std::cerr << "Can not open kernel file" << std::endl;
    return;
  }
  fseek(programHandle, 0, SEEK_END);
  size_t programSize = ftell(programHandle);
  rewind(programHandle);

  char *programBuffer  = (char *) malloc(programSize + 1);
  programBuffer[programSize] = '\0';
  fread(programBuffer, sizeof(char), programSize, programHandle);
  fclose(programHandle);

  if (!program) {
    // create program from buffer
    cl_int  cl_status = CL_SUCCESS;
    program = clCreateProgramWithSource(getContext(), 1,
                                        (const char **) &programBuffer, &programSize, &cl_status);
    free(programBuffer);

    if (CL_SUCCESS != cl_status) {
      std::cout << "create program failed. status:" << cl_status << std::endl;
      return;
    }

    cl_device_id device = getDeviceId();
    cl_status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (CL_SUCCESS != cl_status) {
      std::cout << "clBuild failed. status:" << cl_status << std::endl;
      char tbuf[0x10000];
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0x10000, tbuf, NULL);
      std::cout << tbuf << std::endl;
      return;
    }
  }
}

cl_context OCL::getContext(void) {
  return context;
}

cl_device_id OCL::getDeviceId(void) {
  return device[0];
}

cl_program OCL::getProgram(void) {
  return program;
}

cl_command_queue OCL::getCommandQueue(void) {
  return command_queue;
}

void OCL::ReleaseOpenCL(void) {
  if (0 != context) {
    clReleaseContext(context);
    context = 0;
  }

  if (NULL != device) {
    free(device);
    device = NULL;
  }

  if (NULL != platform) {
    free(platform);
    platform = NULL;
  }
}

} // namespace OCL
