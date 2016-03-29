
#include <iostream>
#include <cassert>
#include "OCL.hpp"

namespace hpcg_cl {
  cl_platform_id *platform = NULL;
  cl_context context = 0;
  cl_device_id *device = NULL;
  cl_command_queue command_queue = 0;

  void InitOpenCL(void)
  {
    if (NULL != platform)
    {
      return;
    }

    cl_uint ret_num_of_platforms = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &ret_num_of_platforms);
    assert(err == CL_SUCCESS && "clGetPlatformIDs\n");
    if (err != CL_SUCCESS || 0 == ret_num_of_platforms)
    {
      std::cout << "ERROR: Getting platforms!" << std::endl;
      return;
    }

    platform = (cl_platform_id*)malloc(ret_num_of_platforms * sizeof(cl_platform_id));
    err = clGetPlatformIDs(ret_num_of_platforms, platform, 0);
    assert(err == CL_SUCCESS && "clGetPlatformIDs\n");

    cl_uint ret_num_of_devices = 0;
    err = clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_GPU, 0, NULL, &ret_num_of_devices);

    assert(err == CL_SUCCESS && "clGetDeviceIds failed\n");

    device = (cl_device_id*)malloc(ret_num_of_devices * sizeof(cl_device_id));
    err = clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_GPU , ret_num_of_devices, device, 0);
    assert(err == CL_SUCCESS && "clGetDeviceIds failed\n");

    cl_context_properties property[] = {CL_CONTEXT_PLATFORM,
                                        (cl_context_properties)(platform[0]),
                                        0};

    context = clCreateContext(property, ret_num_of_devices, &device[0], NULL, NULL, &err);
    assert(err == CL_SUCCESS && "clCreateContext failed\n");

    command_queue = clCreateCommandQueue(context, device[0], 0, &err);
    assert(err == CL_SUCCESS && "clCreateCommandQueue failed \n");

    return;
  }

  cl_command_queue getCommandQueue(void)
  {
    return command_queue;
  }

  cl_context getContext(void)
  {
    return context;
  }

  cl_device_id getDeviceId(void)
  {
    return device[0];
  }

  void ReleaseOpenCL(void)
  {
    if (0 != command_queue)
    {
      clReleaseCommandQueue(command_queue);
      command_queue = 0;
    }

    if (0 != context)
    {
      clReleaseContext(context);
      context = 0;
    }

    if (NULL != device)
    {
      free(device);
      device = NULL;
    }

    if (NULL != platform)
    {
      free(platform);
      platform = NULL;
    }
  }
}

