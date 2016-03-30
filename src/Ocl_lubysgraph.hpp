#ifndef OCLLUBYSGRAPH_HPP
#define OCLLUBYSGRAPH_HPP

#include "OCL.hpp"

namespace LubysGraphKernel {
  cl_program  program = 0;
  cl_kernel   kernel = 0;

  cl_mem  clRow_offset = NULL;
  cl_mem  clCol_index = NULL;
  cl_mem  clColors = NULL;
  cl_mem  clRandom = NULL;

  int allocSize = 0;
  int *colors = NULL;

  const char *kernel_name = "lubys_graph";

  void InitCLMem(int row_size, int *row_offset, int *col_index, int *random)
  {
    cl_int cl_status = CL_SUCCESS;
    if (NULL == clRow_offset)
    {
      clRow_offset = clCreateBuffer(hpcg_cl::getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    (row_size + 1) * sizeof(int), row_offset, &cl_status);
      if (CL_SUCCESS != cl_status || NULL == clRow_offset)
      {
        std::cout << "clRow_offset allocation failed. status: " << cl_status << std::endl;
        return;
      }
    }

    if (NULL == clCol_index)
    {
      clCol_index = clCreateBuffer(hpcg_cl::getContext(), CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR,
                                   (row_size * 27) * sizeof(int), col_index, &cl_status);
      if (CL_SUCCESS != cl_status || NULL == clCol_index)
      {
        std::cout << "clCol_index allocation failed. status: " << cl_status << std::endl;
        return;
      }
    }

    if (NULL == clColors)
    {
      clColors = clCreateBuffer(hpcg_cl::getContext(), CL_MEM_READ_WRITE, row_size * sizeof(int),
                                NULL, &cl_status);
      if (CL_SUCCESS != cl_status || NULL == clColors)
      {
        std::cout << "clColors allocation failed. status: " << cl_status << std::endl;
        return;
      }
    }

    if (NULL == clRandom)
    {
      clRandom = clCreateBuffer(hpcg_cl::getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                row_size * sizeof(int), random, &cl_status);
      if (CL_SUCCESS != cl_status || NULL == clRandom)
      {
        std::cout << "clRandom allocation failed. status: " << cl_status << std::endl;
        return;
      }
    }

    return;
  }

  void InitCpuMem(int size)
  {
    if (NULL == colors)
    {
      colors = (int *)malloc(size);
      allocSize = size;
    }

    return;
  }

  void ReleaseCLMem(void)
  {
    cl_int cl_status = CL_SUCCESS;

    if (NULL != clRow_offset)
    {
      cl_status |= clReleaseMemObject(clRow_offset);
      clRow_offset = NULL;
    }

    if (NULL != clCol_index)
    {
      cl_status |= clReleaseMemObject(clCol_index);
      clCol_index = NULL;
    }

    if (NULL != clColors)
    {
      cl_status |= clReleaseMemObject(clColors);
      clColors = NULL;
    }

    if (NULL != clRandom)
    {
      cl_status |= clReleaseMemObject(clRandom);
      clRandom = NULL;
    }

    if (CL_SUCCESS != cl_status)
    {
      std::cout << "clReleaseMemObject failed." <<std::endl;
    }

    return;
  }

  void ReleaseCpuMem(void)
  {
    if (NULL != colors)
    {
      free(colors);
      colors = NULL;
      allocSize = 0;
    }

    return;
  }

  void WriteBuffer(std::vector<local_int_t> &iColors, cl_mem clBuf)
  {
    cl_int cl_status = CL_SUCCESS;

    assert(allocSize == iColors.size() * sizeof(int));
    assert(NULL != colors);
    std::copy(iColors.begin(), iColors.end(), colors);

    cl_status = clEnqueueWriteBuffer(hpcg_cl::getCommandQueue(), clBuf, CL_TRUE, 0, allocSize,
                                     colors, 0, NULL, NULL);
    if (CL_SUCCESS != cl_status)
    {
      std::cout << "write buffer failed, status:" << cl_status << std::endl;
    }

    return;
  }

  void ReadBuffer(std::vector<local_int_t> &iColors, cl_mem clBuf)
  {
    cl_int cl_status = CL_SUCCESS;

    assert(allocSize == iColors.size() * sizeof(int));
    assert(NULL != colors);

    cl_status |= clEnqueueReadBuffer(hpcg_cl::getCommandQueue(), clBuf, CL_TRUE, 0,
                                     allocSize, colors, 0, NULL, NULL);
    if (CL_SUCCESS != cl_status)
    {
      std::cout << "read buffer failed, status:" << cl_status << std::endl;
    }

    std::copy(colors, colors + iColors.size(), iColors.begin());

    return;
  }

  void ExecuteKernel(int c, int row_size, cl_mem clMemColors)
  {
    cl_int cl_status = CL_SUCCESS;
    if (!program)
    {
      // get size of kernel source
      FILE *programHandle = fopen("..//..//src//kernel.cl", "r");
      if (NULL == programHandle) {
        std::cerr << "Can not open kernel file" << std::endl;
        return;
      }
      fseek(programHandle, 0, SEEK_END);
      size_t programSize = ftell(programHandle);
      rewind(programHandle);

      // read kernel source into buffer
      char *programBuffer  = (char *) malloc(programSize + 1);
      programBuffer[programSize] = '\0';
      fread(programBuffer, sizeof(char), programSize, programHandle);
      fclose(programHandle);
      program = clCreateProgramWithSource(hpcg_cl::getContext(), 1,
                                          (const char **)&programBuffer,
                                          &programSize, &cl_status);
      free(programBuffer);
      programBuffer = NULL;
      if (CL_SUCCESS != cl_status)
      {
        std::cout << "create program failed. status:" << cl_status << std::endl;
        return;
      }

      cl_device_id device = hpcg_cl::getDeviceId();
      cl_status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
      if (CL_SUCCESS != cl_status)
      {
        std::cout << "clBuild failed. status:" << cl_status << std::endl;
        char tbuf[0x10000];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0x10000, tbuf, NULL);
        std::cout << tbuf <<std::endl;
        return;
      }
    }

    if (!kernel)
    {
      kernel = clCreateKernel(program, kernel_name, &cl_status);
      if (CL_SUCCESS != cl_status)
      {
        std::cout << "create kernel failed. status:" << cl_status << std::endl;
        return;
      }
    }

    cl_int c1 = c;
    clSetKernelArg(kernel, 0, sizeof(cl_int), (void *)&c1);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&clRow_offset);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&clCol_index);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&clMemColors);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&clRandom);

    size_t global_size[] = {row_size};
    cl_status = clEnqueueNDRangeKernel(hpcg_cl::getCommandQueue(), kernel, 1, NULL,
                                       global_size, NULL, 0, NULL, NULL);
    if (CL_SUCCESS != cl_status)
    {
      std::cout << "NDRange failed. status:" << cl_status << std::endl;
      return;
    }

    clFinish(hpcg_cl::getCommandQueue());

    return;
  }
}

#endif

