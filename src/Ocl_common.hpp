
#ifndef OCLCOMMON_HPP
#define OCLCOMMON_HPP

#include "OCL.hpp"
namespace SYMGSKernel {
cl_mem  clMatrixValues = NULL;
cl_mem  clMtxIndL = NULL;
cl_mem  clNonzerosInRow = NULL;
cl_mem  clMatrixDiagonal = NULL;
cl_mem  clRv = NULL;
cl_mem  clXv = NULL;
cl_int  cl_status = CL_SUCCESS;

cl_program program = NULL;
cl_kernel  kernel = NULL;

const char *kernel_name = "SYMGS";

void InitCLMem(int localNumberOfRows) {
  if (NULL == clXv) {
    clXv = clCreateBuffer(HPCG_OCL::OCL::getOpenCL()->getContext(), CL_MEM_READ_WRITE,
                          localNumberOfRows * sizeof(double), NULL, &cl_status);
    if (CL_SUCCESS != cl_status || NULL == clXv) {
      std::cout << "clXv allocation failed. status: " << cl_status << std::endl;
      return;
    }
  }

  return;
}

cl_mem CreateCLBuf(cl_mem_flags flags, int size, void *pData) {
  cl_mem clBuf = clCreateBuffer(HPCG_OCL::OCL::getOpenCL()->getContext(), flags, size, pData, &cl_status);
  if (CL_SUCCESS != cl_status || NULL == clBuf) {
    std::cout << "CreateCLBuf failed. status: " << cl_status
              << std::endl;
    return NULL;
  }

  return clBuf;
}

void ReleaseCLBuf(cl_mem *clBuf) {
  clReleaseMemObject(*clBuf);
  *clBuf = NULL;
  return;
}

void WriteBuffer(cl_mem clBuf, void *pData, int size) {
  cl_status = clEnqueueWriteBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(), clBuf, CL_TRUE, 0,
                                   size, pData, 0, NULL, NULL);
  if (CL_SUCCESS != cl_status) {
    std::cout << "write buffer failed, status:" << cl_status << std::endl;
  }

  return;
}

void ReadBuffer(cl_mem clBuf, void *pData, int size) {
  cl_status = clEnqueueReadBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(), clBuf, CL_TRUE, 0,
                                  size, pData, 0, NULL, NULL);
  if (CL_SUCCESS != cl_status) {
    std::cout << "SYMGSKernel Read buffer failed, status:" << cl_status << std::endl;
  }

  return;
}

void BuildProgram(void) {
  program = HPCG_OCL::OCL::getOpenCL()->getProgram();
  if (!kernel) {
    kernel = clCreateKernel(program, kernel_name, &cl_status);
    if (CL_SUCCESS != cl_status) {
      std::cout << "SYMGSKernel failed. status:" << cl_status << std::endl;
      return;
    }
  }
}

void ExecuteKernel(int size, int offset) {

  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&clMatrixValues);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&clMtxIndL);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&clNonzerosInRow);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&clMatrixDiagonal);
  clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&clRv);
  clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&clXv);
  clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&offset);

  size_t global_size[] = {size};
  cl_status = clEnqueueNDRangeKernel(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(), kernel, 1, NULL,
                                     global_size, NULL, 0, NULL, NULL);
  if (CL_SUCCESS != cl_status) {
    std::cout << "NDRange failed. status:" << cl_status << std::endl;
    return;
  }

  clFinish(HPCG_OCL::OCL::getOpenCL()->getCommandQueue());

  return;
}

void ReleaseProgram(cl_program *program) {
  clReleaseProgram(*program);
  *program = NULL;
  return;
}

void ReleaseKernel(cl_kernel *kernel) {
  clReleaseKernel(*kernel);
  *kernel = NULL;
  return;
}
}
#endif

