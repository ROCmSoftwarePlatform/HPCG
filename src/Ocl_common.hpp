
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

const char *kernel_name = "forwardSYMGS";

void InitCLMem(int localNumberOfRows) {
  if (NULL == clXv) {
    clXv = clCreateBuffer(hpcg_cl::getContext(), CL_MEM_READ_WRITE,
                          localNumberOfRows * sizeof(double), NULL, &cl_status);
    if (CL_SUCCESS != cl_status || NULL == clXv) {
      std::cout << "clXv allocation failed. status: " << cl_status << std::endl;
      return;
    }
  }

  return;
}

cl_mem CreateCLBuf(cl_mem_flags flags, int size, void *pData) {
  cl_mem clBuf = clCreateBuffer(hpcg_cl::getContext(), flags, size, pData, &cl_status);
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
  cl_status = clEnqueueWriteBuffer(hpcg_cl::getCommandQueue(), clBuf, CL_TRUE, 0,
                                   size, pData, 0, NULL, NULL);
  if (CL_SUCCESS != cl_status) {
    std::cout << "write buffer failed, status:" << cl_status << std::endl;
  }

  return;
}

void ReadBuffer(cl_mem clBuf, void *pData, int size) {
  cl_status = clEnqueueReadBuffer(hpcg_cl::getCommandQueue(), clBuf, CL_TRUE, 0,
                                  size, pData, 0, NULL, NULL);
  if (CL_SUCCESS != cl_status) {
    std::cout << "SYMGSKernel Read buffer failed, status:" << cl_status << std::endl;
  }

  return;
}

void BuildProgram(void) {
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

  if (!program) {
    // create program from buffer
    program = clCreateProgramWithSource(hpcg_cl::getContext(), 1,
                                        (const char **) &programBuffer, &programSize, &cl_status);
    free(programBuffer);

    if (CL_SUCCESS != cl_status) {
      std::cout << "create program failed. status:" << cl_status << std::endl;
      return;
    }

    cl_device_id device = hpcg_cl::getDeviceId();
    cl_status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (CL_SUCCESS != cl_status) {
      std::cout << "clBuild failed. status:" << cl_status << std::endl;
      char tbuf[0x10000];
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0x10000, tbuf, NULL);
      std::cout << tbuf << std::endl;
      return;
    }
  }

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
  cl_status = clEnqueueNDRangeKernel(hpcg_cl::getCommandQueue(), kernel, 1, NULL,
                                     global_size, NULL, 0, NULL, NULL);
  if (CL_SUCCESS != cl_status) {
    std::cout << "NDRange failed. status:" << cl_status << std::endl;
    return;
  }

  clFinish(hpcg_cl::getCommandQueue());

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


