
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

  cl_int  cl_status = CL_SUCCESS;
  if (!program) {
    // create program from buffer
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

  if (!kernel_lubys_graph) {
    kernel_lubys_graph = clCreateKernel(program, (const char *)"lubys_graph", &cl_status);
    if (CL_SUCCESS != cl_status) {
      std::cout << "SYMGSKernel failed. status:" << cl_status << std::endl;
      return;
    }
  }

  if (!kernel_SYMGS) {
    kernel_SYMGS = clCreateKernel(program, (const char *)"SYMGS", &cl_status);
    if (CL_SUCCESS != cl_status) {
      std::cout << "SYMGSKernel failed. status:" << cl_status << std::endl;
      return;
    }
  }

  if (!kernel_rtzCopy) {
    kernel_rtzCopy = clCreateKernel(program, (const char *)"rtzCopy", &cl_status);
    if (CL_SUCCESS != cl_status) {
      std::cout << "SYMGSKernel failed. status:" << cl_status << std::endl;
      return;
    }
  }

  if (!kernel_computeBeta) {
    kernel_computeBeta = clCreateKernel(program, (const char *)"computeBeta", &cl_status);
    if (CL_SUCCESS != cl_status) {
      std::cout << "SYMGSKernel failed. status:" << cl_status << std::endl;
      return;
    }
  }

  if (!kernel_computeAlpha) {
    kernel_computeAlpha = clCreateKernel(program, (const char *)"computeAlpha", &cl_status);
    if (CL_SUCCESS != cl_status) {
      std::cout << "SYMGSKernel failed. status:" << cl_status << std::endl;
      return;
    }
  }
}

int OCL::initBuffer(SparseMatrix &A, SparseMatrix &A_ref) {
   int cl_status = CL_SUCCESS;
#if 0
  local_int_t nrow = A_ref.localNumberOfRows;
  A_ref.mtxDiagonal = new double[nrow * 27];
  A_ref.mtxValue = new double[nrow * 27];
  A_ref.matrixIndL = new local_int_t[nrow * 27];
  for(int i = 0; i < nrow; ++i) {
    memcpy((void*)&(A_ref.mtxDiagonal[i * 27]), (void *)A_ref.matrixDiagonal[i], 27 * sizeof(double));
    memcpy((void*)&(A_ref.mtxValue[i * 27]), (void *)A_ref.matrixValues[i], 27 * sizeof(double));
    memcpy((void*)&(A_ref.matrixIndL[i * 27]), (void *)A_ref.mtxIndL[i], 27 * sizeof(local_int_t));
  }

   A_ref.clMatrixDiagonal = clCreateBuffer(context,
       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
       nrow * 27 * sizeof(double),
       A_ref.mtxDiagonal,
       &cl_status);
   if (CL_SUCCESS != cl_status || NULL == A_ref.clMatrixDiagonal) {
     std::cout << "create buffer failed. status:" << cl_status << std::endl;
     return -1;
   }

   A_ref.clMatrixValues = clCreateBuffer(context,
       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
       nrow * 27 * sizeof(double),
       A_ref.mtxValue,
       &cl_status);
   if (CL_SUCCESS != cl_status || NULL == A_ref.clMatrixValues) {
     std::cout << "create buffer failed. status:" << cl_status << std::endl;
     return -1;
   }

   A_ref.clNonzerosInRow = clCreateBuffer(context,
       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
       nrow * sizeof(char),
       A_ref.nonzerosInRow,
       &cl_status);
   if (CL_SUCCESS != cl_status || NULL == A_ref.clNonzerosInRow) {
     std::cout << "create buffer failed. status:" << cl_status << std::endl;
     return -1;
   }

   A_ref.clMtxIndL = clCreateBuffer(context,
       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
       nrow * 27 * sizeof(local_int_t),
       A_ref.matrixIndL,
       &cl_status);
   if (CL_SUCCESS != cl_status || NULL == A_ref.clMtxIndL) {
     std::cout << "create buffer failed. status:" << cl_status << std::endl;
     return -1;
   }
#else
  local_int_t nrow = A_ref.localNumberOfRows;
  double *mtxDiagonal = new double[nrow * 27];
  for(int i = 0; i < nrow; ++i) {
    memcpy((void*)&(mtxDiagonal[i * 27]), (void *)A_ref.matrixDiagonal[i], 27 * sizeof(double));
  }

  double *csrValue = new double[A_ref.totalNumberOfNonzeros];
  int *csrCol = new int[A_ref.totalNumberOfNonzeros];
  int *csrRowOff = new int[A_ref.localNumberOfRows + 1];

  int index = 0;
  csrRowOff[0] = 0;
  for (int i = 1; i <= A_ref.totalNumberOfRows; i++)
    csrRowOff[i] = csrRowOff[i - 1] + A_ref.nonzerosInRow[i - 1];

  for (int i = 0; i < A_ref.totalNumberOfRows; i++)
  {
     for(int j = 0; j < A_ref.nonzerosInRow[i]; j++)
     {
       csrValue[index] = A_ref.matrixValues[i][j];
       csrCol[index] = A_ref.mtxIndL[i][j];
       index++;
     }
  }

  A_ref.clCsrValues = clCreateBuffer(context,
       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
       A_ref.totalNumberOfNonzeros * sizeof(double),
       csrValue,
       &cl_status);
  if (CL_SUCCESS != cl_status || NULL == A_ref.clCsrValues) {
    std::cout << "create buffer failed. status:" << cl_status << std::endl;
    return -1;
  }

  A_ref.clCsrCol = clCreateBuffer(context,
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      A_ref.totalNumberOfNonzeros * sizeof(int),
      csrCol,
      &cl_status);
  if (CL_SUCCESS != cl_status || NULL == A_ref.clCsrCol) {
    std::cout << "create buffer failed. status:" << cl_status << std::endl;
    return -1;
  }

  A_ref.clCsrRowOff = clCreateBuffer(context,
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      (A_ref.localNumberOfRows + 1) * sizeof(int),
      csrRowOff,
      &cl_status);
  if (CL_SUCCESS != cl_status || NULL == A_ref.clCsrRowOff) {
    std::cout << "create buffer failed. status:" << cl_status << std::endl;
    return -1;
  }

  A_ref.clMatrixDiagonal = clCreateBuffer(context,
       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
       nrow * 27 * sizeof(double),
       mtxDiagonal,
       &cl_status);
  if (CL_SUCCESS != cl_status || NULL == A_ref.clMatrixDiagonal) {
    std::cout << "create buffer failed. status:" << cl_status << std::endl;
    return -1;
  }

  delete [] csrValue;
  delete [] csrCol;
  delete [] csrRowOff;
  delete [] mtxDiagonal;
#endif

  return 0;
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

cl_kernel OCL::getKernel_SYMGS() {
  return kernel_SYMGS;
}
cl_kernel OCL::getKernel_lubys_graph() {
  return kernel_lubys_graph;
}

cl_kernel OCL::getKernel_rtzCopy() {
  return kernel_rtzCopy;
}
cl_kernel OCL::getKernel_computeBeta() {
  return kernel_computeBeta;
}
cl_kernel OCL::getKernel_computeAlpha() {
  return kernel_computeAlpha;
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

  if(NULL != kernel_SYMGS) {
    clReleaseKernel(kernel_SYMGS);
    kernel_SYMGS = NULL;
  }

  if(NULL != kernel_lubys_graph) {
    clReleaseKernel(kernel_lubys_graph);
    kernel_lubys_graph = NULL;
  }

  if(NULL != kernel_rtzCopy) {
    clReleaseKernel(kernel_rtzCopy);
    kernel_rtzCopy = NULL;
  }

  if(NULL != kernel_computeBeta) {
    clReleaseKernel(kernel_computeBeta);
    kernel_computeBeta = NULL;
  }

  if(NULL != kernel_computeAlpha) {
    clReleaseKernel(kernel_computeAlpha);
    kernel_computeAlpha = NULL;
  }

  if(NULL != command_queue) {
    clReleaseCommandQueue(command_queue);
    command_queue = NULL;
  }

  if(NULL != program) {
    clReleaseProgram(program);
    program = NULL;
  }

}

} // namespace OCL

