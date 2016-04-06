
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

   return 0;
}

int OCL::clsparse_initCsrMatrix(const SparseMatrix h_A, clsparseCsrMatrix &d_A, int *col, int *rowoff) {
  d_A.num_nonzeros = h_A.totalNumberOfNonzeros;
  d_A.num_rows = h_A.localNumberOfRows;
  d_A.num_cols = h_A.localNumberOfColumns;
  cl_int cl_status;
  d_A.values = clCreateBuffer(context, CL_MEM_READ_ONLY,
                              d_A.num_nonzeros * sizeof( double ), NULL, &cl_status );
  d_A.col_indices = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   d_A.num_nonzeros * sizeof(clsparseIdx_t), col, &cl_status );
  d_A.row_pointer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   (d_A.num_rows + 1) * sizeof(clsparseIdx_t), rowoff, &cl_status );
  return 0;
}

int OCL::clsparse_initDenseVector(cldenseVector &d_, int num_rows) {
  cl_int cl_status;
  d_.values = clCreateBuffer(context, CL_MEM_READ_WRITE, num_rows * sizeof(double),
                             NULL, &cl_status);
  d_.num_values = num_rows;
  return 0;
}

int OCL::clsparse_initScalar(clsparseScalar &d_, double val) {
  cl_int cl_status;
  d_.value = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double),
                            nullptr, &cl_status);
  double* t = (double*)clEnqueueMapBuffer(command_queue, d_.value, CL_TRUE, CL_MAP_WRITE,
                                          0, sizeof(double), 0, nullptr, nullptr, &cl_status);
  *t = val;
  cl_status = clEnqueueUnmapMemObject(command_queue, d_.value, t,
                                      0, nullptr, nullptr);
  return 0;
}

int OCL::clsparse_initScalar(clsparseScalar &d_) {
  cl_int cl_status;
  d_.value = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double), NULL, &cl_status);
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

