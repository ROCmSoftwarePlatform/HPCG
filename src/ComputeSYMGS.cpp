
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeSYMGS.cpp

 HPCG routine
 */
#include <stdio.h>
#include <stdlib.h>
#include "iostream"
#include "ComputeSYMGS.hpp"
#include "OptimizeProblem.hpp"
#include "ComputeSYMGS_ref.hpp"
#include <vector>
using namespace std;

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

void ExecuteKernel(int size, int offset) {

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


/*!
  Routine to one step of symmetrix Gauss-Seidel:

  Assumption about the structure of matrix A:
  - Each row 'i' of the matrix has nonzero diagonal value whose address is matrixDiagonal[i]
  - Entries in row 'i' are ordered such that:
       - lower triangular terms are stored before the diagonal element.
       - upper triangular terms are stored after the diagonal element.
       - No other assumptions are made about entry ordering.

  Symmetric Gauss-Seidel notes:
  - We use the input vector x as the RHS and start with an initial guess for y of all zeros.
  - We perform one forward sweep.  Since y is initially zero we can ignore the upper triangular terms of A.
  - We then perform one back sweep.
       - For simplicity we include the diagonal contribution in the for-j loop, then correct the sum after

  @param[in]  A the known system matrix
  @param[in]  x the input vector
  @param[out] y On exit contains the result of one symmetric GS sweep with x as the RHS.

  @return returns 0 upon success and non-zero otherwise

  @warning Early versions of this kernel (Version 1.1 and earlier) had the r and x arguments in reverse order, and out of sync with other kernels.

  @see ComputeSYMGS_ref
*/

int ComputeSYMGS(const SparseMatrix &A, const Vector &r, Vector &x) {

  // This line and the next two lines should be removed and your version of ComputeSYMGS should be used.
  assert(x.localLength == A.localNumberOfColumns); // Make sure x contain space for halo values

#ifndef HPCG_NO_MPI
  ExchangeHalo(A, x);
#endif
  const local_int_t nrow = A.localNumberOfRows;
  double **matrixDiagonal = A.matrixDiagonal;   // An array of pointers to the diagonal entries A.matrixValues
  const double *const rv = r.values;
  double *const xv = x.values;

  // forward sweep to be carried out in parallel.
  local_int_t i = 0;
  int k;
#if 1
  SYMGSKernel::InitCLMem(nrow);
  SYMGSKernel::WriteBuffer(SYMGSKernel::clXv, (void *)x.values, nrow * sizeof(double));

  for (k = 1; k < (int)(A.counters.size() - 1); k++) {
    //int max = (nrow < (A.counters[k] + 1)) ? (i < nrow ? nrow : i) : (i < (A.counters[k] + 1) ? (A.counters[k] + 1) : i);
    if (!(i < nrow && i <= A.counters[k])) {
      continue;
    }
    int threadNum = std::min(nrow, A.counters[k] + 1) - i;

    double *dlMatrixValues = new double[(threadNum) * 27];
    int  *iMtxIndL = new int[(threadNum) * 27];
    double *dlMatrixDiagonal = new double[(threadNum)];
    char *cNonzerosInRow = new char[(threadNum)];
    double *dlRv = new double[(threadNum)];
    double *dlXv = new double[(threadNum)];
    for (int index = 0; index < (threadNum); index++) {
      const double *const currentValues = A.matrixValues[i + index];
      const local_int_t *const currentColIndices = A.mtxIndL[i + index];
      dlMatrixDiagonal[index] = matrixDiagonal[i + index][0];
      for (int m = 0; m < 27; m++) {
        dlMatrixValues[index * 27 + m] = currentValues[m];
        iMtxIndL[index * 27 + m] = currentColIndices[m];
      }
      cNonzerosInRow[index] = A.nonzerosInRow[i + index];
      dlRv[index] = r.values[i + index];
      dlXv[index] = x.values[i + index];
    }

    SYMGSKernel::clMatrixValues = SYMGSKernel::CreateCLBuf(
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    (threadNum) * 27 * sizeof(double),
                                    (void *)dlMatrixValues);

    SYMGSKernel::clMtxIndL = SYMGSKernel::CreateCLBuf(
                               CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               (threadNum) * 27 * sizeof(int),
                               (void *)iMtxIndL);

    SYMGSKernel::clNonzerosInRow = SYMGSKernel::CreateCLBuf(
                                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     (threadNum) * sizeof(char),
                                     (void *)cNonzerosInRow);

    SYMGSKernel::clMatrixDiagonal = SYMGSKernel::CreateCLBuf(
                                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      (threadNum) * sizeof(double), (void *)dlMatrixDiagonal);

    SYMGSKernel::clRv = SYMGSKernel::CreateCLBuf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        (threadNum) * sizeof(double),
                        (void *)dlRv);

    SYMGSKernel::ExecuteKernel(threadNum, i);

    SYMGSKernel::ReleaseCLBuf(&SYMGSKernel::clMatrixValues);
    SYMGSKernel::ReleaseCLBuf(&SYMGSKernel::clMtxIndL);
    SYMGSKernel::ReleaseCLBuf(&SYMGSKernel::clNonzerosInRow);
    SYMGSKernel::ReleaseCLBuf(&SYMGSKernel::clMatrixDiagonal);
    SYMGSKernel::ReleaseCLBuf(&SYMGSKernel::clRv);

    i += (threadNum);

    delete [] dlMatrixDiagonal;
    delete [] dlMatrixValues;
    delete [] iMtxIndL;
    delete [] cNonzerosInRow;
    delete [] dlRv;
    delete [] dlXv;
  }
  SYMGSKernel::ReadBuffer(SYMGSKernel::clXv, (void *)x.values,
                          nrow * sizeof(double));
#else

  for (k = 1; k < (int)(A.counters.size() - 1); k++) {
    for (; i < nrow && (i <= A.counters[k]); i++) {
      const double *const currentValues = A.matrixValues[i];
      const local_int_t *const currentColIndices = A.mtxIndL[i];
      const int currentNumberOfNonzeros = A.nonzerosInRow[i];
      const double  currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
      double sum = rv[i]; // RHS value

      for (int j = 0; j < currentNumberOfNonzeros; j++) {
        local_int_t curCol = currentColIndices[j];
        sum -= currentValues[j] * xv[curCol];
      }

      sum += xv[i] * currentDiagonal; // Remove diagonal contribution from previous loop
      xv[i] = sum / currentDiagonal;
    }
  }
#endif

 /* 
    for (int index = 0; index < nrow; index++)
    {
      std::cout << " " << xv[index];
    }
    std::cout << std::endl;*/

  // backward sweep to be computed in parallel.
  i = nrow - 1;
  for (k = (int)(A.counters.size() - 1); k > 0; k--) {
    for (; i >= 0 && (i >= A.counters[(k - 1)]); i--) {
      const double *const currentValues = A.matrixValues[i];
      const local_int_t *const currentColIndices = A.mtxIndL[i];
      const int currentNumberOfNonzeros = A.nonzerosInRow[i];
      const double  currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
      double sum = rv[i]; // RHS value

      for (int j = 0; j < currentNumberOfNonzeros; j++) {
        local_int_t curCol = currentColIndices[j];
        sum -= currentValues[j] * xv[curCol];
      }
      sum += xv[i] * currentDiagonal; // Remove diagonal contribution from previous loop

      xv[i] = sum / currentDiagonal;
    }
  }
  return 0;
}
