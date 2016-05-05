
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
#include "OCL.hpp"
#include <vector>
using namespace std;

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

static void ComputeSYMGS_OCL(SparseMatrix &A, const Vector &r, Vector &x) {
  const local_int_t nrow = A.localNumberOfRows;
  local_int_t i = 0;
  int k;

  int cl_status = CL_SUCCESS;
  if (NULL == A.clXv) {
    A.clXv = clCreateBuffer(HPCG_OCL::OCL::getOpenCL()->getContext(), CL_MEM_READ_WRITE,
                          nrow * sizeof(double), NULL, &cl_status);
    if (CL_SUCCESS != cl_status || NULL == A.clXv) {
      std::cout << "A.clXv allocation failed. status: " << cl_status << std::endl;
      return;
    }
  }
  if (NULL == A.clRv) {
    A.clRv = clCreateBuffer(HPCG_OCL::OCL::getOpenCL()->getContext(), CL_MEM_READ_ONLY,
                          nrow * sizeof(double), NULL, &cl_status);
    if (CL_SUCCESS != cl_status || NULL == A.clRv) {
      std::cout << "A.clXv allocation failed. status: " << cl_status << std::endl;
      return;
    }
  }

  clEnqueueWriteBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(),
                       A.clXv,
                       CL_TRUE,
                       0,
                       nrow * sizeof(double),
                       (void *)x.values,
                       0, NULL, NULL);

  clEnqueueWriteBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(),
                       A.clRv,
                       CL_TRUE,
                       0,
                       nrow * sizeof(double),
                       (void *)r.values,
                       0, NULL, NULL);

  cl_kernel kernel = HPCG_OCL::OCL::getOpenCL()->getKernel(std::string("SYMGS"));
  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&A.clMatrixValues);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&A.clMtxIndL);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&A.clNonzerosInRow);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&A.clMatrixDiagonal);
  clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&A.clRv);
  clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&A.clXv);
  clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&A.clColors);

  // forward sweep to be carried out in parallel.
  for (k = 1; k < (int)(A.counters.size()); k++) {
    if (!(i < nrow && i < A.counters[k])) {
      continue;
    }
    int threadNum = std::min(nrow, A.counters[k]) - i;

    clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&i);
    size_t global_size[] = {threadNum * 32};
    size_t local_size[] = {32};
    clEnqueueNDRangeKernel(
        HPCG_OCL::OCL::getOpenCL()->getCommandQueue(),
        kernel,
        1,
        NULL,
        global_size,
        local_size,
        0, NULL, NULL);

    i += (threadNum);
  }

  // backward sweep to be computed in parallel.
  i = nrow - 1;
  for (k = (int)(A.counters.size()); k > 0; k--) {
    if (!(i >= 0 && i >= A.counters[(k - 1)])) {
      continue;
    }
    int threadNum = i - std::max(0, A.counters[k - 1]) + 1;
    int ii = i - threadNum + 1;

    clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&ii);
    size_t global_size[] = {threadNum * 32};
    size_t local_size[] = {32};
    clEnqueueNDRangeKernel(
        HPCG_OCL::OCL::getOpenCL()->getCommandQueue(),
        kernel,
        1,
        NULL,
        global_size,
        local_size,
        0, NULL, NULL);

    i -= (threadNum);
  }

  clEnqueueReadBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(),
                      A.clXv,
                      CL_TRUE,
                      0,
                      nrow * sizeof(double),
                      (void *)x.values,
                      0, NULL, NULL);
}

static void ComputeSYMGS_CPU(const SparseMatrix &A, const Vector &r, Vector &x) {
  const local_int_t nrow = A.localNumberOfRows;
  double **matrixDiagonal = A.matrixDiagonal;   // An array of pointers to the diagonal entries A.matrixValues
  const double *const rv = r.values;
  double *const xv = x.values;

  // forward sweep to be carried out in parallel.
  local_int_t i = 0;
  int k;
  for (k = 1; k < (int)(A.counters.size()); k++) {
    for (; i < nrow && (i < A.counters[k]); i++) {
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

  // backward sweep to be computed in parallel.
  i = nrow - 1;
  for (k = (int)(A.counters.size()); k > 0; k--) {
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
}

int ComputeSYMGS(const SparseMatrix &A, const Vector &r, Vector &x) {
  // This line and the next two lines should be removed and your version of ComputeSYMGS should be used.
  assert(x.localLength == A.localNumberOfColumns); // Make sure x contain space for halo values
#ifndef HPCG_NO_MPI
  ExchangeHalo(A, x);
#endif

#ifdef __OCL__
  ComputeSYMGS_OCL((SparseMatrix &)A, r, x);
#else
  ComputeSYMGS_CPU(A, r, x);
#endif
  return 0;
}
