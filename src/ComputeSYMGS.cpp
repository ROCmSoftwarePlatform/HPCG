
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
#include "Ocl_common.hpp"
#include <vector>
using namespace std;

#define __OCL__
//#define __DEBUG__

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

static void ComputeSYMGS_OCL(const SparseMatrix &A, const Vector &r, Vector &x) {
  const local_int_t nrow = A.localNumberOfRows;
  double **matrixDiagonal = A.matrixDiagonal;   // An array of pointers to the diagonal entries A.matrixValues
  const double *const rv = r.values;
  double *const xv = x.values;

  // forward sweep to be carried out in parallel.
  local_int_t i = 0;
  int k;
  SYMGSKernel::InitCLMem(nrow);
  SYMGSKernel::WriteBuffer(SYMGSKernel::clXv, (void *)x.values, nrow * sizeof(double));

  SYMGSKernel::BuildProgram();
  for (k = 1; k < (int)(A.counters.size()); k++) {
    if (!(i < nrow && i < A.counters[k])) {
      continue;
    }
    int threadNum = std::min(nrow, A.counters[k]) - i;

    double *dlMatrixValues = new double[(threadNum) * 27];
    int  *iMtxIndL = new int[(threadNum) * 27];
    double *dlMatrixDiagonal = new double[(threadNum)];
    char *cNonzerosInRow = new char[(threadNum)];
    double *dlRv = new double[(threadNum)];
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
  }
  // backward sweep to be computed in parallel.
  i = nrow - 1;
  for (k = (int)(A.counters.size()); k > 0; k--) {
    if (!(i >= 0 && i >= A.counters[(k - 1)])) {
      continue;
    }
    int threadNum = i - std::max(0, A.counters[k-1]) + 1;

    int ii = i - threadNum + 1;

    double *dlMatrixValues = new double[(threadNum) * 27];
    int  *iMtxIndL = new int[(threadNum) * 27];
    double *dlMatrixDiagonal = new double[(threadNum)];
    char *cNonzerosInRow = new char[(threadNum)];
    double *dlRv = new double[(threadNum)];
    for (int index = 0; index < (threadNum); index++) {
      const double *const currentValues = A.matrixValues[ii + index];
      const local_int_t *const currentColIndices = A.mtxIndL[ii + index];
      dlMatrixDiagonal[index] = matrixDiagonal[ii + index][0];
      for (int m = 0; m < 27; m++) {
        dlMatrixValues[index * 27 + m] = currentValues[m];
        iMtxIndL[index * 27 + m] = currentColIndices[m];
      }
      cNonzerosInRow[index] = A.nonzerosInRow[ii + index];
      dlRv[index] = r.values[ii + index];
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

    SYMGSKernel::ExecuteKernel(threadNum, ii);

    SYMGSKernel::ReleaseCLBuf(&SYMGSKernel::clMatrixValues);
    SYMGSKernel::ReleaseCLBuf(&SYMGSKernel::clMtxIndL);
    SYMGSKernel::ReleaseCLBuf(&SYMGSKernel::clNonzerosInRow);
    SYMGSKernel::ReleaseCLBuf(&SYMGSKernel::clMatrixDiagonal);
    SYMGSKernel::ReleaseCLBuf(&SYMGSKernel::clRv);

    i -= (threadNum);

    delete [] dlMatrixDiagonal;
    delete [] dlMatrixValues;
    delete [] iMtxIndL;
    delete [] cNonzerosInRow;
    delete [] dlRv;
  }

  SYMGSKernel::ReadBuffer(SYMGSKernel::clXv, (void *)x.values,
                          nrow * sizeof(double));

#ifdef __DEBUG__
  for (int index = 0; index < nrow; index++) {
    std::cout << " " << xv[index];
  }
  std::cout << std::endl;
#endif

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
//        sum -= currentValues[j];
      }

      sum += xv[i] * currentDiagonal; // Remove diagonal contribution from previous loop
      xv[i] = sum / currentDiagonal;
    }
  }

#ifdef __DEBUG__
  for (int index = 0; index < nrow; index++) {
    std::cout << " " << xv[index];
  }
  std::cout << std::endl;
#endif

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
  ComputeSYMGS_OCL(A, r, x);
#else
  ComputeSYMGS_CPU(A, r, x);
#endif

  return 0;
}
