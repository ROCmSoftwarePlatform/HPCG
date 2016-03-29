
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


 #ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif
#include "ComputeSYMGS_ref.hpp"
#include "ComputeSYMGS.hpp"
#include <cassert>

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
int ComputeSYMGS( const SparseMatrix & A, const Vector & r, Vector & x) {

  
  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

#ifndef HPCG_NO_MPI
  ExchangeHalo(A,x);
#endif

  const local_int_t nrow = A.localNumberOfRows;
  double ** matrixDiagonal = A.matrixDiagonal;  // An array of pointers to the diagonal entries A.matrixValues
  const double * const rv = r.values;
  double * const xv = x.values;

  // implemented level scheduling algorithm for forward sweep
  int level = 0;
  for(level = 0; level <= A.level_no; level++)
  {
    #ifndef HPCG_NO_OPENMP
      #pragma omp parallel for
    #endif
    for (local_int_t i=0; i< nrow; i++) {
      if(A.level_array[i] == level)
      {
          const double * const currentValues = A.matrixValues[i];
          const local_int_t * const currentColIndices = A.mtxIndL[i];
          const int currentNumberOfNonzeros = A.nonzerosInRow[i];
          const double  currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
          double sum = rv[i]; // RHS value

          for (int j=0; j< currentNumberOfNonzeros; j++) {
            local_int_t curCol = currentColIndices[j];
            sum -= currentValues[j] * xv[curCol];
          }
          sum += xv[i]*currentDiagonal; // Remove diagonal contribution from previous loop

          xv[i] = sum/currentDiagonal;
      }
    }
  }

  // implemented level scheduling algorithm for backward sweep.
  for(level = A.level_no; level >= 0; level--)
  {
    #ifndef HPCG_NO_OPENMP
      #pragma omp parallel for
    #endif
    for (local_int_t i=nrow-1; i>=0; i--) {
      if(A.level_array[i] == level)
      {
        const double * const currentValues = A.matrixValues[i];
        const local_int_t * const currentColIndices = A.mtxIndL[i];
        const int currentNumberOfNonzeros = A.nonzerosInRow[i];
        const double  currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
        double sum = rv[i]; // RHS value

        for (int j = 0; j< currentNumberOfNonzeros; j++) {
          local_int_t curCol = currentColIndices[j];
          sum -= currentValues[j]*xv[curCol];
        }
        sum += xv[i]*currentDiagonal; // Remove diagonal contribution from previous loop

        xv[i] = sum/currentDiagonal;
      }
    }
  }

  return 0;
}
