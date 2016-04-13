
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
 @file ComputeSPMV.cpp

 HPCG routine
 */

#include "ComputeSPMV.hpp"
#include "ComputeSPMV_ref.hpp"
#include <iostream>
#include <sys/time.h>
#include <CL/cl.hpp>
#include "clSPARSE.h"
#include "clSPARSE-error.h"



/*!
Routine to compute sparse matrix vector product y = Ax where:
Precondition: First call exchange_externals to get off-processor values of x

This routine calls the reference SpMV implementation by default, but
can be replaced by a custom, optimized routine suited for
the target system.

@param[in]  A the known system matrix
@param[in]  x the known vector
@param[out] y the On exit contains the result: Ax.

@return returns 0 upon success and non-zero otherwise

@see ComputeSPMV_ref
*/

int ComputeSPMV(clsparseCsrMatrix &d_A, cldenseVector &d_x, cldenseVector &d_y,
                clsparseScalar &d_alpha, clsparseScalar &d_beta, clsparseCreateResult createResult) {
  clsparseStatus status = clsparseDcsrmv(&d_alpha, &d_A, &d_x, &d_beta, &d_y, createResult.control);
  if (status != clsparseSuccess) {
    std::cerr << "Problem with execution SpMV algorithm."
              << " Error: " << status << std::endl;
  }
  return 0;
}
