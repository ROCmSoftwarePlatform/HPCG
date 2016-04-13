
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
 @file ComputeWAXPBY.cpp

 HPCG routine
 */
#include <iostream>
#include "ComputeWAXPBY.hpp"
#include "ComputeWAXPBY_ref.hpp"
#include <CL/cl.hpp>
#include "clSPARSE.h"
#include <cmath>

/*!
  Routine to compute the update of a vector with the sum of two
  scaled vectors where: w = alpha*x + beta*y

  This routine calls the reference WAXPBY implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in] n the number of vector elements (on this processor)
  @param[in] alpha, beta the scalars applied to x and y respectively.
  @param[in] x, y the input vectors
  @param[out] w the output vector
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeWAXPBY_ref
*/

int ComputeWAXPBY(clsparseScalar alpha, cldenseVector &x, clsparseScalar beta,
                  cldenseVector &y, cldenseVector &w, clsparseCreateResult &createResult) {
  clsparseStatus status = cldenseDaxpby(&w, &alpha, &x, &beta, &y, createResult.control);
  if (status != clsparseSuccess) {
    std::cout << "Problem with execution of clsparse AXPBY algorithm" << std::endl;
  }
  return 0;
}
