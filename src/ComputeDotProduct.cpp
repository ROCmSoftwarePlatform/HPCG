
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
 @file ComputeDotProduct.cpp

 HPCG routine
 */

#include "ComputeDotProduct.hpp"
#include "ComputeDotProduct_ref.hpp"
#include <iostream>
#include <CL/cl.hpp>
#include "clSPARSE.h"
#include <cmath>


/*!
  Routine to compute the dot product of two vectors.

  This routine calls the reference dot-product implementation by default, but
  can be replaced by a custom routine that is optimized and better suited for
  the target system.

  @param[in]  n the number of vector elements (on this processor)
  @param[in]  x, y the input vectors
  @param[out] result a pointer to scalar value, on exit will contain the result.
  @param[out] time_allreduce the time it took to perform the communication between processes
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeDotProduct_ref
*/

int ComputeDotProduct(cldenseVector &x, cldenseVector &y,
                      clsparseScalar &r, double &time_allreduce, clsparseCreateResult createResult) {
  clsparseStatus status = cldenseDdot(&r, &x, &y, createResult.control);
  if (status != clsparseSuccess) {
    std::cout << "Problem with execution of clsparse DOT algorithm"
              << " error: [" << status << "]" << std::endl;
  }
  return 0;
}
