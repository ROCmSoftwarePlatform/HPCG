
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

extern cl_context context;
extern cl_command_queue command_queue;
extern cl_int cl_status;
extern clsparseCreateResult createResult;
extern clsparseStatus status;
extern cldenseVector x;
extern cldenseVector y;
extern clsparseScalar d_beta;

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
int ComputeDotProduct(const local_int_t n, const Vector & h_x, const Vector & h_y,
    double & result, double & time_allreduce, bool & isOptimized) {
    
  // This line and the next two lines should be removed and your version of ComputeDotProduct should be used.
  //isOptimized = false;
  //return ComputeDotProduct_ref(n, h_x, h_y, result, time_allreduce);
  
  clEnqueueWriteBuffer(command_queue, x.values, CL_TRUE, 0,
                              n * sizeof( double ), h_x.values, 0, NULL, NULL );                              
  clEnqueueWriteBuffer(command_queue, y.values, CL_TRUE, 0,
                              n * sizeof( double ), h_y.values, 0, NULL, NULL ); 
                       
  status = cldenseDdot(&d_beta, &x, &y, createResult.control);
  
  if (status != clsparseSuccess)
  {
      std::cout << "Problem with execution of clsparse DOT algorithm"
                << " error: [" << status << "]" << std::endl;
  }
  
  clEnqueueReadBuffer(command_queue, d_beta.value, CL_TRUE, 0,
                              sizeof(double), &result, 0, NULL, NULL );                         
                              
  return 0;                                                                       
}

int ComputeDotProduct_rr(double & result) {
  status = cldenseDdot(&d_beta, &y, &y, createResult.control);

  if (status != clsparseSuccess)
  {
      std::cout << "Problem with execution of clsparse DOT algorithm"
                << " error: [" << status << "]" << std::endl;
  }
  clEnqueueReadBuffer(command_queue, d_beta.value, CL_TRUE, 0,
                      sizeof(double), &result, 0, NULL, NULL );
  return 0;
}
