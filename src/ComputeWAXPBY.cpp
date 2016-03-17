
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

clsparseScalar d_beta;

extern cl_context context;
extern cl_command_queue command_queue;
extern cl_int cl_status;
extern clsparseCreateResult createResult;
extern clsparseStatus status;
extern clsparseScalar alpha;
extern cldenseVector x;
extern cldenseVector y;

int ComputeWAXPBY(const local_int_t n, const double h_alpha, const Vector & h_x,
    const double h_beta, const Vector & h_y, Vector & h_w, bool & isOptimized) {

  // This line and the next two lines should be removed and your version of ComputeWAXPBY should be used.
  //isOptimized = false;
  //return ComputeWAXPBY_ref(n, h_alpha, h_x, h_beta, h_y, h_w);  
  
  static int count_axpby;
  if (!count_axpby)
  {
    clsparseInitScalar(&d_beta);
    d_beta.value =  ::clCreateBuffer (context, CL_MEM_READ_ONLY,
                          sizeof(double), NULL, &cl_status);                                                                   
    ++count_axpby;                                                                    
  }   
  
  clEnqueueWriteBuffer(command_queue, d_beta.value, CL_TRUE, 0,
                              sizeof( double ), &h_beta, 0, NULL, NULL );                                                             
  clEnqueueWriteBuffer(command_queue, x.values, CL_TRUE, 0,
                              n * sizeof( double ), h_x.values, 0, NULL, NULL );                              
  clEnqueueWriteBuffer(command_queue, y.values, CL_TRUE, 0,
                              n * sizeof( double ), h_y.values, 0, NULL, NULL );    
  
  status = cldenseDaxpby(&y, &alpha, &x, &d_beta, &y, createResult.control);

  if (status != clsparseSuccess)
  {
      std::cout << "Problem with execution of clsparse AXPBY algorithm"
                << " error: [" << status << "]" << std::endl;
  }
  
  clEnqueueReadBuffer(command_queue, y.values, CL_TRUE, 0,
                              n * sizeof(double), h_w.values, 0, NULL, NULL );  
                                                       
  return 0;
}
