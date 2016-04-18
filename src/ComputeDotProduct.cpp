
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

#include "OCL.hpp"

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
namespace DotKernel {
cl_mem  output_buffer = NULL;
}

int ComputeDotProduct_OCL(cldenseVector &x, cldenseVector &y,
                          clsparseScalar &r){
  size_t max_local_size, global_size;
  cl_uint num_groups;
  double *output_vec;

  int cl_status = CL_SUCCESS;
  clGetDeviceInfo(HPCG_OCL::OCL::getOpenCL()->getDeviceId(), CL_DEVICE_MAX_WORK_GROUP_SIZE,
                  sizeof(max_local_size), &max_local_size, NULL);
  num_groups = (x.num_values / 4) / max_local_size;
  output_vec = (double*) malloc(num_groups * sizeof(double));

  if (NULL == DotKernel::output_buffer) {
    DotKernel::output_buffer = clCreateBuffer(HPCG_OCL::OCL::getOpenCL()->getContext(), CL_MEM_READ_ONLY,
                          num_groups * sizeof(double), NULL, &cl_status);
    if (CL_SUCCESS != cl_status || NULL == DotKernel::output_buffer) {
      std::cout << "output_buffer allocation failed. status: " << cl_status << std::endl;
      return 0;
    }
  }

  cl_kernel kernel = HPCG_OCL::OCL::getOpenCL()->getKernel(std::string("dot_product"));

  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&x.values);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&y.values);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &DotKernel::output_buffer);
  clSetKernelArg(kernel, 3, max_local_size * 4 * sizeof(double), NULL);

  global_size = x.num_values / 4;

  cl_status = clEnqueueNDRangeKernel(
    HPCG_OCL::OCL::getOpenCL()->getCommandQueue(),
    kernel,
    1,
    NULL,
    &global_size,
    &max_local_size,
    0, NULL, NULL);
  if(cl_status < 0) {
    std::cout << "Couldn't enqueue the dot product kernel. status: " << cl_status << std::endl;
    return 0;
  }
  cl_kernel kernel_add = HPCG_OCL::OCL::getOpenCL()->getKernel(std::string("dot_add"));

  clSetKernelArg(kernel_add, 0, sizeof(cl_mem), &DotKernel::output_buffer);
  clSetKernelArg(kernel_add, 1, sizeof(cl_mem), &r.value);
  for(size_t t = num_groups; t > 1  ; t /= max_local_size) {
    size_t local_size_;
    if(t < max_local_size) {
      local_size_ = t;
    } else {
      local_size_ = max_local_size;
    }
    clSetKernelArg(kernel_add, 2, local_size_ * 4 * sizeof(double), NULL);
    global_size = t;
    cl_status = clEnqueueNDRangeKernel(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(),
                                       kernel_add,
                                       1,
                                       NULL,
                                       &global_size,
                                       &local_size_,
                                       0, NULL, NULL);
    if(cl_status < 0) {
      std::cout << "Couldn't enqueue the dot add kernel. status: " << cl_status << std::endl;
      return 0;
    }
  }
  free(output_vec);
  return 0;
}

int ComputeDotProduct(cldenseVector &x, cldenseVector &y,
                      clsparseScalar &r, double &time_allreduce, clsparseCreateResult createResult) {
 #ifdef __OCL__
  ComputeDotProduct_OCL(x, y, r);
#else
  clsparseStatus status = cldenseDdot(&r, &x, &y, createResult.control);
  if (status != clsparseSuccess) {
    std::cout << "Problem with execution of clsparse DOT algorithm"
              << " error: [" << status << "]" << std::endl;
  }
#endif
  return 0;
}
