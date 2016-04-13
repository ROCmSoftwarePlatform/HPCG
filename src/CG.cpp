
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
 @file CG.cpp

 HPCG routine
 */

#include <fstream>

#include <cmath>
#include <iostream>

#include "hpcg.hpp"

#include "CG.hpp"
#include "mytimer.hpp"
#include "ComputeSPMV.hpp"
#include "ComputeMG.hpp"
#include "ComputeMG_ref.hpp"
#include "ComputeDotProduct.hpp"
#include "ComputeWAXPBY.hpp"
#include "OptimizeProblem.hpp"
#include "ComputeSPMV_ref.hpp"
#include "ComputeDotProduct_ref.hpp"
#include "ComputeWAXPBY_ref.hpp"
#include <iostream>

#include <CL/cl.hpp>
#include "clSPARSE.h"
#include "clSPARSE-error.h"
#include "OCL.hpp"

// Use TICK and TOCK to time a code section in MATLAB-like fashion
#define TICK()  t0 = mytimer() //!< record current time in 't0'
#define TOCK(t) t += mytimer() - t0 //!< store time difference in 't' using time in 't0'

/*!
  Routine to compute an approximate solution to Ax = b

  @param[in]    geom The description of the problem's geometry.
  @param[inout] A    The known system matrix
  @param[inout] data The data structure with all necessary CG vectors preallocated
  @param[in]    b    The known right hand side vector
  @param[inout] x    On entry: the initial guess; on exit: the new approximate solution
  @param[in]    max_iter  The maximum number of iterations to perform, even if tolerance is not met.
  @param[in]    tolerance The stopping criterion to assert convergence: if norm of residual is <= to tolerance.
  @param[out]   niters    The number of iterations actually performed.
  @param[out]   normr     The 2-norm of the residual vector after the last iteration.
  @param[out]   normr0    The 2-norm of the residual vector before the first iteration.
  @param[out]   times     The 7-element vector of the timing information accumulated during all of the iterations.
  @param[in]    doPreconditioning The flag to indicate whether the preconditioner should be invoked at each iteration.

  @return Returns zero on success and a non-zero value otherwise.

  @see CG_ref()
*/

extern double *val;
extern clsparseCsrMatrix d_A;
extern cldenseVector d_p, d_Ap, d_b, d_r, d_x;
extern clsparseScalar d_alpha, d_beta, d_normr, d_minus;
extern clsparseScalar d_rtz, d_oldrtz, d_Beta, d_Alpha, d_minusAlpha, d_pAp;
extern clsparseCreateResult createResult;


int CG(const SparseMatrix &A, SparseMatrix &A_ref, CGData &data, const Vector &b, Vector &x,
       const int max_iter, const double tolerance, int &niters, double &normr, double &normr0,
       double *times, bool doPreconditioning) {
  cl_int err;
  double t_begin = mytimer();  // Start timing right away
  normr = 0.0;
  double rtz = 0.0, oldrtz = 0.0, alpha = 0.0, beta = 0.0, pAp = 0.0, minusAlpha = 0.0;
  double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0, t5 = 0.0;
  size_t globalSize = 64;

  local_int_t nrow = A.localNumberOfRows;
  Vector &r = data.r;  // Residual vector
  Vector &z = data.z;  // Preconditioned residual vector
  Vector &p = data.p;  // Direction vector (in MPI mode ncol>=nrow)
  Vector &Ap = data.Ap;

  if (!doPreconditioning && A.geom->rank == 0) {
    HPCG_fout << "WARNING: PERFORMING UNPRECONDITIONED ITERATIONS" << std::endl;
  }

#ifdef HPCG_DEBUG
  int print_freq = 1;
  if (print_freq > 50) {
    print_freq = 50;
  }
  if (print_freq < 1) {
    print_freq = 1;
  }
#endif
  // p is of length ncols, copy x to p for sparse MV operation
  CopyVector(x, p);
  /* Create GPU buffers */
  int k = 0;
  for (int i = 0; i < A.totalNumberOfRows; i++) {
    for (int j = 0; j < A.nonzerosInRow[i]; j++) {
      val[k] = A.matrixValues[i][j];
      k++;
    }
  }

  clEnqueueWriteBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(), d_A.values, CL_TRUE, 0,
                       d_A.num_nonzeros * sizeof(double), val, 0, NULL, NULL);
  clEnqueueWriteBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(), d_p.values, CL_TRUE, 0,
                       d_A.num_rows * sizeof(double), p.values, 0, NULL, NULL);
  clEnqueueWriteBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(), d_b.values, CL_TRUE, 0,
                       d_A.num_rows * sizeof(double), b.values, 0, NULL, NULL);

  TICK(); ComputeSPMV(d_A, d_p, d_Ap, d_alpha, d_beta, createResult); TOCK(t3); // Ap = A*p
  TICK(); ComputeWAXPBY(d_alpha, d_b, d_minus, d_Ap, d_r, createResult);  TOCK(t2); // r = b - Ax (x stored in p)
  TICK(); ComputeDotProduct(d_r, d_r, d_normr, t4, createResult); TOCK(t1);


  clEnqueueReadBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(), d_r.values, CL_TRUE, 0,
                      d_r.num_values * sizeof(double), r.values, 0, NULL, NULL);
  clEnqueueReadBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(), d_normr.value, CL_TRUE, 0,
                      sizeof(double), &normr, 0, NULL, NULL);

  normr = sqrt(normr);

#ifdef HPCG_DEBUG
  if (A.geom->rank == 0) {
    HPCG_fout << "Initial Residual = " << normr << std::endl;
  }
#endif

  // Record initial residual for convergence testing
  normr0 = normr;

  // Start iterations

  cl_kernel kernelRtz = HPCG_OCL::OCL::getOpenCL()->getKernel(std::string("rtzCopy"));
  cl_kernel kernelBeta = HPCG_OCL::OCL::getOpenCL()->getKernel(std::string("computeBeta"));
  cl_kernel kernelAlpha = HPCG_OCL::OCL::getOpenCL()->getKernel(std::string("computeAlpha"));

  for (int k = 1; k <= max_iter && normr / normr0 > tolerance; k++) {
    TICK();
    if (doPreconditioning) {
      ComputeMG(A, A_ref, r, z);  // Apply preconditioner
    } else {
      CopyVector(r, z);  // copy r to z (no preconditioning)
    }
    TOCK(t5); // Preconditioner apply time

    clEnqueueWriteBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(), d_b.values, CL_TRUE, 0,
                         d_A.num_rows * sizeof(double), z.values, 0, NULL, NULL);

    if (k == 1) {
      TICK(); ComputeWAXPBY(d_alpha, d_b, d_beta, d_b, d_p, createResult); TOCK(t2); // Copy Mr to p
      TICK(); ComputeDotProduct(d_r, d_b, d_rtz, t4, createResult); TOCK(t1); // rtz = r'*z

    } else {
      // Execute the kernel over the entire range of the data set
      err = clEnqueueNDRangeKernel(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(),
                                   kernelRtz, 1, NULL, &globalSize, NULL,
                                   0, NULL, NULL);
      TICK(); ComputeDotProduct(d_r, d_b, d_rtz, t4, createResult); TOCK(t1); // rtz = r'*z
      // Execute the kernel over the entire range of the data set
      err = clEnqueueNDRangeKernel(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(),
                                   kernelBeta, 1, NULL, &globalSize, NULL,
                                   0, NULL, NULL);
      TICK(); ComputeWAXPBY(d_alpha, d_b, d_Beta, d_p, d_p, createResult);  TOCK(t2); // p = beta*p + z
    }

    TICK(); ComputeSPMV(d_A, d_p, d_Ap, d_alpha, d_beta, createResult); TOCK(t3); // Ap = A*p
    TICK(); ComputeDotProduct(d_p, d_Ap, d_pAp, t4, createResult); TOCK(t1); // alpha = p'*Ap

    //alpha = rtz/pAp;
    // Execute the kernel over the entire range of the data set
    err = clEnqueueNDRangeKernel(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(),
                                 kernelAlpha, 1, NULL, &globalSize, NULL,
                                 0, NULL, NULL);

    TICK();
    ComputeWAXPBY(d_alpha, d_x, d_Alpha, d_p, d_x, createResult);// x = x + alpha*p
    ComputeWAXPBY(d_alpha, d_r, d_minusAlpha, d_Ap, d_r, createResult);
    TOCK(t2);// r = r - alpha*Ap

    TICK(); ComputeDotProduct(d_r, d_r, d_normr, t4, createResult); TOCK(t1);

    clEnqueueReadBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(),
                        d_normr.value, CL_TRUE, 0,
                        sizeof(double), &normr, 0, NULL, NULL);

    normr = sqrt(normr);
#ifdef HPCG_DEBUG
    if (A.geom->rank == 0 && (k % print_freq == 0 || k == max_iter)) {
      HPCG_fout << "Iteration = " << k << "   Scaled Residual = " << normr / normr0 << std::endl;
    }
#endif
    niters = k;

    clEnqueueReadBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(),
                        d_r.values, CL_TRUE, 0,
                        d_r.num_values * sizeof(double), r.values,
                        0, NULL, NULL);
    clEnqueueReadBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(),
                        d_b.values, CL_TRUE, 0,
                        d_b.num_values * sizeof(double), z.values,
                        0, NULL, NULL);
  }

  clEnqueueReadBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(), d_p.values, CL_TRUE, 0,
                      d_p.num_values * sizeof(double), p.values, 0, NULL, NULL);
  clEnqueueReadBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(), d_Ap.values, CL_TRUE, 0,
                      d_Ap.num_values * sizeof(double), Ap.values, 0, NULL, NULL);

  clEnqueueReadBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(), d_x.values, CL_TRUE, 0,
                      d_x.num_values * sizeof(double), x.values, 0, NULL, NULL);


  // Store times
  times[1] += t1; // dot-product time
  times[2] += t2; // WAXPBY time
  times[3] += t3; // SPMV time
  times[4] += t4; // AllReduce time
  times[5] += t5; // preconditioner apply time
  //#ifndef HPCG_NO_MPI
  //  times[6] += t6; // exchange halo time
  //#endif
  times[0] += mytimer() - t_begin;  // Total time. All done...
  return 0;
}
