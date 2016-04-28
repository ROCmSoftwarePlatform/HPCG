
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
 @file TestSymmetry.cpp

 HPCG routine
 */

// The MPI include must be first for Windows platforms
#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif
#include <fstream>
#include <iostream>
#include <cfloat>
using std::endl;
#include <vector>
#include <cmath>

#include "hpcg.hpp"

#include "ComputeSPMV.hpp"
#include "ComputeMG.hpp"
#include "ComputeDotProduct.hpp"
#include "ComputeResidual.hpp"
#include "Geometry.hpp"
#include "SparseMatrix.hpp"
#include "TestSymmetry.hpp"

#include "ComputeSPMV_ref.hpp"
#include "ComputeDotProduct_ref.hpp"

#include <CL/cl.hpp>
#include "clSPARSE.h"
#include "OCL.hpp"

extern clsparseScalar d_Beta, d_Alpha;

/*!
  Tests symmetry-preserving properties of the sparse matrix vector multiply and
  symmetric Gauss-Siedel routines.

  @param[in]    geom   The description of the problem's geometry.
  @param[in]    A      The known system matrix
  @param[in]    b      The known right hand side vector
  @param[in]    xexact The exact solution vector
  @param[inout] testsymmetry_data The data structure with the results of the CG symmetry test including pass/fail information

  @return returns 0 upon success and non-zero otherwise

  @see ComputeDotProduct
  @see ComputeDotProduct_ref
  @see ComputeSPMV
  @see ComputeSPMV_ref
  @see ComputeMG
  @see ComputeMG_ref
*/

int TestSymmetry(SparseMatrix &A, Vector &b, Vector &xexact, TestSymmetryData &testsymmetry_data) {
  SparseMatrix &A_ref = *(SparseMatrix *)A.optimizationData;
  local_int_t nrow = A.localNumberOfRows;
  local_int_t ncol = A.localNumberOfColumns;

  Vector x_ncol, y_ncol, z_ncol;
  InitializeVector(x_ncol, ncol);
  InitializeVector(y_ncol, ncol);
  InitializeVector(z_ncol, ncol);

  double t4 = 0.0; // Needed for dot-product call, otherwise unused
  testsymmetry_data.count_fail = 0;

  // Test symmetry of matrix

  // First load vectors with random values
  FillRandomVector(x_ncol);
  FillRandomVector(y_ncol);

  double xNorm2, yNorm2;
  double ANorm = 2 * 26.0;

  int k = 0;
  for (int i = 0; i < A.totalNumberOfRows; i++) {
    for (int j = 0; j < A.nonzerosInRow[i]; j++) {
      A.val[k] = A.matrixValues[i][j];
      k++;
    }
  }

  clEnqueueWriteBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(), A.d_A.values, CL_TRUE, 0,
                       A.d_A.num_nonzeros * sizeof(double), A.val, 0, NULL, NULL);

  // Next, compute x'*A*y
  ComputeDotProduct(A.d_p, A.d_p, A.d_rtz, t4, A.createResult);
  int ierr = ComputeSPMV(A.d_A, A.d_p, A.d_Ap, A.d_alpha, A.d_beta, A.createResult); // z_nrow = A*y_overlap
  if (ierr) {
    HPCG_fout << "Error in call to SpMV: " << ierr << ".\n" << endl;
  }
  double xtAy = 0.0;
  ierr = ComputeDotProduct(A.d_b, A.d_Ap, A.d_oldrtz, t4, A.createResult); // x'*A*y
  if (ierr) {
    HPCG_fout << "Error in call to dot: " << ierr << ".\n" << endl;
  }

  // Next, compute y'*A*x
  ComputeDotProduct(A.d_b, A.d_b, d_Beta, t4, A.createResult);
  ierr = ComputeSPMV(A.d_A, A.d_b, A.d_Ap, A.d_alpha, A.d_beta, A.createResult); // b_computed = A*x_overlap
  if (ierr) {
    HPCG_fout << "Error in call to SpMV: " << ierr << ".\n" << endl;
  }
  double ytAx = 0.0;
  ierr = ComputeDotProduct(A.d_p, A.d_Ap, d_Alpha, t4, A.createResult); // y'*A*x
  if (ierr) {
    HPCG_fout << "Error in call to dot: " << ierr << ".\n" << endl;
  }

  clEnqueueReadBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(), A.d_p.values, CL_TRUE, 0,
                      A.d_p.num_values * sizeof(double), y_ncol.values, 0, NULL, NULL);
  clEnqueueReadBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(), A.d_Ap.values, CL_TRUE, 0,
                      A.d_Ap.num_values * sizeof(double), z_ncol.values, 0, NULL, NULL);
  clEnqueueReadBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(), d_Beta.value, CL_TRUE, 0,
                      sizeof(double), &xNorm2, 0, NULL, NULL);
  clEnqueueReadBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(), A.d_rtz.value, CL_TRUE, 0,
                      sizeof(double), &yNorm2, 0, NULL, NULL);
  clEnqueueReadBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(), A.d_oldrtz.value, CL_TRUE, 0,
                      sizeof(double), &xtAy, 0, NULL, NULL);
  clEnqueueReadBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(), d_Alpha.value, CL_TRUE, 0,
                      sizeof(double), &ytAx, 0, NULL, NULL);

  testsymmetry_data.depsym_spmv = std::fabs((long double)(xtAy - ytAx)) / ((xNorm2 * ANorm * yNorm2 + yNorm2 * ANorm * xNorm2) * (DBL_EPSILON));
  if (testsymmetry_data.depsym_spmv > 1.0) {
    ++testsymmetry_data.count_fail;  // If the difference is > 1, count it wrong
  }
  if (A.geom->rank == 0) {
    HPCG_fout << "Departure from symmetry (scaled) for SpMV abs(x'*A*y - y'*A*x) = " << testsymmetry_data.depsym_spmv << endl;
  }

  // Test symmetry of symmetric Gauss-Seidel

 // Compute x'*Minv*y
 // Create r_copy and z_copy to copy y_ncol and z_ncol vectors respectively. Rearrange them according to the color ordering.
  Vector r_copy;
  r_copy.values = new double[A.localNumberOfRows];
  Vector z_copy;
  z_copy.values = new double[A.localNumberOfRows];

  for(int i = 0; i < A.localNumberOfRows; i++)
  {
    r_copy.values[i] = y_ncol.values[A_ref.colors[i]];
    z_copy.values[i] = z_ncol.values[A_ref.colors[i]];
  }
  r_copy.localLength = y_ncol.localLength;
  z_copy.localLength = z_ncol.localLength;

  // Call ComputeMG with reordered r_copy, z_copy and reference sparse matrix. 
  ierr = ComputeMG(A, A_ref, r_copy, z_copy); // z_ncol = Minv*y_ncol

  // Restore the z_ncol vector from z_copy. 
  for(int i = 0; i < A_ref.localNumberOfRows; i++)
  {
    z_ncol.values[A_ref.colors[i]] = z_copy.values[i];
  }

  if (ierr) {
    HPCG_fout << "Error in call to MG: " << ierr << ".\n" << endl;
  }
  double xtMinvy = 0.0;

  clEnqueueWriteBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(), A.d_Ap.values, CL_TRUE, 0,
                       A.d_Ap.num_values * sizeof(double), z_ncol.values, 0, NULL, NULL);

  ierr = ComputeDotProduct(A.d_b, A.d_Ap, A.d_minusAlpha, t4, A.createResult); // x'*Minv*y
  if (ierr) {
    HPCG_fout << "Error in call to dot: " << ierr << ".\n" << endl;
  }

  clEnqueueReadBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(), A.d_b.values, CL_TRUE, 0,
                      A.d_b.num_values * sizeof(double), x_ncol.values, 0, NULL, NULL);

  // Next, compute z'*Minv*x
  // Use r_copy and z_copy to copy x_ncol and z_ncol vectors respectively. Rearrange them according to the color ordering.
  for(int i = 0; i < A.localNumberOfRows; i++)
  {
    r_copy.values[i] = x_ncol.values[A_ref.colors[i]];
    z_copy.values[i] = z_ncol.values[A_ref.colors[i]];
  }
  r_copy.localLength = x_ncol.localLength;
  z_copy.localLength = z_ncol.localLength;

  // Call ComputeMG with reordered r_copy, z_copy and reference sparse matrix. 
  ierr = ComputeMG(A, A_ref, r_copy, z_copy); // z_ncol = Minv*x_ncol

  /* Restore the z_ncol vector from z_copy. Restore back the MgData from reference sparse matrix 
  to A matrix. */
  if(A.level != 3)
  {
    for(int i = 0; i < A_ref.localNumberOfRows; i++)
    {
      z_ncol.values[A_ref.colors[i]] = z_copy.values[i];
      A.mgData->Axf->values[A_ref.colors[i]] = A_ref.mgData->Axf->values[i];
    }
    for(int i = 0; i < A.mgData->rc->localLength; i++)
    {
      A.mgData->rc->values[A_ref.Ac->colors[i]] = A_ref.mgData->rc->values[i];
      A.mgData->xc->values[A_ref.Ac->colors[i]] = A_ref.mgData->xc->values[i];
    }
  }

  if (ierr) {
    HPCG_fout << "Error in call to MG: " << ierr << ".\n" << endl;
  }
  double ytMinvx = 0.0;

  clEnqueueWriteBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(), A.d_Ap.values, CL_TRUE, 0,
                       A.d_Ap.num_values * sizeof(double), z_ncol.values, 0, NULL, NULL);

  ierr = ComputeDotProduct(A.d_p, A.d_Ap, A.d_pAp, t4, A.createResult); // y'*Minv*x
  if (ierr) {
    HPCG_fout << "Error in call to dot: " << ierr << ".\n" << endl;
  }

  clEnqueueReadBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(), A.d_minusAlpha.value, CL_TRUE, 0,
                      sizeof(double), &xtMinvy, 0, NULL, NULL);
  clEnqueueReadBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(), A.d_pAp.value, CL_TRUE, 0,
                      sizeof(double), &ytMinvx, 0, NULL, NULL);


  testsymmetry_data.depsym_mg = std::fabs((long double)(xtMinvy - ytMinvx)) / ((xNorm2 * ANorm * yNorm2 + yNorm2 * ANorm * xNorm2) * (DBL_EPSILON));
  if (testsymmetry_data.depsym_mg > 1.0) {
    ++testsymmetry_data.count_fail;  // If the difference is > 1, count it wrong
  }
  if (A.geom->rank == 0) {
    HPCG_fout << "Departure from symmetry (scaled) for MG abs(x'*Minv*y - y'*Minv*x) = " << testsymmetry_data.depsym_mg << endl;
  }

  CopyVector(xexact, x_ncol); // Copy exact answer into overlap vector

  clEnqueueWriteBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(), A.d_b.values, CL_TRUE, 0,
                       A.d_b.num_values * sizeof(double), x_ncol.values, 0, NULL, NULL);

  int numberOfCalls = 2;
  double residual = 0.0;
  for (int i = 0; i < numberOfCalls; ++i) {
    ierr = ComputeSPMV(A.d_A, A.d_b, A.d_Ap, A.d_alpha, A.d_beta, A.createResult); // b_computed = A*x_overlap
    if (ierr) {
      HPCG_fout << "Error in call to SpMV: " << ierr << ".\n" << endl;
    }

    clEnqueueReadBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(), A.d_Ap.values, CL_TRUE, 0,
                        A.d_Ap.num_values * sizeof(double), z_ncol.values, 0, NULL, NULL);

    if ((ierr = ComputeResidual(A.localNumberOfRows, b, z_ncol, residual))) {
      HPCG_fout << "Error in call to compute_residual: " << ierr << ".\n" << endl;
    }
    if (A.geom->rank == 0) {
      HPCG_fout << "SpMV call [" << i << "] Residual [" << residual << "]" << endl;
    }
  }

  DeleteVector(x_ncol);
  DeleteVector(y_ncol);
  DeleteVector(z_ncol);

  return 0;
}

