
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

extern cl_context context;
extern cl_command_queue command_queue;
extern clsparseCsrMatrix d_A;
extern cldenseVector d_p, d_Ap, d_b, d_r, d_x;
extern clsparseScalar d_alpha, d_beta, d_normr, d_minus; 
  
extern double *val;
extern int *col, *rowoff;

extern clsparseScalar d_rtz, d_oldrtz, d_Beta, d_Alpha, d_minusAlpha, d_pAp;

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
int TestSymmetry(SparseMatrix & A, Vector & b, Vector & xexact, TestSymmetryData & testsymmetry_data) {
//std::cerr << "Test Symmetry \n";
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
  for(int i = 0; i < A.totalNumberOfRows; i++) 
  {
     for(int j = 0; j < A.nonzerosInRow[i]; j++)
     {
       val[k] = A.matrixValues[i][j];
       k++;
     }
  }        
  
  clEnqueueWriteBuffer(command_queue, d_A.values, CL_TRUE, 0,
                              d_A.num_nonzeros * sizeof( double ), val, 0, NULL, NULL ); 

 // Next, compute x'*A*y
 ComputeDotProduct(d_p, d_p, d_rtz, t4);
 int ierr = ComputeSPMV(d_A, d_p, d_Ap); // z_nrow = A*y_overlap
 if (ierr) HPCG_fout << "Error in call to SpMV: " << ierr << ".\n" << endl;
 double xtAy = 0.0;
 ierr = ComputeDotProduct(d_b, d_Ap, d_oldrtz, t4); // x'*A*y
 if (ierr) HPCG_fout << "Error in call to dot: " << ierr << ".\n" << endl;

 // Next, compute y'*A*x
 ComputeDotProduct(d_b, d_b, d_Beta, t4);
 ierr = ComputeSPMV(d_A, d_b, d_Ap); // b_computed = A*x_overlap
 if (ierr) HPCG_fout << "Error in call to SpMV: " << ierr << ".\n" << endl;
 double ytAx = 0.0;
 ierr = ComputeDotProduct(d_p, d_Ap, d_Alpha, t4); // y'*A*x
 if (ierr) HPCG_fout << "Error in call to dot: " << ierr << ".\n" << endl;
 
 clEnqueueReadBuffer(command_queue, d_p.values, CL_TRUE, 0,
                              d_p.num_values * sizeof( double ), y_ncol.values, 0, NULL, NULL );   
 clEnqueueReadBuffer(command_queue, d_Ap.values, CL_TRUE, 0,
                              d_Ap.num_values * sizeof(double), z_ncol.values, 0, NULL, NULL );                                 
 clEnqueueReadBuffer(command_queue, d_Beta.value, CL_TRUE, 0,
                              sizeof(double), &xNorm2, 0, NULL, NULL );  
 clEnqueueReadBuffer(command_queue, d_rtz.value, CL_TRUE, 0,
                              sizeof(double), &yNorm2, 0, NULL, NULL ); 
 clEnqueueReadBuffer(command_queue, d_oldrtz.value, CL_TRUE, 0,
                              sizeof(double), &xtAy, 0, NULL, NULL );  
 clEnqueueReadBuffer(command_queue, d_Alpha.value, CL_TRUE, 0,
                              sizeof(double), &ytAx, 0, NULL, NULL );                                                              

 testsymmetry_data.depsym_spmv = std::fabs((long double) (xtAy - ytAx))/((xNorm2*ANorm*yNorm2 + yNorm2*ANorm*xNorm2) * (DBL_EPSILON));
 if (testsymmetry_data.depsym_spmv > 1.0) ++testsymmetry_data.count_fail;  // If the difference is > 1, count it wrong
 if (A.geom->rank==0) HPCG_fout << "Departure from symmetry (scaled) for SpMV abs(x'*A*y - y'*A*x) = " << testsymmetry_data.depsym_spmv << endl;

 // Test symmetry of symmetric Gauss-Seidel

 // Compute x'*Minv*y
 ierr = ComputeMG(A, y_ncol, z_ncol); // z_ncol = Minv*y_ncol
 if (ierr) HPCG_fout << "Error in call to MG: " << ierr << ".\n" << endl;
 double xtMinvy = 0.0;
 
 clEnqueueWriteBuffer(command_queue, d_Ap.values, CL_TRUE, 0,
                              d_Ap.num_values * sizeof( double ), z_ncol.values, 0, NULL, NULL ); 
 
 ierr = ComputeDotProduct(d_b, d_Ap, d_minusAlpha, t4); // x'*Minv*y
 if (ierr) HPCG_fout << "Error in call to dot: " << ierr << ".\n" << endl;
                           
 clEnqueueReadBuffer(command_queue, d_b.values, CL_TRUE, 0,
                              d_b.num_values * sizeof( double ), x_ncol.values, 0, NULL, NULL );  
 
 // Next, compute z'*Minv*x
 ierr = ComputeMG(A, x_ncol, z_ncol); // z_ncol = Minv*x_ncol
 if (ierr) HPCG_fout << "Error in call to MG: " << ierr << ".\n" << endl;
 double ytMinvx = 0.0;
 
 clEnqueueWriteBuffer(command_queue, d_Ap.values, CL_TRUE, 0,
                              d_Ap.num_values * sizeof( double ), z_ncol.values, 0, NULL, NULL );                          
 
 ierr = ComputeDotProduct(d_p, d_Ap, d_pAp, t4); // y'*Minv*x
 if (ierr) HPCG_fout << "Error in call to dot: " << ierr << ".\n" << endl;
 
 clEnqueueReadBuffer(command_queue, d_minusAlpha.value, CL_TRUE, 0,
                              sizeof(double), &xtMinvy, 0, NULL, NULL );  
 clEnqueueReadBuffer(command_queue, d_pAp.value, CL_TRUE, 0,
                              sizeof(double), &ytMinvx, 0, NULL, NULL ); 
                               

 testsymmetry_data.depsym_mg = std::fabs((long double) (xtMinvy - ytMinvx))/((xNorm2*ANorm*yNorm2 + yNorm2*ANorm*xNorm2) * (DBL_EPSILON));
 if (testsymmetry_data.depsym_mg > 1.0) ++testsymmetry_data.count_fail;  // If the difference is > 1, count it wrong
 if (A.geom->rank==0) HPCG_fout << "Departure from symmetry (scaled) for MG abs(x'*Minv*y - y'*Minv*x) = " << testsymmetry_data.depsym_mg << endl;

 CopyVector(xexact, x_ncol); // Copy exact answer into overlap vector
 
 clEnqueueWriteBuffer(command_queue, d_b.values, CL_TRUE, 0,
                              d_b.num_values * sizeof( double ), x_ncol.values, 0, NULL, NULL ); 

 int numberOfCalls = 2;
 double residual = 0.0;
 for (int i=0; i< numberOfCalls; ++i) {
   ierr = ComputeSPMV(d_A, d_b, d_Ap); // b_computed = A*x_overlap
   if (ierr) HPCG_fout << "Error in call to SpMV: " << ierr << ".\n" << endl;
   
   clEnqueueReadBuffer(command_queue, d_Ap.values, CL_TRUE, 0,
                              d_Ap.num_values * sizeof(double), z_ncol.values, 0, NULL, NULL ); 
   
   if ((ierr = ComputeResidual(A.localNumberOfRows, b, z_ncol, residual)))
     HPCG_fout << "Error in call to compute_residual: " << ierr << ".\n" << endl;
   if (A.geom->rank==0) HPCG_fout << "SpMV call [" << i << "] Residual [" << residual << "]" << endl;
 }
 DeleteVector(x_ncol);
 DeleteVector(y_ncol);
 DeleteVector(z_ncol);

 return 0;
}

