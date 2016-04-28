
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
 @file ComputeMG.cpp

 HPCG routine
 */

#include "ComputeMG.hpp"
#include "ComputeMG_ref.hpp"
#include "OptimizeProblem.hpp"
#include "ComputeSYMGS_ref.hpp"
#include "ComputeSYMGS.hpp"
#include "ComputeSPMV.hpp"
#include "ComputeRestriction_ref.hpp"
#include "ComputeProlongation_ref.hpp"

#include "Vector.hpp"
#include <iostream>
#include "OCL.hpp"
 using namespace std;
/*!
  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On exit contains the result of the multigrid V-cycle with r as the RHS, x is the approximation to Ax = r.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeMG_ref
*/

int ComputeMG(SparseMatrix  & origA, const SparseMatrix  & A, const Vector & r, Vector & x) {

  assert(x.localLength==A.localNumberOfColumns); 

  ZeroVector(x); 
  int ierr = 0;
  if (A.mgData!=0) { // Go to next coarse level if defined

    int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
    for (int i=0; i< numberOfPresmootherSteps; ++i) 
    {
      /* call symgs by with reordered reference sparse matrix , rhs vector
      and x vector */
      ierr += ComputeSYMGS(A, r, x);

     }
    if (ierr!=0) return ierr;
    
    //ierr = ComputeSPMV_ref(A, x, *A.mgData->Axf); 
    
    origA.dA.num_rows = A.totalNumberOfRows;
    origA.dA.num_cols = A.localNumberOfColumns;
    origA.dA.num_nonzeros = A.totalNumberOfNonzeros;
    
    int k = 0;
    for (int i = 0; i < A.totalNumberOfRows; i++) {
      for (int j = 0; j < A.nonzerosInRow[i]; j++) {
        origA.val[k] = A.matrixValues[i][j];
        origA.fcol[k] = A.mtxIndL[i][j];
        k++;
      }
    }
    
    k = 0;
    origA.frowOff[0] = 0;
    for (int i = 1; i <= A.totalNumberOfRows; i++) {
      origA.frowOff[i] = origA.frowOff[i - 1] + A.nonzerosInRow[i - 1];
    }
    
    clEnqueueWriteBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(), origA.dA.values, CL_TRUE, 0,
                       origA.dA.num_nonzeros * sizeof(double), origA.val, 0, NULL, NULL); 
    clEnqueueWriteBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(), origA.dA.col_indices, CL_TRUE, 0,
                        origA.dA.num_nonzeros * sizeof(clsparseIdx_t), origA.fcol, 0, NULL, NULL);  
    clEnqueueWriteBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(), origA.dA.row_pointer, CL_TRUE, 0,
                        (origA.dA.num_rows + 1) * sizeof(clsparseIdx_t), origA.frowOff, 0, NULL, NULL);  
                        
    // This function allocates memory for rowBlocks structure. If not called
    // the structure will not be calculated and clSPARSE will run the vectorized
    // version of SpMV instead of adaptive;
    clsparseCsrMetaCreate(&origA.dA, origA.createResult.control);  
    
    
    origA.dx.num_values = x.localLength; 
    origA.dAxf.num_values = A.mgData->Axf->localLength;  
    clEnqueueWriteBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(), origA.dx.values, CL_TRUE, 0,
                         origA.dx.num_values * sizeof(double), x.values, 0, NULL, NULL);      
                         
    ierr = ComputeSPMV(origA.dA, origA.dx, origA.dAxf, origA.d_alpha, origA.d_beta, origA.createResult);  
    
    clEnqueueReadBuffer(HPCG_OCL::OCL::getOpenCL()->getCommandQueue(), origA.dAxf.values, CL_TRUE, 0,
                      origA.dAxf.num_values * sizeof(double), A.mgData->Axf->values, 0, NULL, NULL);        
    clsparseCsrMetaDelete( &origA.dA );
    
    if (ierr!=0) return ierr;
    // Perform restriction operation using simple injection
    ierr = ComputeRestriction_ref(A, r);  if (ierr!=0) return ierr;
    ierr = ComputeMG(origA, *A.Ac, *A.mgData->rc, *A.mgData->xc);  if (ierr!=0) return ierr;
    ierr = ComputeProlongation_ref(A, x);  if (ierr!=0) return ierr;
    int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
    for (int i=0; i< numberOfPostsmootherSteps; ++i) 
    {
      /* call symgs by with reordered reference sparse matrix , rhs vector
      and x vector */
      ierr += ComputeSYMGS(A, r, x);
    }
    if (ierr!=0) return ierr;
  }
  else {
      
      /* call symgs by with reordered reference sparse matrix , rhs vector
      and x vector */
      ierr += ComputeSYMGS(A, r, x);
     
      if (ierr!=0) return ierr;
  }

  return 0;
}
