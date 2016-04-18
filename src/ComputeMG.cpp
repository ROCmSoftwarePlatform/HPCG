
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
#include "ComputeSPMV_ref.hpp"
#include "ComputeRestriction_ref.hpp"
#include "ComputeProlongation_ref.hpp"

#include "Vector.hpp"
#include <iostream>
 using namespace std;
/*!
  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On exit contains the result of the multigrid V-cycle with r as the RHS, x is the approximation to Ax = r.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeMG_ref
*/

// free the reference matrix
void free_refmatrix(SparseMatrix &A)
{
  for(int i =0 ; i < A.localNumberOfRows; i++)
  {
    delete [] A.matrixValues[i];
    delete [] A.mtxIndL[i];
  }
}

// free vector
void free_vector(Vector &v)
{
  delete [] v.values;
}

// copy the vector back to the original index after computation
void copy_back_vectors(Vector &x, Vector &x_copy, std::vector<local_int_t> &colors, local_int_t nrows)
{
  for( int i = 0; i < nrows; i++)
  {
    x.values[colors[i]] = x_copy.values[i];
  }
}

// rearrange the vector keeping the color vector as reference index
void rearrange_vector(const Vector &x, Vector &x_copy, std::vector<local_int_t> &colors, local_int_t nrows)
{
  for(int i = 0; i < nrows; i++)
  {
    x_copy.values[i] = x.values[colors[i]];
  }
   x_copy.localLength = x.localLength;
}

int ComputeMG(const SparseMatrix  & A, SparseMatrix &A_ref , const Vector & r, Vector & x, double *dur) {

  assert(x.localLength==A.localNumberOfColumns); 

  Vector r_copy; // rhs vector to be rearranged
  r_copy.values = new double[A.localNumberOfRows];

  ZeroVector(x); 
  int ierr = 0;
  if (A.mgData!=0) { // Go to next coarse level if defined

    int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
    for (int i=0; i< numberOfPresmootherSteps; ++i) 
    {
      Vector x_copy;
      x_copy.values = new double[A.localNumberOfRows];

      /* Rearrange x vector according to the color index order
       and store it in x_copy */
      rearrange_vector(x, x_copy, A_ref.colors, A_ref.localNumberOfRows);
      
      /* Rearrange rhs vector according to the color index order 
      and store it in r_copy */
      rearrange_vector(r, r_copy, A_ref.colors, A_ref.localNumberOfRows);
      
      /* call symgs by with reordered reference sparse matrix , rhs vector
      and x vector */
      ierr += ComputeSYMGS(A_ref, r_copy, x_copy, dur);

      /* copy back the x vector back to the original index*/
      copy_back_vectors(x, x_copy, A_ref.colors, A.localNumberOfRows);
     
      free_vector(x_copy); // free the reordered x_copy vector
     }
    if (ierr!=0) return ierr;
    ierr = ComputeSPMV_ref(A, x, *A.mgData->Axf); if (ierr!=0) return ierr;
    // Perform restriction operation using simple injection
    ierr = ComputeRestriction_ref(A, r);  if (ierr!=0) return ierr;
    ierr = ComputeMG(*A.Ac, *A_ref.Ac, *A.mgData->rc, *A.mgData->xc, dur);  if (ierr!=0) return ierr;
    ierr = ComputeProlongation_ref(A, x);  if (ierr!=0) return ierr;
    int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
    for (int i=0; i< numberOfPostsmootherSteps; ++i) 
    {
      Vector x_copy;
      x_copy.values = new double[A.localNumberOfRows];
      
      /* Rearrange x vector according to the color index order
       and store it in x_copy */
      rearrange_vector(r, r_copy, A_ref.colors, A_ref.localNumberOfRows);
      
      /* Rearrange rhs vector according to the color index order 
      and store it in r_copy */
      rearrange_vector(x, x_copy, A_ref.colors, A_ref.localNumberOfRows);
      
      /* call symgs by with reordered reference sparse matrix , rhs vector
      and x vector */
      ierr += ComputeSYMGS(A_ref, r_copy, x_copy, dur);
      
      /* copy back the x vector back to the original index*/
      copy_back_vectors(x, x_copy, A_ref.colors, A.localNumberOfRows);
      
      free_vector(x_copy);// free the reordered x_copy vector
    }
    if (ierr!=0) return ierr;
  }
  else {
      Vector x_copy;
      x_copy.values = new double[A.localNumberOfRows];
      
      /* Rearrange x vector according to the color index order
       and store it in x_copy */
      rearrange_vector(r, r_copy, A_ref.colors, A_ref.localNumberOfRows);
      
      /* Rearrange rhs vector according to the color index order 
      and store it in r_copy */
      rearrange_vector(x, x_copy, A_ref.colors, A_ref.localNumberOfRows);
      
      /* call symgs by with reordered reference sparse matrix , rhs vector
      and x vector */
      ierr += ComputeSYMGS(A_ref, r_copy, x_copy, dur);
      
      /* copy back the x vector back to the original index*/
      copy_back_vectors(x, x_copy, A_ref.colors, A.localNumberOfRows);
      
      free_vector(x_copy);// free the reordered x_copy vector
      if (ierr!=0) return ierr;
  }
  free_vector(r_copy); // free the reordered r_copy vector

  return 0;
}
