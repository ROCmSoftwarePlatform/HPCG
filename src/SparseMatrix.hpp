
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
 @file SparseMatrix.hpp

 HPCG data structures for the sparse matrix
 */

#ifndef SPARSEMATRIX_HPP
#define SPARSEMATRIX_HPP

#include <map>
#include <vector>
#include <cassert>
#include "Geometry.hpp"
#include "Vector.hpp"
#include "MGData.hpp"
#include "clSPARSE.h"
#include "clSPARSE-error.h"

#ifdef __OCL__
#include <CL/cl.hpp>
#endif

struct SparseMatrix_STRUCT {
  char  * title; //!< name of the sparse matrix
  Geometry * geom; //!< geometry associated with this matrix
  global_int_t totalNumberOfRows; //!< total number of matrix rows across all processes
  global_int_t totalNumberOfNonzeros; //!< total number of matrix nonzeros across all processes
  local_int_t localNumberOfRows; //!< number of rows local to this process
  local_int_t localNumberOfColumns;  //!< number of columns local to this process
  local_int_t localNumberOfNonzeros;  //!< number of nonzeros local to this process
  char  * nonzerosInRow;  //!< The number of nonzeros in a row will always be 27 or fewer
  global_int_t ** mtxIndG; //!< matrix indices as global values
  local_int_t ** mtxIndL; //!< matrix indices as local values
  double ** matrixValues; //!< values of matrix entries
  double ** matrixDiagonal; //!< values of matrix diagonal entries
  std::map< global_int_t, local_int_t > globalToLocalMap; //!< global-to-local mapping
  std::vector< global_int_t > localToGlobalMap; //!< local-to-global mapping
  mutable bool isDotProductOptimized;
  mutable bool isSpmvOptimized;
  mutable bool isMgOptimized;
  mutable bool isWaxpbyOptimized;
  std::vector<local_int_t> colors; //  save the reordered row index.
  std::vector<local_int_t> counters; // save the color offset.
  /*!
   This is for storing optimized data structres created in OptimizeProblem and
   used inside optimized ComputeSPMV().
   */
  mutable struct SparseMatrix_STRUCT * Ac; // Coarse grid matrix
  mutable MGData * mgData; // Pointer to the coarse level data for this fine matrix
  void * optimizationData;  // pointer that can be used to store implementation-specific data

#ifndef HPCG_NO_MPI
  local_int_t numberOfExternalValues; //!< number of entries that are external to this process
  int numberOfSendNeighbors; //!< number of neighboring processes that will be send local data
  local_int_t totalToBeSent; //!< total number of entries to be sent
  local_int_t * elementsToSend; //!< elements to send to neighboring processes
  int * neighbors; //!< neighboring processes
  local_int_t * receiveLength; //!< lenghts of messages received from neighboring processes
  local_int_t * sendLength; //!< lenghts of messages sent to neighboring processes
  double * sendBuffer; //!< send buffer for non-blocking sends

#endif

  double *val;
  clsparseCreateResult createResult;
  cldenseVector d_p, d_Ap, d_b, d_r, d_x;
  clsparseCsrMatrix d_A;
  clsparseScalar d_alpha, d_beta, d_normr, d_minus; 
  clsparseScalar d_rtz, d_oldrtz, d_minusAlpha, d_pAp;
  clsparseCsrMatrix Od_A, d_Q, d_Qt, d_A_ref;
  float *fval;
  int *fcol, *frowOff;

#ifdef __OCL__
cl_mem  clMatrixValues;
cl_mem  clMtxIndL;
cl_mem  clNonzerosInRow;
cl_mem  clMatrixDiagonal;
double* mtxDiagonal;
double* mtxValue;
local_int_t* matrixIndL;
#endif
};
typedef struct SparseMatrix_STRUCT SparseMatrix;

/*!
  Initializes the known system matrix data structure members to 0.

  @param[in] A the known system matrix
 */
inline void InitializeSparseMatrix(SparseMatrix & A, Geometry * geom) {
  A.title = 0;
  A.geom = geom;
  A.totalNumberOfRows = 0;
  A.totalNumberOfNonzeros = 0;
  A.localNumberOfRows = 0;
  A.localNumberOfColumns = 0;
  A.localNumberOfNonzeros = 0;
  A.nonzerosInRow = 0;
  A.mtxIndG = 0;
  A.mtxIndL = 0;
  A.matrixValues = 0;
  A.matrixDiagonal = 0;

  // Optimization is ON by default. The code that switches it OFF is in the
  // functions that are meant to be optimized.
  A.isDotProductOptimized = true;
  A.isSpmvOptimized       = true;
  A.isMgOptimized      = true;
  A.isWaxpbyOptimized     = true;

#ifndef HPCG_NO_MPI
  A.numberOfExternalValues = 0;
  A.numberOfSendNeighbors = 0;
  A.totalToBeSent = 0;
  A.elementsToSend = 0;
  A.neighbors = 0;
  A.receiveLength = 0;
  A.sendLength = 0;
  A.sendBuffer = 0;
#endif

  A.val = NULL;
#ifdef __OCL__
  A.clMatrixValues = NULL;
  A.clMtxIndL = NULL;
  A.clNonzerosInRow = NULL;
  A.clMatrixDiagonal = NULL;
  A.mtxDiagonal = NULL;
  A.mtxValue = NULL;
  A.matrixIndL = NULL;
#endif
  A.mgData = 0; // Fine-to-coarse grid transfer initially not defined.
  A.Ac =0;
  return;
}

/*!
  Copy values from matrix diagonal into user-provided vector.

  @param[in] A the known system matrix.
  @param[inout] diagonal  Vector of diagonal values (must be allocated before call to this function).
 */
inline void CopyMatrixDiagonal(SparseMatrix & A, Vector & diagonal) {
    double ** curDiagA = A.matrixDiagonal;
    double * dv = diagonal.values;
    assert(A.localNumberOfRows==diagonal.localLength);
    for (local_int_t i=0; i<A.localNumberOfRows; ++i) dv[i] = *(curDiagA[i]);
  return;
}
/*!
  Replace specified matrix diagonal value.

  @param[inout] A The system matrix.
  @param[in] diagonal  Vector of diagonal values that will replace existing matrix diagonal values.
 */
inline void ReplaceMatrixDiagonal(SparseMatrix & A, Vector & diagonal) {
    double ** curDiagA = A.matrixDiagonal;
    double * dv = diagonal.values;
    assert(A.localNumberOfRows==diagonal.localLength);
    for (local_int_t i=0; i<A.localNumberOfRows; ++i) *(curDiagA[i]) = dv[i];
  return;
}
/*!
  Deallocates the members of the data structure of the known system matrix provided they are not 0.

  @param[in] A the known system matrix
 */
inline void DeleteMatrix(SparseMatrix & A) {

  for (local_int_t i = 0; i< A.localNumberOfRows; ++i) {
    delete [] A.matrixValues[i];
    delete [] A.mtxIndG[i];
    delete [] A.mtxIndL[i];
  }

  if (A.title)                  delete [] A.title;
  if (A.nonzerosInRow)             delete [] A.nonzerosInRow;
  if (A.mtxIndG) delete [] A.mtxIndG;
  if (A.mtxIndL) delete [] A.mtxIndL;
  if (A.matrixValues) delete [] A.matrixValues;
  if (A.matrixDiagonal)           delete [] A.matrixDiagonal;

#ifndef HPCG_NO_MPI
  if (A.elementsToSend)       delete [] A.elementsToSend;
  if (A.neighbors)              delete [] A.neighbors;
  if (A.receiveLength)            delete [] A.receiveLength;
  if (A.sendLength)            delete [] A.sendLength;
  if (A.sendBuffer)            delete [] A.sendBuffer;
#endif

  if (A.geom!=0) { delete A.geom; A.geom = 0;}
  if (A.Ac!=0) { DeleteMatrix(*A.Ac); delete A.Ac; A.Ac = 0;} // Delete coarse matrix
  if (A.mgData!=0) { DeleteMGData(*A.mgData); delete A.mgData; A.mgData = 0;} // Delete MG data

  clReleaseMemObject(A.d_alpha.value);
  clReleaseMemObject(A.d_beta.value);
  clReleaseMemObject(A.d_normr.value);
  clReleaseMemObject(A.d_minus.value);

  clsparseCsrMetaDelete(&A.d_A);

  clReleaseMemObject(A.d_A.col_indices);
  clReleaseMemObject(A.d_A.row_pointer);
  clReleaseMemObject(A.d_A.values);
  clReleaseMemObject(A.d_p.values);
  clReleaseMemObject(A.d_Ap.values);
  clReleaseMemObject(A.d_b.values);
  clReleaseMemObject(A.d_r.values);
  clReleaseMemObject(A.d_x.values);
  clReleaseMemObject(A.d_rtz.value);
  clReleaseMemObject(A.d_oldrtz.value);
  clReleaseMemObject(A.d_minusAlpha.value);
  clReleaseMemObject(A.d_pAp.value);

  clReleaseMemObject(A.d_Q.col_indices);
  clReleaseMemObject(A.d_Q.row_pointer);
  clReleaseMemObject(A.d_Q.values);

  clReleaseMemObject(A.Od_A.col_indices);
  clReleaseMemObject(A.Od_A.row_pointer);
  clReleaseMemObject(A.Od_A.values);

  clReleaseMemObject(A.d_Qt.col_indices);
  clReleaseMemObject(A.d_Qt.row_pointer);
  clReleaseMemObject(A.d_Qt.values);
  
  clReleaseMemObject(A.d_A_ref.col_indices);
  clReleaseMemObject(A.d_A_ref.row_pointer);
  clReleaseMemObject(A.d_A_ref.values);

  
  clsparseReleaseControl(A.createResult.control);

  delete [] A.fval;
  delete [] A.fcol;
  delete [] A.frowOff;

  return;
}

#endif // SPARSEMATRIX_HPP
