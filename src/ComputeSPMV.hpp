
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

#ifndef COMPUTESPMV_HPP
#define COMPUTESPMV_HPP
#include "Vector.hpp"
#include "SparseMatrix.hpp"
#include "clSPARSE.h"
int ComputeSPMV(clsparseCsrMatrix &d_A, cldenseVector &d_x, cldenseVector &d_y,
                clsparseScalar &d_alpha, clsparseScalar &d_beta, clsparseCreateResult createResult);
#endif  // COMPUTESPMV_HPP
