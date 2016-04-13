
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

#ifndef COMPUTEWAXPBY_HPP
#define COMPUTEWAXPBY_HPP
#include "Vector.hpp"
#include "clSPARSE.h"
int ComputeWAXPBY(clsparseScalar alpha, cldenseVector &x, clsparseScalar beta,
                  cldenseVector &y, cldenseVector &w, clsparseCreateResult &createResult);
#endif // COMPUTEWAXPBY_HPP
