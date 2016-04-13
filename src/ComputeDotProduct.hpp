
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

#ifndef COMPUTEDOTPRODUCT_HPP
#define COMPUTEDOTPRODUCT_HPP
#include "Vector.hpp"
#include "clSPARSE.h"
int ComputeDotProduct(cldenseVector &x, cldenseVector &y, clsparseScalar &r,
                      double &time_allreduce, clsparseCreateResult createResult);

#endif // COMPUTEDOTPRODUCT_HPP
