
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

#ifndef COMPUTESYMGS_HPP
#define COMPUTESYMGS_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"

 int ComputeSYMGS_check( const SparseMatrix & A, const Vector & r_copy, Vector & x_copy);
int ComputeSYMGS( const SparseMatrix & A, const Vector & r, Vector & x, double * dur);

#endif // COMPUTESYMGS_HPP
