
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

#ifndef OPTIMIZEPROBLEM_HPP
#define OPTIMIZEPROBLEM_HPP

#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "CGData.hpp"
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <iostream>

int OptimizeProblem(const SparseMatrix & A,SparseMatrix & A_ref, std::vector<local_int_t> &colors);
void lubys_graph_coloring (int c,int *row_offset,int *col_index,int *Colors,int *random,int *temp);
void copy_value(int * dest, int *source);
int hash_function(int index , int nnz);

void free_refmatrix_m(SparseMatrix &A);
void copy_sparse_matrix_m(const SparseMatrix &A, SparseMatrix &A_ref);

// This helper function should be implemented in a non-trivial way if OptimizeProblem is non-trivial
// It should return as type double, the total number of bytes allocated and retained after calling OptimizeProblem.
// This value will be used to report Gbytes used in ReportResults (the value returned will be divided by 1000000000.0).

double OptimizeProblemMemoryUse(const SparseMatrix & A);

#endif  // OPTIMIZEPROBLEM_HPP
