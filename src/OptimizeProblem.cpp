
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
 @file OptimizeProblem.cpp

 HPCG routine
 */

#include "OptimizeProblem.hpp"
#include <algorithm>
/*!
  Optimizes the data structures used for CG iteration to increase the
  performance of the benchmark version of the preconditioned CG algorithm.

  @param[inout] A      The known system matrix, also contains the MG hierarchy in attributes Ac and mgData.
  @param[inout] data   The data structure with all necessary CG vectors preallocated
  @param[inout] b      The known right hand side vector
  @param[inout] x      The solution vector to be computed in future CG iteration
  @param[inout] xexact The exact solution vector

  @return returns 0 upon success and non-zero otherwise

  @see GenerateGeometry
  @see GenerateProblem
*/

// leveling to find levels based on dependency to be deployed in level scheduling algorithm.

void  leveling(SparseMatrix &A){  
  
  std::vector<int> root_nodes(A.localNumberOfRows,-1);
  double ** matrix = new double*[A.localNumberOfRows];
  for(int i = 0; i < A.localNumberOfRows; i++)
  {
    matrix[i] = new double[27];
  }
  int root_count;
  int flag = 1;
  int k = 0;
  A.level_no = 0;

  for(int i = 0; i < A.localNumberOfRows; i++)
  {
    for(int j = 0; j < A.nonzerosInRow[i]; j++)
    {
      matrix[i][j] = A.matrixValues[i][j];
    }
  }

  while(1)
  {  
     flag = 1;
     k = 0;

    for (int i = 0; i < A.localNumberOfRows ; i++)
    {   
      if(A.level_array[i] == -1)
      {
        for(int j = 0;j < A.nonzerosInRow[i] ;j++)
        {
          if(i > A.mtxIndL[i][j])
          {
            if(matrix[i][j] != 0)
              flag = 0;
            else
              continue;
          }
        }
        if(flag)
        {
          root_nodes[k++] = i;
        }
      }
    }

    root_count = std::count(root_nodes.begin(), root_nodes.end(), -1);
         
    for(int m = 0; m < k; m++)
    {
       for(int i = 0; i < A.localNumberOfRows; i++)
       {
          for(int j = 0; j < A.nonzerosInRow[i]; j++)
          {
            if(i > j)
            {
              if(A.mtxIndL[i][j] == root_nodes[m])
              {
                matrix[i][j] = 0;
              }
            }
            else
              continue;
          }
       }
    }
     

    for(int i = 0 ;root_nodes[i]!=-1;i++)
    {
      A.level_array[root_nodes[i]] = A.level_no;
      root_nodes[i] = -1;
    }

    int left = std::count(A.level_array.begin(), A.level_array.end(), -1);    
    if(left == 0 || A.level_no > A.localNumberOfRows)
      break;
    A.level_no++;
 }
 for(int i = 0; i < A.localNumberOfRows; i++)
 {
  delete [] matrix[i];
 }
 delete [] matrix;
}

int OptimizeProblem(SparseMatrix & A, CGData & data, Vector & b, Vector & x, Vector & xexact) {

  // This function can be used to completely transform any part of the data structures.
  // Right now it does nothing, so compiling with a check for unused variables results in complaints

#if defined(HPCG_USE_MULTICOLORING)
  const local_int_t nrow = A.localNumberOfRows;
  std::vector<local_int_t> colors(nrow, nrow); // value `nrow' means `uninitialized'; initialized colors go from 0 to nrow-1
  int totalColors = 1;
  colors[0] = 0; // first point gets color 0

  // Finds colors in a greedy (a likely non-optimal) fashion.

  for (local_int_t i=1; i < nrow; ++i) {
    if (colors[i] == nrow) { // if color not assigned
      std::vector<int> assigned(totalColors, 0);
      int currentlyAssigned = 0;
      const local_int_t * const currentColIndices = A.mtxIndL[i];
      const int currentNumberOfNonzeros = A.nonzerosInRow[i];

      for (int j=0; j< currentNumberOfNonzeros; j++) { // scan neighbors
        local_int_t curCol = currentColIndices[j];
        if (curCol < i) { // if this point has an assigned color (points beyond `i' are unassigned)
          if (assigned[colors[curCol]] == 0)
            currentlyAssigned += 1;
          assigned[colors[curCol]] = 1; // this color has been used before by `curCol' point
        } // else // could take advantage of indices being sorted
      }

      if (currentlyAssigned < totalColors) { // if there is at least one color left to use
        for (int j=0; j < totalColors; ++j)  // try all current colors
          if (assigned[j] == 0) { // if no neighbor with this color
            colors[i] = j;
            break;
          }
      } else {
        if (colors[i] == nrow) {
          colors[i] = totalColors;
          totalColors += 1;
        }
      }
    }
  }

  std::vector<local_int_t> counters(totalColors);
  for (local_int_t i=0; i<nrow; ++i)
    counters[colors[i]]++;

  local_int_t old, old0;
  for (int i=1; i < totalColors; ++i) {
    old0 = counters[i];
    counters[i] = counters[i-1] + old;
    old = old0;
  }
  counters[0] = 0;

  // translate `colors' into a permutation
  for (local_int_t i=0; i<nrow; ++i) // for each color `c'
    colors[i] = counters[colors[i]]++;
#endif

  return 0;
}

// Helper function (see OptimizeProblem.hpp for details)
double OptimizeProblemMemoryUse(const SparseMatrix & A) {

  return 0.0;

}
