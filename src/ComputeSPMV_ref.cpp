
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
 @file ComputeSPMV_ref.cpp

 HPCG routine
 */

#include "ComputeSPMV_ref.hpp"
#include <iostream>
//#include "ComputeSPMV.hpp"

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif
#include <vector>
#include <cassert>
#include <time.h>
#include <hcsparse.h>
#inckude <sys/time.h>
  
hcsparseCsrMatrix gCsrMat;
hcdenseVector gX;
hcdenseVector gY;
hcsparseScalar gAlpha;
hcsparseScalar gBeta;
hcsparseStatus status;
double *host_alpha;
double *host_beta;

double *val;
int *col, *rowoff;
ulong *rowBlocks;
  
std::vector<Concurrency::accelerator>acc = Concurrency::accelerator::get_all();
accelerator_view accl_view = (acc[1].create_view()); 

hcsparseControl control(accl_view);

double spmv_time;

int hcsparse_setup(const SparseMatrix h_A)
{
  host_alpha = new double[1];
  host_beta = new double[1];
  host_alpha[0] = 1;
  host_beta[0] = 0;
  
  hcsparseSetup();
  hcsparseInitCsrMatrix(&gCsrMat);
  hcsparseInitScalar(&gAlpha);
  hcsparseInitScalar(&gBeta);
  hcsparseInitVector(&gX);
  hcsparseInitVector(&gY);
  
  gAlpha.offValue = 0;
  gBeta.offValue = 0;
  gX.offValues = 0;
  gY.offValues = 0;

  gCsrMat.offValues = 0;
  gCsrMat.offColInd = 0;
  gCsrMat.offRowOff = 0;

  val = new double[h_A.localNumberOfNonzeros];
  col = new int[h_A.localNumberOfNonzeros];
  rowoff = new int[h_A.localNumberOfRows + 1];
  rowBlocks = new ulong[h_A.localNumberOfNonzeros];
    
  return 0;
}

int hcsparse_csrmv_adaptive(const SparseMatrix & h_A, Vector & h_x, Vector & h_y)
{  
  static int call_no;
  if (!call_no)
  {
    hcsparse_setup(h_A);
    ++call_no;
  }
 
  int num_nonzero, num_row, num_col;
  num_nonzero = h_A.localNumberOfNonzeros;
  num_row = h_A.localNumberOfRows;
  num_col = h_A.localNumberOfColumns;
  
  gCsrMat.num_rows = num_row;
  gCsrMat.num_cols = num_col;
  gCsrMat.num_nonzeros = num_nonzero;
 
  Concurrency::array_view<double> dev_X(num_col, h_x.values);
  Concurrency::array_view<double> dev_Y(num_row, h_y.values);
  Concurrency::array_view<double> dev_alpha(1, host_alpha);
  Concurrency::array_view<double> dev_beta(1, host_beta);

  gAlpha.value = &dev_alpha;
  gBeta.value = &dev_beta;
  gX.values = &dev_X;
  gY.values = &dev_Y;
  
  gX.num_values = num_col;
  gY.num_values = num_row;
  
  int k = 0;
  rowoff[0] = 0;
  for(int i = 1; i <= h_A.localNumberOfRows; i++)
    rowoff[i] = rowoff[i - 1] + h_A.nonzerosInRow[i - 1];
  for(int i = 0; i < h_A.localNumberOfRows; i++) 
  {
     for(int j = 0; j < h_A.nonzerosInRow[i]; j++)
     {
       val[k] = h_A.matrixValues[i][j];
       col[k] = h_A.mtxIndL[i][j];
       k++;
     }
  }

  Concurrency::array_view<double> av_values(num_nonzero, val);
  Concurrency::array_view<int> av_rowOff(num_row+1, rowoff);
  Concurrency::array_view<int> av_colIndices(num_nonzero, col);
  Concurrency::array_view<ulong> av_rowBlocks(num_nonzero, rowBlocks);

  gCsrMat.values = &av_values;
  gCsrMat.rowOffsets = &av_rowOff;
  gCsrMat.colIndices = &av_colIndices;
  gCsrMat.rowBlocks = &av_rowBlocks;
  
  hcsparseCsrMetaSize(&gCsrMat, &control);
  hcsparseCsrMetaCompute(&gCsrMat, &control);

  hcsparseDcsrmv(&gAlpha, &gCsrMat, &gX, &gBeta, &gY, &control); 

  av_values.synchronize();
  av_rowOff.synchronize();
  av_colIndices.synchronize();
  av_rowBlocks.synchronize();
  dev_X.synchronize();
  dev_Y.synchronize();
  dev_alpha.synchronize();
  dev_beta.synchronize();

  delete [] val;
  delete [] col;
  delete [] rowoff;
  delete [] rowBlocks;
  
  return 0;
}

/*!
  Routine to compute matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This is the reference SPMV implementation.  It CANNOT be modified for the
  purposes of this benchmark.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV
*/
int ComputeSPMV_ref( const SparseMatrix & A, Vector & x, Vector & y) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  
  assert(x.localLength>=A.localNumberOfColumns); // Test vector lengths
  assert(y.localLength>=A.localNumberOfRows);

#ifndef HPCG_NO_MPI
    ExchangeHalo(A,x);
#endif

  /*const double * const xv = x.values;
  double * const yv = y.values;
  const local_int_t nrow = A.localNumberOfRows;
#ifndef HPCG_NO_OPENMP
  #pragma omp parallel for
#endif
  for (local_int_t i=0; i< nrow; i++)  {
    double sum = 0.0;
    const double * const cur_vals = A.matrixValues[i];
    const local_int_t * const cur_inds = A.mtxIndL[i];
    const int cur_nnz = A.nonzerosInRow[i];

    for (int j=0; j< cur_nnz; j++)
      sum += cur_vals[j]*xv[cur_inds[j]];
    yv[i] = sum;
  }
    
  double *host_res = new double[y.localLength];
  for (int j=0; j< y.localLength; j++)
    host_res[j] = yv[j];
    
  for (int j=0; j< y.localLength; j++)
    yv[j] = 0;*/
  
  hcsparse_csrmv_adaptive(A, x, y);
  
  /*for (int j=0; j< y.localLength; j++)
  {
    /*if (yv[j] != host_res[j])
      std::cerr << "\nhost: " << host_res[j] << '\t' << "device: " << yv[j];*/
      
       /*double diff = std::abs(host_res[j] - yv[j]);
        if (diff > 0.01)
        {
            std::cout<<host_res[j]<<" "<<yv[j]<<" "<<diff<<std::endl;
            
        }
  }*/
  
  gettimeofday(&stop, NULL);
  spmv_time += ((((stop.tv_sec * 1000000) + stop.tv_usec) - ((start.tv_sec * 1000000) + start.tv_usec)) / 1000000.0);
  
  return 0;
}
