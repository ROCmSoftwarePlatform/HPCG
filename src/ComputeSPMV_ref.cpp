
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
#include "ComputeSPMV.hpp"

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif
#include <vector>
#include <cassert>
#include <sys/time.h>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <CL/cl.hpp>
#include "clSPARSE.h"

cl_platform_id *platform;
cl_context context;
cl_device_id *device;
cl_command_queue command_queue;
cl_int err;
cl_int cl_status;

clsparseControl control;
clsparseStatus status;
clsparseScalar alpha;
clsparseScalar beta;
cldenseVector x;
cldenseVector y;
clsparseCsrMatrix A;
  
double spmv_time;
 
int clsparse_setup()
{
  cl_uint ret_num_of_platforms = 0;
  cl_int err = clGetPlatformIDs(0, NULL, &ret_num_of_platforms);
  assert(err == CL_SUCCESS && "clGetPlatformIDs\n");

  platform = (cl_platform_id*)malloc(ret_num_of_platforms * sizeof(cl_platform_id));
  err = clGetPlatformIDs(ret_num_of_platforms, platform, 0);
  assert(err == CL_SUCCESS && "clGetPlatformIDs\n");

  cl_uint ret_num_of_devices = 0;
  err = clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_GPU, 0, NULL, &ret_num_of_devices);
   
  assert(err == CL_SUCCESS && "clGetDeviceIds failed\n");

  device = (cl_device_id*)malloc(ret_num_of_devices * sizeof(cl_device_id));
  err = clGetDeviceIDs(platform[0],CL_DEVICE_TYPE_GPU , ret_num_of_devices, device, 0);
  assert(err == CL_SUCCESS && "clGetDeviceIds failed\n");

  cl_context_properties property[] = {
                             	CL_CONTEXT_PLATFORM,
                        (cl_context_properties)(platform[0]),
                        0};

  context = clCreateContext(property, ret_num_of_devices, &device[0], NULL, NULL, &err);
  assert(err == CL_SUCCESS && "clCreateContext failed\n");

  command_queue = clCreateCommandQueue(context, device[0], 0, &err);
  assert(err == CL_SUCCESS && "clCreateCommandQueue failed \n");
  
  clsparseInitScalar(&alpha);
  alpha.value = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double),
                               nullptr, &cl_status);
                               
  clsparseInitScalar(&beta);

  beta.value = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double),
                              nullptr, &cl_status);
  clsparseInitVector(&x);
  clsparseInitVector(&y);  
  clsparseInitCsrMatrix(&A);
  
  status = clsparseSetup();
  if (status != clsparseSuccess)
  {
      std::cout << "Problem with executing clsparseSetup()" << std::endl;
      return -3;
  }


  // Create clsparseControl object
  control = clsparseCreateControl(command_queue, &status);
  if (status != CL_SUCCESS)
  {
      std::cout << "Problem with creating clSPARSE control object"
                <<" error [" << status << "]" << std::endl;
      return -4;
  }
   
  double one = 1.0;
  double zero = 0.0;

  // alpha = 1;
  double* halpha = (double*) clEnqueueMapBuffer(command_queue, alpha.value, CL_TRUE, CL_MAP_WRITE,
                                              0, sizeof(double), 0, nullptr, nullptr, &cl_status);
  *halpha = one;

  cl_status = clEnqueueUnmapMemObject(command_queue, alpha.value, halpha,
                                      0, nullptr, nullptr);

  //beta = 0;
  double* hbeta = (double*) clEnqueueMapBuffer(command_queue, beta.value, CL_TRUE, CL_MAP_WRITE,
                                             0, sizeof(double), 0, nullptr, nullptr, &cl_status);
  *hbeta = zero;

  cl_status = clEnqueueUnmapMemObject(command_queue, beta.value, hbeta,
                                      0, nullptr, nullptr);
                                      
  return 0;
} 
 
int hpcg2clsparse(const SparseMatrix & hA, Vector & hx, Vector & hy)
{         
  static int call_count;
  if (!call_count)
  {    
    clsparse_setup();    
    ++call_count;     
  }

  A.num_nonzeros = hA.totalNumberOfNonzeros;
  A.num_rows = hA.localNumberOfRows;
  A.num_cols = hA.localNumberOfColumns;
  
  double *val = new double[hA.totalNumberOfNonzeros];
  int *col = new int[hA.totalNumberOfNonzeros];
  int *rowoff = new int[hA.localNumberOfRows + 1];
  
  int k = 0;
  rowoff[0] = 0;
  for(int i = 1; i <= hA.totalNumberOfRows; i++)
    rowoff[i] = rowoff[i - 1] + hA.nonzerosInRow[i - 1];
  for(int i = 0; i < hA.totalNumberOfRows; i++) 
  {
     for(int j = 0; j < hA.nonzerosInRow[i]; j++)
     {
       val[k] = hA.matrixValues[i][j];
       col[k] = hA.mtxIndL[i][j];
       k++;
     }
  }
  
  // Allocate memory for CSR matrix
  A.values = ::clCreateBuffer( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               A.num_nonzeros * sizeof( double ), val, &cl_status );

  A.colIndices = ::clCreateBuffer( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   A.num_nonzeros * sizeof( cl_int ), col, &cl_status );

  A.rowOffsets = ::clCreateBuffer( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   ( A.num_rows + 1 ) * sizeof( cl_int ), rowoff, &cl_status );

  A.rowBlocks = ::clCreateBuffer( context, CL_MEM_READ_ONLY,
                                  A.rowBlockSize * sizeof( cl_ulong ), NULL, &cl_status );

  // This function allocates memory for rowBlocks structure. If not called
  // the structure will not be calculated and clSPARSE will run the vectorized
  // version of SpMV instead of adaptive;

  clsparseCsrMetaSize( &A, control );
  A.rowBlocks = ::clCreateBuffer( context, CL_MEM_READ_WRITE,
          A.rowBlockSize * sizeof( cl_ulong ), NULL, &cl_status );
  clsparseCsrMetaCompute( &A, control );    

  x.num_values = hx.localLength;
  x.values = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, x.num_values * sizeof(double),
                            hx.values, &cl_status);

  y.num_values = hy.localLength;
  y.values = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, y.num_values * sizeof(double),
                            hy.values, &cl_status);

 /* Call the spmv algorithm */
  status = clsparseDcsrmv(&alpha, &A, &x, &beta, &y, control);

  if (status != clsparseSuccess)
  {
      std::cerr << "Problem with execution SpMV algorithm."
                << " Error: " << status << std::endl;
  }

  clEnqueueReadBuffer(command_queue, y.values, CL_TRUE, 0,
                              y.num_values * sizeof(double), hy.values, 0, NULL, NULL );  

  //release mem;
  clReleaseMemObject ( A.values );
  clReleaseMemObject ( A.colIndices );
  clReleaseMemObject ( A.rowOffsets );
  clReleaseMemObject ( A.rowBlocks );

  clReleaseMemObject ( x.values );
  clReleaseMemObject ( y.values );

  delete [] val;
  delete [] col;    
  delete [] rowoff; 
  
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
  
  hpcg2clsparse(A, x, y);  
  
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
