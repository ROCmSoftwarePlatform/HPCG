
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
 @file ComputeSPMV.cpp

 HPCG routine
 */

#include "ComputeSPMV.hpp"
#include "ComputeSPMV_ref.hpp"
#include <iostream>
#include <sys/time.h>
#include <CL/cl.hpp>
#include "clSPARSE.h"
#include "clSPARSE-error.h"

cl_platform_id *platform;
cl_context context;
cl_device_id *device;
cl_command_queue command_queue;
cl_int err;
cl_int cl_status;

clsparseCreateResult createResult;
clsparseStatus status;
clsparseScalar alpha;
clsparseScalar beta;
cldenseVector x;
cldenseVector y;
clsparseCsrMatrix A;
  
double *val;
int *col, *rowoff;
//double spmv_time;

int clsparse_setup(const SparseMatrix hA, Vector hx, Vector hy)
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
  createResult = clsparseCreateControl(command_queue);
  CLSPARSE_V( createResult.status, "Failed to create clsparse control" );
   
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
                                      
  val = new double[hA.totalNumberOfNonzeros];
  col = new int[hA.totalNumberOfNonzeros];
  rowoff = new int[hA.localNumberOfRows + 1];     
  
  int k = 0;
  rowoff[0] = 0;
  for(int i = 1; i <= hA.totalNumberOfRows; i++)
    rowoff[i] = rowoff[i - 1] + hA.nonzerosInRow[i - 1];
    
  for(int i = 0; i < hA.totalNumberOfRows; i++) 
  {
     for(int j = 0; j < hA.nonzerosInRow[i]; j++)
     {
       col[k] = hA.mtxIndL[i][j];
       k++;
     }
  }             
  
  A.num_nonzeros = hA.totalNumberOfNonzeros;
  A.num_rows = hA.localNumberOfRows;
  A.num_cols = hA.localNumberOfColumns;
  x.num_values = hx.localLength;
  y.num_values = hy.localLength;                   
               
  A.values = ::clCreateBuffer( context, CL_MEM_READ_ONLY,
                               A.num_nonzeros * sizeof( double ), NULL, &cl_status );
  
  A.col_indices = ::clCreateBuffer( context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                   A.num_nonzeros * sizeof( clsparseIdx_t ), col, &cl_status );

  A.row_pointer = ::clCreateBuffer( context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                   ( A.num_rows + 1 ) * sizeof( clsparseIdx_t ), rowoff, &cl_status );               
                                      
  // This function allocates memory for rowBlocks structure. If not called
  // the structure will not be calculated and clSPARSE will run the vectorized
  // version of SpMV instead of adaptive;
  clsparseCsrMetaCreate( &A, createResult.control );                                      
                                     
  x.values = clCreateBuffer(context, CL_MEM_READ_ONLY, x.num_values * sizeof(double),
                            NULL, &cl_status);  
  y.values = clCreateBuffer(context, CL_MEM_READ_WRITE, y.num_values * sizeof(double),
                            NULL, &cl_status);                                                                        
                                      
  return 0;
} 
 
int hpcg2clsparse(const SparseMatrix & hA, Vector & hx, Vector & hy)
{         
  static int call_count;
  
  if (!call_count) 
  {  
    clsparse_setup(hA, hx, hy); 
    ++call_count; 
  } 
    
  int k = 0;
  for(int i = 0; i < hA.totalNumberOfRows; i++) 
  {
     for(int j = 0; j < hA.nonzerosInRow[i]; j++)
     {
       val[k] = hA.matrixValues[i][j];
       k++;
     }
  }                            
  
  clEnqueueWriteBuffer(command_queue, A.values, CL_TRUE, 0,
                              A.num_nonzeros * sizeof( double ), val, 0, NULL, NULL );                              
  clEnqueueWriteBuffer(command_queue, x.values, CL_TRUE, 0,
                              x.num_values * sizeof( double ), hx.values, 0, NULL, NULL );                                

 /* Call the spmv algorithm */
  status = clsparseDcsrmv(&alpha, &A, &x, &beta, &y, createResult.control);

  if (status != clsparseSuccess)
  {
      std::cerr << "Problem with execution SpMV algorithm."
                << " Error: " << status << std::endl;
  }

  clEnqueueReadBuffer(command_queue, y.values, CL_TRUE, 0,
                              y.num_values * sizeof(double), hy.values, 0, NULL, NULL );  
  
  return 0;
}

/*!
  Routine to compute sparse matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This routine calls the reference SpMV implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV_ref
*/
int ComputeSPMV( const SparseMatrix & A, Vector & x, Vector & y) {
  /*struct timeval start, stop;
  gettimeofday(&start, NULL);*/
  //std::cerr << "SPMV" << '\n';
  // This line and the next two lines should be removed and your version of ComputeSPMV should be used.
  //A.isSpmvOptimized = false;
  //return ComputeSPMV_ref(A, x, y);
  
  hpcg2clsparse(A, x, y);
  
  /*gettimeofday(&stop, NULL);
  spmv_time += ((((stop.tv_sec * 1000000) + stop.tv_usec) - ((start.tv_sec * 1000000) + start.tv_usec)) / 1000000.0);*/
  
  return 0;
  
  
  
}
