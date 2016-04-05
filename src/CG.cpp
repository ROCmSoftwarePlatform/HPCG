
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
 @file CG.cpp

 HPCG routine
 */

#include <fstream>

#include <cmath>
 #include <iostream>

#include "hpcg.hpp"

#include "CG.hpp"
#include "mytimer.hpp"
#include "ComputeSPMV.hpp"
#include "ComputeMG.hpp"
 #include "ComputeMG_ref.hpp"
#include "ComputeDotProduct.hpp"
#include "ComputeWAXPBY.hpp"
#include "OptimizeProblem.hpp"
#include "ComputeSPMV_ref.hpp"
#include "ComputeDotProduct_ref.hpp"
#include "ComputeWAXPBY_ref.hpp"
#include <iostream>

#include <CL/cl.hpp>
#include "clSPARSE.h"
#include "clSPARSE-error.h"

// Use TICK and TOCK to time a code section in MATLAB-like fashion
#define TICK()  t0 = mytimer() //!< record current time in 't0'
#define TOCK(t) t += mytimer() - t0 //!< store time difference in 't' using time in 't0'

cl_platform_id *platform;
cl_context context;
cl_device_id *device;
cl_command_queue command_queue;
cl_int err;
cl_int cl_status;

cl_program program;              
cl_kernel kernel1, kernel2, kernel3;                 

clsparseCreateResult createResult;
clsparseStatus status;
clsparseCsrMatrix d_A;
cldenseVector d_p, d_Ap, d_b, d_r, d_x;
clsparseScalar d_alpha, d_beta, d_normr, d_minus; 
  
double *val;
int *col, *rowoff;

clsparseScalar d_rtz, d_oldrtz, d_Beta, d_Alpha, d_minusAlpha, d_pAp;

const char *kernelSource =                                       "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                   \n" \
"__kernel void rtzCopy( __global double *rtz,                    \n" \
"                       __global double *oldrtz)                 \n" \
"{                                                               \n" \
"    //Get our global thread ID                                  \n" \
"    int id = get_global_id(0);                                  \n" \
"                                                                \n" \
"    //Make sure we do not go out of bounds                      \n" \
"    if (!id)                                                    \n" \
"      *oldrtz = *rtz;                                           \n" \
"}                                                               \n" \
"__kernel void computeBeta( __global double *rtz,                \n" \
"                           __global double *oldrtz,             \n" \
"                           __global double *beta)               \n" \
"{                                                               \n" \
"    //Get our global thread ID                                  \n" \
"    int id = get_global_id(0);                                  \n" \
"                                                                \n" \
"    //Make sure we do not go out of bounds                      \n" \
"    if (!id)                                                    \n" \
"      *beta = *rtz / *oldrtz;                                   \n" \
"}                                                               \n" \
"__kernel void computeAlpha( __global double *rtz,               \n" \
"                           __global double *pAp,                \n" \
"                           __global double *alpha,              \n" \
"                           __global double *minusAlpha)         \n" \
"{                                                               \n" \
"    //Get our global thread ID                                  \n" \
"    int id = get_global_id(0);                                  \n" \
"                                                                \n" \
"    //Make sure we do not go out of bounds                      \n" \
"    if (!id)                                                    \n" \
"    {                                                           \n" \
"      *alpha = *rtz / *pAp;                                     \n" \
"      *minusAlpha = -(*alpha);                                  \n" \
"    }                                                           \n" \
"}                                                               \n" \
                                                                "\n" ;

/*!
  Routine to compute an approximate solution to Ax = b

  @param[in]    geom The description of the problem's geometry.
  @param[inout] A    The known system matrix
  @param[inout] data The data structure with all necessary CG vectors preallocated
  @param[in]    b    The known right hand side vector
  @param[inout] x    On entry: the initial guess; on exit: the new approximate solution
  @param[in]    max_iter  The maximum number of iterations to perform, even if tolerance is not met.
  @param[in]    tolerance The stopping criterion to assert convergence: if norm of residual is <= to tolerance.
  @param[out]   niters    The number of iterations actually performed.
  @param[out]   normr     The 2-norm of the residual vector after the last iteration.
  @param[out]   normr0    The 2-norm of the residual vector before the first iteration.
  @param[out]   times     The 7-element vector of the timing information accumulated during all of the iterations.
  @param[in]    doPreconditioning The flag to indicate whether the preconditioner should be invoked at each iteration.

  @return Returns zero on success and a non-zero value otherwise.

  @see CG_ref()
*/

int count_cg;

int opencl_setup()
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
  
   // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1,
                            (const char **) & kernelSource, NULL, &err);
  // Build the program executable 
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
 
    // Create the compute kernel in the program we wish to run
    kernel1 = clCreateKernel(program, "rtzCopy", &err);
    kernel2 = clCreateKernel(program, "computeBeta", &err);
    kernel3 = clCreateKernel(program, "computeAlpha", &err);
 
              
  
  return 0;
}

int clsparse_setup(const SparseMatrix h_A)
{
  status = clsparseSetup();
  if (status != clsparseSuccess)
  {
      std::cout << "Problem with executing clsparseSetup()" << std::endl;
      return -3;
  }

  // Create clsparseControl object
  createResult = clsparseCreateControl(command_queue);
  CLSPARSE_V( createResult.status, "Failed to create clsparse control" );
  
  clsparseInitCsrMatrix(&d_A);
  
  val = new double[h_A.totalNumberOfNonzeros];
  col = new int[h_A.totalNumberOfNonzeros];
  rowoff = new int[h_A.localNumberOfRows + 1];     
  
  int k = 0;
  rowoff[0] = 0;
  for(int i = 1; i <= h_A.totalNumberOfRows; i++)
    rowoff[i] = rowoff[i - 1] + h_A.nonzerosInRow[i - 1];
    
  for(int i = 0; i < h_A.totalNumberOfRows; i++) 
  {
     for(int j = 0; j < h_A.nonzerosInRow[i]; j++)
     {
       col[k] = h_A.mtxIndL[i][j];
       k++;
     }
  }             
  
  d_A.num_nonzeros = h_A.totalNumberOfNonzeros;
  d_A.num_rows = h_A.localNumberOfRows;
  d_A.num_cols = h_A.localNumberOfColumns;                
               
  d_A.values = ::clCreateBuffer( context, CL_MEM_READ_ONLY,
                               d_A.num_nonzeros * sizeof( double ), NULL, &cl_status );
  
  d_A.col_indices = ::clCreateBuffer( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   d_A.num_nonzeros * sizeof( clsparseIdx_t ), col, &cl_status );

  d_A.row_pointer = ::clCreateBuffer( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   ( d_A.num_rows + 1 ) * sizeof( clsparseIdx_t ), rowoff, &cl_status );               
                                      
  // This function allocates memory for rowBlocks structure. If not called
  // the structure will not be calculated and clSPARSE will run the vectorized
  // version of SpMV instead of adaptive;
  clsparseCsrMetaCreate( &d_A, createResult.control );
  
  clsparseInitVector(&d_p);
  clsparseInitVector(&d_Ap);
  clsparseInitVector(&d_b);
  clsparseInitVector(&d_r);
  clsparseInitVector(&d_x);
  
  clsparseInitScalar(&d_alpha);
  clsparseInitScalar(&d_beta);
  clsparseInitScalar(&d_normr);
  clsparseInitScalar(&d_minus);
  
  clsparseInitScalar(&d_rtz);
  clsparseInitScalar(&d_oldrtz);
  clsparseInitScalar(&d_pAp);
  clsparseInitScalar(&d_Alpha);
  clsparseInitScalar(&d_Beta);
  clsparseInitScalar(&d_minusAlpha);
  
  d_p.values = clCreateBuffer(context, CL_MEM_READ_WRITE, d_A.num_rows * sizeof(double),
                            NULL, &cl_status);    
  d_p.num_values = d_A.num_rows; 
  d_Ap.values = clCreateBuffer(context, CL_MEM_READ_WRITE, d_A.num_rows * sizeof(double),
                            NULL, &cl_status);    
  d_Ap.num_values = d_A.num_rows; 
  d_b.values = clCreateBuffer(context, CL_MEM_READ_WRITE, d_A.num_rows * sizeof(double),
                            NULL, &cl_status);    
  d_b.num_values = d_A.num_rows;        
  d_r.values = clCreateBuffer(context, CL_MEM_READ_WRITE, d_A.num_rows * sizeof(double),
                            NULL, &cl_status);    
  d_r.num_values = d_A.num_rows;                
  d_x.values = clCreateBuffer(context, CL_MEM_READ_WRITE, d_A.num_rows * sizeof(double),
                            NULL, &cl_status);    
  d_x.num_values = d_A.num_rows;                
  double one = 1.0;
  double zero = 0.0;
  double minus = -1.0; 

  d_alpha.value = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double),
                               nullptr, &cl_status);
  d_beta.value = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double),
                               nullptr, &cl_status);    
  d_minus.value = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double),
                               nullptr, &cl_status);                                                          

  // alpha = 1;
  double* halpha = (double*) clEnqueueMapBuffer(command_queue, d_alpha.value, CL_TRUE, CL_MAP_WRITE,
                                              0, sizeof(double), 0, nullptr, nullptr, &cl_status);
  *halpha = one;

  cl_status = clEnqueueUnmapMemObject(command_queue, d_alpha.value, halpha,
                                      0, nullptr, nullptr);

  //beta = 0;
  double* hbeta = (double*) clEnqueueMapBuffer(command_queue, d_beta.value, CL_TRUE, CL_MAP_WRITE,
                                             0, sizeof(double), 0, nullptr, nullptr, &cl_status);
  *hbeta = zero;

  cl_status = clEnqueueUnmapMemObject(command_queue, d_beta.value, hbeta,
                                      0, nullptr, nullptr);         
  double* hminus = (double*) clEnqueueMapBuffer(command_queue, d_minus.value, CL_TRUE, CL_MAP_WRITE,
                                             0, sizeof(double), 0, nullptr, nullptr, &cl_status);
  *hminus = minus;

  cl_status = clEnqueueUnmapMemObject(command_queue, d_minus.value, hminus,
                                      0, nullptr, nullptr);                                               
  d_normr.value = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double),
                            NULL, &cl_status);                                                   
  
  // Create the input and output arrays in device memory for our calculation
    d_rtz.value = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof( double ), &zero, NULL);
    d_oldrtz.value = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof( double ), &zero, NULL);
    d_pAp.value = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof( double ), NULL, NULL);
    d_Beta.value = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof( double ), NULL, NULL);
    d_Alpha.value = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof( double ), NULL, NULL);
    d_minusAlpha.value = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof( double ), NULL, NULL);
    
     // Set the arguments to our compute kernel
    err  = clSetKernelArg(kernel1, 0, sizeof(cl_mem), &d_rtz.value);
    err |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &d_oldrtz.value);      
    
    // Set the arguments to our compute kernel
    err  = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &d_rtz.value);
    err |= clSetKernelArg(kernel2, 1, sizeof(cl_mem), &d_oldrtz.value);
    err |= clSetKernelArg(kernel2, 2, sizeof(cl_mem), &d_Beta.value);      
    
    // Set the arguments to our compute kernel
    err  = clSetKernelArg(kernel3, 0, sizeof(cl_mem), &d_rtz.value);
    err |= clSetKernelArg(kernel3, 1, sizeof(cl_mem), &d_pAp.value);
    err |= clSetKernelArg(kernel3, 2, sizeof(cl_mem), &d_Alpha.value);
    err |= clSetKernelArg(kernel3, 3, sizeof(cl_mem), &d_minusAlpha.value);      
  
  return 0;
}


int CG(const SparseMatrix & A, SparseMatrix &A_ref, CGData & data, const Vector & b, Vector & x,
    const int max_iter, const double tolerance, int & niters, double & normr, double & normr0,
    double * times, bool doPreconditioning) {
//std::cerr << "CG \n";    
  double t_begin = mytimer();  // Start timing right away
  normr = 0.0;
  double rtz = 0.0, oldrtz = 0.0, alpha = 0.0, beta = 0.0, pAp = 0.0, minusAlpha = 0.0;
  double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0, t5 = 0.0;
//#ifndef HPCG_NO_MPI
//  double t6 = 0.0;
//#endif
  size_t globalSize = 64;
  static int call_count;
  
  if (!call_count) 
  {  
    opencl_setup();
    clsparse_setup(A);
    ++call_count; 
  } 
 
  local_int_t nrow = A.localNumberOfRows;
  Vector & r = data.r; // Residual vector
  Vector & z = data.z; // Preconditioned residual vector
  Vector & p = data.p; // Direction vector (in MPI mode ncol>=nrow)
  Vector & Ap = data.Ap;

  if (!doPreconditioning && A.geom->rank==0) HPCG_fout << "WARNING: PERFORMING UNPRECONDITIONED ITERATIONS" << std::endl;

#ifdef HPCG_DEBUG
  int print_freq = 1;
  if (print_freq>50) print_freq=50;
  if (print_freq<1)  print_freq=1;
#endif
  // p is of length ncols, copy x to p for sparse MV operation
  CopyVector(x, p);
  /* Create GPU buffers */
  int k = 0;
  for(int i = 0; i < A.totalNumberOfRows; i++) 
  {
     for(int j = 0; j < A.nonzerosInRow[i]; j++)
     {
       val[k] = A.matrixValues[i][j];
       k++;
     }
  }                            
  
  clEnqueueWriteBuffer(command_queue, d_A.values, CL_TRUE, 0,
                              d_A.num_nonzeros * sizeof( double ), val, 0, NULL, NULL ); 
  clEnqueueWriteBuffer(command_queue, d_p.values, CL_TRUE, 0,
                              d_A.num_rows * sizeof( double ), p.values, 0, NULL, NULL );           
  clEnqueueWriteBuffer(command_queue, d_b.values, CL_TRUE, 0,
                              d_A.num_rows * sizeof( double ), b.values, 0, NULL, NULL );                                                                                                          
  
  TICK(); ComputeSPMV(d_A, d_p, d_Ap); TOCK(t3); // Ap = A*p
  TICK(); ComputeWAXPBY(d_alpha, d_b, d_minus, d_Ap, d_r);  TOCK(t2); // r = b - Ax (x stored in p)
  TICK(); ComputeDotProduct(d_r, d_r, d_normr, t4); TOCK(t1);
  
  
  clEnqueueReadBuffer(command_queue, d_r.values, CL_TRUE, 0,
                              d_r.num_values * sizeof(double), r.values, 0, NULL, NULL );   
  clEnqueueReadBuffer(command_queue, d_normr.value, CL_TRUE, 0,
                              sizeof(double), &normr, 0, NULL, NULL );  
                              
  normr = sqrt(normr);                                                                                          

#ifdef HPCG_DEBUG
  if (A.geom->rank==0) HPCG_fout << "Initial Residual = "<< normr << std::endl;
#endif

  // Record initial residual for convergence testing
  normr0 = normr;

  // Start iterations

  for (int k=1; k<=max_iter && normr/normr0 > tolerance; k++ ) {
    TICK();
    if (doPreconditioning)
      ComputeMG(A, A_ref, r, z); // Apply preconditioner
    else
      CopyVector (r, z); // copy r to z (no preconditioning)
    TOCK(t5); // Preconditioner apply time

    clEnqueueWriteBuffer(command_queue, d_b.values, CL_TRUE, 0,
                              d_A.num_rows * sizeof( double ), z.values, 0, NULL, NULL );     

    if (k == 1) {
      TICK(); ComputeWAXPBY(d_alpha, d_b, d_beta, d_b, d_p); TOCK(t2); // Copy Mr to p
      TICK(); ComputeDotProduct(d_r, d_b, d_rtz, t4); TOCK(t1); // rtz = r'*z
      
    } else {
      //oldrtz = rtz;
 
      // Execute the kernel over the entire range of the data set  
      err = clEnqueueNDRangeKernel(command_queue, kernel1, 1, NULL, &globalSize, NULL,
                                                                0, NULL, NULL);
      
      TICK(); ComputeDotProduct(d_r, d_b, d_rtz, t4); TOCK(t1); // rtz = r'*z
      
      //beta = rtz/oldrtz;
      
      // Execute the kernel over the entire range of the data set  
      err = clEnqueueNDRangeKernel(command_queue, kernel2, 1, NULL, &globalSize, NULL,
                                                                0, NULL, NULL);
      
      TICK(); ComputeWAXPBY(d_alpha, d_b, d_Beta, d_p, d_p);  TOCK(t2); // p = beta*p + z
    }

    TICK(); ComputeSPMV(d_A, d_p, d_Ap); TOCK(t3); // Ap = A*p
    TICK(); ComputeDotProduct(d_p, d_Ap, d_pAp, t4); TOCK(t1); // alpha = p'*Ap
    
    //alpha = rtz/pAp;                     
    
    // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(command_queue, kernel3, 1, NULL, &globalSize, NULL,
                                                              0, NULL, NULL);                                                                 
    
    TICK(); ComputeWAXPBY(d_alpha, d_x, d_Alpha, d_p, d_x);// x = x + alpha*p    
            ComputeWAXPBY(d_alpha, d_r, d_minusAlpha, d_Ap, d_r);  TOCK(t2);// r = r - alpha*Ap
    TICK(); ComputeDotProduct(d_r, d_r, d_normr, t4); TOCK(t1);
    
    clEnqueueReadBuffer(command_queue, d_normr.value, CL_TRUE, 0,
                              sizeof(double), &normr, 0, NULL, NULL );  
    
    normr = sqrt(normr);
#ifdef HPCG_DEBUG
    if (A.geom->rank==0 && (k%print_freq == 0 || k == max_iter))
      HPCG_fout << "Iteration = "<< k << "   Scaled Residual = "<< normr/normr0 << std::endl;
#endif
    niters = k;
    
    clEnqueueReadBuffer(command_queue, d_r.values, CL_TRUE, 0,
                              d_r.num_values * sizeof(double), r.values, 0, NULL, NULL );  
    clEnqueueReadBuffer(command_queue, d_b.values, CL_TRUE, 0,
                              d_b.num_values * sizeof(double), z.values, 0, NULL, NULL );  
                                  
    
  }

    clEnqueueReadBuffer(command_queue, d_p.values, CL_TRUE, 0,
                              d_p.num_values * sizeof(double), p.values, 0, NULL, NULL );  
    clEnqueueReadBuffer(command_queue, d_Ap.values, CL_TRUE, 0,
                              d_Ap.num_values * sizeof(double), Ap.values, 0, NULL, NULL );     
                                                                                                                    
    clEnqueueReadBuffer(command_queue, d_x.values, CL_TRUE, 0,
                              d_x.num_values * sizeof(double), x.values, 0, NULL, NULL );  
   
  
  // Store times
  times[1] += t1; // dot-product time
  times[2] += t2; // WAXPBY time
  times[3] += t3; // SPMV time
  times[4] += t4; // AllReduce time
  times[5] += t5; // preconditioner apply time
//#ifndef HPCG_NO_MPI
//  times[6] += t6; // exchange halo time
//#endif
  times[0] += mytimer() - t_begin;  // Total time. All done...
  return 0;
}
