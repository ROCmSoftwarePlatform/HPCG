
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
#include <iostream>

#include <CL/cl.hpp>
#include "clSPARSE.h"
#include "clSPARSE-error.h"

using namespace std;
int row;

int call_count;
float *val, *qt_matrixValues;
int *col, *rowOff, *nnzInRow, *Count;
local_int_t *qt_mtxIndl, *qt_rowOffset, *q_mtxIndl, *q_rowOffset;

cl_platform_id *platform;
cl_context context;
cl_device_id *device;
cl_command_queue command_queue;
cl_int err;
cl_int cl_status;

clsparseCreateResult createResult;
clsparseStatus status;
clsparseCsrMatrix d_A, d_Q, d_Qt, d_A_ref;

/*!
  Optimizes the data structures used for CG iteration to increase the
  performance of the benchmark version of the preconditioned CG algorithm.

  @param[inout] A      The known system matrix, also contains the MG hierarchy in attributes Ac and mgData.
  @param[inout] A_ref  The reference which needs to be reordered accordingly.
  @param[inout] colors The vecotor to store the index of the reordering order. 

  @return returns 0 upon success and non-zero otherwise

  @see GenerateGeometry
  @see GenerateProblem
*/

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
  
  return 0;
}

int clsparse_setup()
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
  clsparseInitCsrMatrix(&d_Q);
  clsparseInitCsrMatrix(&d_Qt);
  clsparseInitCsrMatrix(&d_A_ref); 
  
  return 0;
}

int mem_alloc(SparseMatrix A)
{
  val = new float[A.totalNumberOfNonzeros];
  col = new int[A.totalNumberOfNonzeros];
  rowOff = new int[A.localNumberOfRows + 1];  
  
  nnzInRow = new int[A.localNumberOfRows]();
  Count = new int[A.localNumberOfRows]();
  
  qt_matrixValues = new float[A.localNumberOfRows];
  qt_mtxIndl = new local_int_t[A.localNumberOfRows];
  qt_rowOffset = new local_int_t[A.localNumberOfRows + 1];
  q_mtxIndl = new local_int_t[A.localNumberOfRows];
  q_rowOffset = new local_int_t[A.localNumberOfRows + 1];
  
  d_A.values = clCreateBuffer( context, CL_MEM_READ_ONLY,
                               d_A.num_nonzeros * sizeof( float ), NULL, &cl_status );
  
  d_A.col_indices = clCreateBuffer( context, CL_MEM_READ_ONLY,
                                   d_A.num_nonzeros * sizeof( clsparseIdx_t ),NULL, &cl_status );

  d_A.row_pointer = clCreateBuffer( context, CL_MEM_READ_ONLY,
                                   ( d_A.num_rows + 1 ) * sizeof( clsparseIdx_t ),NULL, &cl_status );  
     
  
  d_Qt.values = clCreateBuffer( context, CL_MEM_READ_WRITE,
                               d_Qt.num_nonzeros * sizeof( float ), NULL, &cl_status );
  
  d_Qt.col_indices = clCreateBuffer( context, CL_MEM_READ_WRITE,
                                   d_Qt.num_nonzeros * sizeof( clsparseIdx_t ), NULL, &cl_status );

  d_Qt.row_pointer = clCreateBuffer( context, CL_MEM_READ_WRITE,
                                   ( d_Qt.num_rows + 1 ) * sizeof( clsparseIdx_t ), NULL, &cl_status );  
                                   
  
  d_Q.values = clCreateBuffer( context, CL_MEM_READ_ONLY,
                               d_Q.num_nonzeros * sizeof( float ), NULL, &cl_status );
  
  d_Q.col_indices = clCreateBuffer( context, CL_MEM_READ_ONLY,
                                   d_Q.num_nonzeros * sizeof( clsparseIdx_t ), NULL, &cl_status );

  d_Q.row_pointer = clCreateBuffer( context, CL_MEM_READ_ONLY,
                                   ( d_Q.num_rows + 1 ) * sizeof( clsparseIdx_t ), NULL, &cl_status ); 
                                
                                   
  d_A_ref.values = clCreateBuffer( context, CL_MEM_READ_WRITE,
                               d_A_ref.num_nonzeros * sizeof( float ), NULL, &cl_status );
  
  d_A_ref.col_indices = clCreateBuffer( context, CL_MEM_READ_WRITE,
                                   d_A_ref.num_nonzeros * sizeof( clsparseIdx_t ), NULL, &cl_status );

  d_A_ref.row_pointer = clCreateBuffer( context, CL_MEM_READ_WRITE,
                                   ( d_A_ref.num_rows + 1 ) * sizeof( clsparseIdx_t ), NULL, &cl_status );       
  
  return 0;
}

// free the reference matrix
void free_refmatrix_m(SparseMatrix &A)
{
  for(int i =0 ; i < A.localNumberOfRows; i++)
  {
    delete [] A.matrixValues[i];
    delete [] A.mtxIndL[i];
  }
}

// luby's graph coloring algorthim - nvidia's approach

void lubys_graph_coloring (int c,int *row_offset,int *col_index, std::vector<local_int_t> &colors,int *random,std::vector<local_int_t> &temp)
{
    for(int i=0;i<row;i++)
    {
       int flag = 1;
       if(colors[i] != -1)
          continue;
       int ir = random[i];
       for(int k=row_offset[i];k<row_offset[i+1];k++)
       {
          int j = col_index[k];
          int jc = colors[j];
          if (((jc != -1) && (jc != c)) || (i == j)) 
            continue;
          int jr = random[j];
          if(ir <= jr)
            flag = 0;
       }
       if(flag)
       {
        colors[i] = c;
       }
        
    }
}


int OptimizeProblem(const SparseMatrix & A,SparseMatrix & A_ref) {

  const local_int_t nrow = A.localNumberOfRows;
  row = nrow;
  int *random = new int [nrow];
  std::vector<local_int_t> temp(nrow, -1);
  int *row_offset,*col_index;
  col_index = new int [nrow * 27];
  row_offset = new int [(nrow + 1)];

 // Initialize local Color array and random array using rand functions.
  srand(1459166450);
  for (int i = 0; i < nrow; i++)
  {
      random[i] = rand(); 
  }
  row_offset[0] = 0;


  int k = 0;
  // Save the mtxIndL in a single dimensional array for column index reference.
  for(int i = 0; i < nrow; i++)
  {
    for(int j = 0; j < A.nonzerosInRow[i]; j++)
    {
        col_index[k] = A.mtxIndL[i][j];
        k++;
    }
  }
  
 
  k = 0;
  // Calculate the row offset.
  int ridx = 1;
  int sum = 0;
  for(int i = 0; i < nrow; i++)
  {
     sum =  sum + A.nonzerosInRow[i];
     row_offset[ridx] = sum;
     ridx++;
  }

  // Call luby's graph coloring algorithm. 
  int c = 0;
  for( c = 0; c < nrow; c++)
  {

      lubys_graph_coloring(c,row_offset,col_index,A_ref.colors,random,temp);
      int left = std::count(A_ref.colors.begin(), A_ref.colors.end(), -1);
        if(left == 0)
          break;
  }
  
  // Calculate number of rows with the same color and save it in counter vector.
  std::vector<local_int_t> counters(c+2);
  A_ref.counters.resize(c+2);
  std::fill(counters.begin(), counters.end(), 0);
  for (local_int_t i = 0; i < nrow; ++i)
  {
    counters[A_ref.colors[i]]++;
  }

  // Calculate color offset using counter vector. 
  local_int_t old = 0 , old0 = 0;
  for (int i = 1; i <= c + 1; ++i) {
    old0 = counters[i];
    counters[i] = counters[i-1] + old;
    old = old0;
  }
  counters[0] = 0;


  for (int i = 0; i <= c + 1; ++i) {
    A_ref.counters[i] = counters[i];
  }
  
  // translate `colors' into a permutation.
  std::vector<local_int_t> colors(nrow);
  int k1 = 0;
  for(int i = 0; i < c+1; i++)
  {
      for(int j = 0; j < nrow; j++)
      {
          if(A_ref.colors[j] == i)
          {
              colors[k1] = j;
              k1++;
          }
      }
  }
  for(int i = 0; i < nrow; i++)
    A_ref.colors[i] = colors[i];

  // declare and allocate qt sparse matrix to be used in CSR format
  float * qt_matrixValues = new float[nrow];
  local_int_t * qt_mtxIndl = new local_int_t[nrow];
  local_int_t * qt_rowOffset = new local_int_t[nrow + 1];

  // declare and allocate q sparse matrix to be used in CSR format
  local_int_t * q_mtxIndl = new local_int_t[nrow];
  local_int_t * q_rowOffset = new local_int_t[nrow + 1];

  // Generating qt based on the reordering order
  int indx = 0;
  qt_rowOffset[0] = 0;
  for(int i = 0; i < nrow; i++)
  {
    for(int j = 0; j < nrow; j++)
    {
      if(A_ref.colors[i] == j)
      {
        qt_matrixValues[indx] = 1;
        qt_mtxIndl[indx++] = j;
        qt_rowOffset[indx] = qt_rowOffset[indx - 1] + 1;
        break;
      }
    }
  }




/*
  
  // take transpose for qt such that " q = transpose(qt)"

  // convert A matrix to CSR format

  // Perform the multiplication A_ref = qt * A * q

  // convert A_ref to ELL format and save matrix values to A_ref.matrixValues 
     and col indices to A_ref.mtxIndl

*/

  
  /* Opencl and clSPARSE setup */
  if (!call_count)
  {
    opencl_setup();
    clsparse_setup();
  }

  d_A.num_nonzeros = A.totalNumberOfNonzeros;
  d_A.num_rows = A.localNumberOfRows;
  d_A.num_cols = A.localNumberOfColumns;  
  
  d_Qt.num_nonzeros = nrow;
  d_Qt.num_rows = A.localNumberOfRows;
  d_Qt.num_cols = A.localNumberOfColumns; 
  
  d_Q.num_nonzeros = nrow;
  d_Q.num_rows = A.localNumberOfRows;
  d_Q.num_cols = A.localNumberOfColumns;     
  
  d_A_ref.num_nonzeros = A_ref.totalNumberOfNonzeros;
  d_A_ref.num_rows = A_ref.localNumberOfRows;
  d_A_ref.num_cols = A_ref.localNumberOfColumns;     
  
  if (call_count)
  {
    for(int i = 0; i < nrow; i++)
      nnzInRow[i] = Count[i] = 0;
  }
  
  if (!call_count)
  {
    mem_alloc(A);
    ++call_count;
  }


  /* Transpose */  
 
  for(int i = 0; i < nrow; i++)
    nnzInRow[qt_mtxIndl[i]] += 1;
    
  q_rowOffset[0] = 0;
  for(int i = 1; i <= nrow; i++)
     q_rowOffset[i] =  q_rowOffset[i - 1] + nnzInRow[i - 1];
    
  for(int i = 0; i < nrow; i++)
  {
    q_mtxIndl[ q_rowOffset[qt_mtxIndl[i]] + Count[qt_mtxIndl[i]]] = i;
    Count[qt_mtxIndl[i]]++;
  }
  
  /* CSR Matrix */  
  /* A */  
  k = 0;
  rowOff[0] = 0;
  for(int i = 1; i <= A.totalNumberOfRows; i++)
    rowOff[i] = rowOff[i - 1] + A.nonzerosInRow[i - 1];
    
  for(int i = 0; i < A.totalNumberOfRows; i++) 
  {
     for(int j = 0; j < A.nonzerosInRow[i]; j++)
     {
       col[k] = A.mtxIndL[i][j];
       val[k] = (float)A.matrixValues[i][j];
       k++;
     }
  }              
               
  clEnqueueWriteBuffer(command_queue, d_A.values, CL_TRUE, 0,
                              d_A.num_nonzeros * sizeof( float ), val, 0, NULL, NULL ); 
  clEnqueueWriteBuffer(command_queue, d_A.col_indices, CL_TRUE, 0,
                              d_A.num_nonzeros * sizeof( clsparseIdx_t ), col, 0, NULL, NULL ); 
  clEnqueueWriteBuffer(command_queue, d_A.row_pointer, CL_TRUE, 0,
                              (d_A.num_rows + 1) * sizeof( clsparseIdx_t ), rowOff, 0, NULL, NULL );  
                              
  /* Qt */
  clEnqueueWriteBuffer(command_queue, d_Qt.values, CL_TRUE, 0,
                              d_Qt.num_nonzeros * sizeof( float ), qt_matrixValues, 0, NULL, NULL ); 
  clEnqueueWriteBuffer(command_queue, d_Qt.col_indices, CL_TRUE, 0,
                              d_Qt.num_nonzeros * sizeof( clsparseIdx_t ), qt_mtxIndl, 0, NULL, NULL ); 
  clEnqueueWriteBuffer(command_queue, d_Qt.row_pointer, CL_TRUE, 0,
                              (d_Qt.num_rows + 1) * sizeof( clsparseIdx_t ), qt_rowOffset, 0, NULL, NULL );                                                                                                                    
            
  /* Q */
  clEnqueueWriteBuffer(command_queue, d_Q.values, CL_TRUE, 0,
                              d_Q.num_nonzeros * sizeof( float ), qt_matrixValues, 0, NULL, NULL ); 
  clEnqueueWriteBuffer(command_queue, d_Q.col_indices, CL_TRUE, 0,
                              d_Q.num_nonzeros * sizeof( clsparseIdx_t ), q_mtxIndl, 0, NULL, NULL ); 
  clEnqueueWriteBuffer(command_queue, d_Q.row_pointer, CL_TRUE, 0,
                              (d_Q.num_rows + 1) * sizeof( clsparseIdx_t ), q_rowOffset, 0, NULL, NULL );             
                                
  /* clSPARSE matrix-matrix multiplication */
  clsparseScsrSpGemm(&d_Qt, &d_A, &d_A_ref, createResult.control);
  clsparseScsrSpGemm(&d_A_ref, &d_Q, &d_A_ref, createResult.control);

  /* Copy back the result */ 
  clEnqueueReadBuffer(command_queue, d_A_ref.values, CL_TRUE, 0,
                              d_A_ref.num_nonzeros * sizeof(float), val, 0, NULL, NULL );
                                                                                                                                     
  clEnqueueReadBuffer(command_queue, d_A_ref.col_indices, CL_TRUE, 0,
                              d_A_ref.num_nonzeros * sizeof(clsparseIdx_t ), col, 0, NULL, NULL );    
                                                               
  
  // Rearranges the reference matrix according to the coloring index.
  #ifndef HPCG_NO_OPENMP
    #pragma omp parallel for
  #endif
  for(int i = 0; i < nrow; i++)
    A_ref.nonzerosInRow[i] = A.nonzerosInRow[A_ref.colors[i]];
	   
  /* Copy back A_ref values and col indices */
  k = 0;
  for(int i = 0; i < A_ref.totalNumberOfRows; i++) 
  {
     for(int j = 0; j < A_ref.nonzerosInRow[i]; j++)
     {
       A_ref.matrixValues[i][j] = (double)val[k];
       A_ref.mtxIndL[i][j] = col[k];
       if (i == A_ref.mtxIndL[i][j])
         A_ref.matrixDiagonal[i] = &A_ref.matrixValues[i][j];
       k++;
     }
  }       
  
  delete [] row_offset;
  delete [] col_index;
  delete [] random;

  return 0;
}

// Helper function (see OptimizeProblem.hpp for details)
double OptimizeProblemMemoryUse(const SparseMatrix & A) {

  return 0.0;

}
