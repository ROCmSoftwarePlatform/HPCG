
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
using namespace std;
int row ;

cl_platform_id *platform = NULL;
cl_context context = 0;
cl_device_id *device = NULL;
cl_command_queue command_queue = 0;
cl_program program = 0;
cl_kernel  kernel = 0;
cl_int err = CL_SUCCESS;
cl_int cl_status = CL_SUCCESS;

cl_mem  clRow_offset = NULL;
cl_mem  clCol_index = NULL;
cl_mem  clColors = NULL;
cl_mem  clRandom = NULL;
cl_mem  clTemp = NULL;


const char *kernel_name = "lubys_graph";
const char *Lubys_graph_kernel = "                                                    \n\
  __kernel void lubys_graph(int c, __global int *row_offset, __global int *col_index, \n\
                            __global int *Colors, __global int *random,               \n\
                            __global int *temp)                                       \n\
  {                                                                                   \n\
    int x = get_global_id(0);                                                         \n\
    int flag = 1;                                                                     \n\
    if(temp[x] == -1)                                                                 \n\
    {                                                                                 \n\
      int ir = random[x];                                                             \n\
      for(int k = row_offset[x]; k < row_offset[x + 1]; k++)                          \n\
      {                                                                               \n\
        int j = col_index[k];                                                         \n\
        int jc = Colors[j];                                                           \n\
        if (((jc != -1) && (jc != c)) || (x == j))                                    \n\
        {                                                                             \n\
          continue;                                                                   \n\
        }                                                                             \n\
        int jr = random[j];                                                           \n\
        if(ir <= jr)                                                                  \n\
        {                                                                             \n\
          flag = 0;                                                                   \n\
        }                                                                             \n\
      }                                                                               \n\
      if(flag)                                                                        \n\
      {                                                                               \n\
        temp[x] = c;                                                                  \n\
      }                                                                               \n\
    }                                                                                 \n\
  }                                                                                   \n\
  ";

void InitOpenCL(void)
{
  if (NULL != platform)
  {
    return;
  }

  cl_uint ret_num_of_platforms = 0;
  cl_int err = clGetPlatformIDs(0, NULL, &ret_num_of_platforms);
  assert(err == CL_SUCCESS && "clGetPlatformIDs\n");
  if (err != CL_SUCCESS || 0 == ret_num_of_platforms)
  {
    std::cout << "ERROR: Getting platforms!" << std::endl;
    return;
  }

  platform = (cl_platform_id*)malloc(ret_num_of_platforms * sizeof(cl_platform_id));
  err = clGetPlatformIDs(ret_num_of_platforms, platform, 0);
  assert(err == CL_SUCCESS && "clGetPlatformIDs\n");

  cl_uint ret_num_of_devices = 0;
  err = clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_GPU, 0, NULL, &ret_num_of_devices);

  assert(err == CL_SUCCESS && "clGetDeviceIds failed\n");

  device = (cl_device_id*)malloc(ret_num_of_devices * sizeof(cl_device_id));
  err = clGetDeviceIDs(platform[0],CL_DEVICE_TYPE_GPU , ret_num_of_devices, device, 0);
  assert(err == CL_SUCCESS && "clGetDeviceIds failed\n");

  cl_context_properties property[] = {CL_CONTEXT_PLATFORM,
                                      (cl_context_properties)(platform[0]),
                                      0};

  context = clCreateContext(property, ret_num_of_devices, &device[0], NULL, NULL, &err);
  assert(err == CL_SUCCESS && "clCreateContext failed\n");

  command_queue = clCreateCommandQueue(context, device[0], 0, &err);
  assert(err == CL_SUCCESS && "clCreateCommandQueue failed \n");

  return;
}

void InitCLMem(int row_size, int *row_offset, int *col_index, int *random)
{
  if (NULL == clRow_offset)
  {
    clRow_offset = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  (row_size + 1) * sizeof(int), row_offset, &cl_status);
    if (CL_SUCCESS != cl_status || NULL == clRow_offset)
    {
      std::cout << "clRow_offset allocation failed. status: " << cl_status << std::endl;
      return;
    }
  }

  if (NULL == clCol_index)
  {
    clCol_index = clCreateBuffer(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR,
                                 (row_size * 27) * sizeof(int), col_index, &cl_status);
    if (CL_SUCCESS != cl_status || NULL == clCol_index)
    {
      std::cout << "clCol_index allocation failed. status: " << cl_status << std::endl;
      return;
    }
  }

  if (NULL == clColors)
  {
    clColors = clCreateBuffer(context, CL_MEM_READ_WRITE, row_size * sizeof(int),
                              NULL, &cl_status);
    if (CL_SUCCESS != cl_status || NULL == clColors)
    {
      std::cout << "clColors allocation failed. status: " << cl_status << std::endl;
      return;
    }
  }

  if (NULL == clRandom)
  {
    clRandom = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              row_size * sizeof(int), random, &cl_status);
    if (CL_SUCCESS != cl_status || NULL == clRandom)
    {
      std::cout << "clRandom allocation failed. status: " << cl_status << std::endl;
      return;
    }
  }

  if (NULL == clTemp)
  {
    clTemp = clCreateBuffer(context, CL_MEM_READ_WRITE, row_size * sizeof(int),
                          NULL, &cl_status);
    if (CL_SUCCESS != cl_status || NULL == clTemp)
    {
      std::cout << "clTemp allocation failed. status: " << cl_status << std::endl;
      return;
    }
  }

  return;
}

void ReleaseCLMem(void)
{
  cl_status = CL_SUCCESS;

  if (NULL != clRow_offset)
  {
    cl_status |= clReleaseMemObject(clRow_offset);
    clRow_offset = NULL;
  }

  if (NULL != clCol_index)
  {
    cl_status |= clReleaseMemObject(clCol_index);
    clCol_index = NULL;
  }

  if (NULL != clColors)
  {
    cl_status |= clReleaseMemObject(clColors);
    clColors = NULL;
  }

  if (NULL != clRandom)
  {
    cl_status |= clReleaseMemObject(clRandom);
    clRandom = NULL;
  }

  if (NULL != clTemp)
  {
    cl_status |= clReleaseMemObject(clTemp);
    clTemp = NULL;
  }

  if (CL_SUCCESS != cl_status)
  {
    std::cout << "clReleaseMemObject failed." <<std::endl;
  }

  return;
}

void ExecuteKernel(int c, int row_size, std::vector<local_int_t> &iColors)
{
  int *colors = NULL;

  assert(row_size == iColors.size());

  colors = (int *)malloc(sizeof(int) * row_size);

  if (NULL == colors)
  {
    return;
  }

  //std::copy(iColors.begin(), iColors.end(), colors);
  for (int i = 0; i < iColors.size(); i++)
  {
    colors[i] = iColors[i];
  }

  cl_status = clEnqueueWriteBuffer(command_queue, clColors, CL_TRUE, 0, row_size * sizeof(int),
                                   colors, 0, NULL, NULL);

  cl_status |= clEnqueueWriteBuffer(command_queue, clTemp, CL_TRUE, 0, row_size * sizeof(int),
                                    colors, 0, NULL, NULL);
  if (CL_SUCCESS != cl_status)
  {
    std::cout << "DMA failed, status:" << cl_status << std::endl;
  }

  size_t sourceSize[] = { strlen(Lubys_graph_kernel) };
  if (!program)
  {
    program = clCreateProgramWithSource(context, 1, &Lubys_graph_kernel,
                                        sourceSize, &cl_status);
    if (CL_SUCCESS != cl_status)
    {
      std::cout << "create program failed. status:" << cl_status << std::endl;
      return;
    }

    cl_status = clBuildProgram(program, 1, &device[0], NULL, NULL, NULL);
    if (CL_SUCCESS != cl_status)
    {
      std::cout << "clBuild failed. status:" << cl_status << std::endl;
      char tbuf[0x10000];
      clGetProgramBuildInfo(program, device[0], CL_PROGRAM_BUILD_LOG, 0x10000, tbuf, NULL);
      std::cout << tbuf <<std::endl;
      return;
    }
  }

  if (!kernel)
  {
    kernel = clCreateKernel(program, kernel_name, &cl_status);
    if (CL_SUCCESS != cl_status)
    {
      std::cout << "create kernel failed. status:" << cl_status << std::endl;
      return;
    }
  }

  cl_int c1 = c;
  clSetKernelArg(kernel, 0, sizeof(cl_int), (void *)&c1);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&clRow_offset);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&clCol_index);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&clColors);
  clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&clRandom);
  clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&clTemp);

  size_t global_size[] = {row_size};
  cl_status = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                                     global_size, NULL, 0, NULL, NULL);
  if (CL_SUCCESS != cl_status)
  {
    std::cout << "NDRange failed. status:" << cl_status << std::endl;
    return;
  }

  clFinish(command_queue);

  cl_status |= clEnqueueReadBuffer(command_queue, clTemp, CL_TRUE, 0,
                                   row_size * sizeof(int), colors, 0, NULL, NULL);

  //std::copy(colors, colors + row_size, iColors.begin());
  for (int i = 0; i < iColors.size(); i++)
  {
    iColors[i] = colors[i];
  }

  free(colors);

  return;
}

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

// free the reference matrix
void free_refmatrix_m(SparseMatrix &A)
{
  for(int i =0 ; i < A.localNumberOfRows; i++)
  {
    delete [] A.matrixValues[i];
    delete [] A.mtxIndL[i];
  }
}

// Hash fucntion

int hash_function(int index , int nnz)
{
   int i;
   unsigned int uiHash = 0U;
   uiHash =  ( index * (unsigned int) row ) + nnz;
   return uiHash;
}


// Copy source to destination

void copy_value( std::vector<local_int_t> &dest,  std::vector<local_int_t> &source)
{
    for (int i = 0; i <row; ++i)
    {
      dest[i] = source[i];
    }
}

// luby's graph coloring algorthim - nvidia's approach

void lubys_graph_coloring (int c,int *row_offset,int *col_index, std::vector<local_int_t> &colors,int *random,std::vector<local_int_t> &temp)
{

    copy_value(temp,colors);
    for(int i=0;i<row;i++)
    {
       int flag = 1;
       if(temp[i] != -1)
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
        temp[i] = c;
       }

    }
    copy_value(colors,temp);
}


int OptimizeProblem(const SparseMatrix & A,SparseMatrix & A_ref) {

  const local_int_t nrow = A.localNumberOfRows;
  row = nrow;
  int *random = new int [nrow];
  std::vector<local_int_t> temp(nrow, -1);
  int *row_offset,*col_index;
  col_index = new int [nrow * 27];
  row_offset = new int [(nrow + 1)];

 // Initialize local Color array and random array using hash functions.
  for (int i = 0; i < nrow; i++)
  {
      //Colors[i] = -1;
      random[i] = hash_function(i,A.nonzerosInRow[i]);
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

  InitOpenCL();

  InitCLMem(nrow, row_offset, col_index, random);

  //std::cout << "size: " << nrow << std::endl;

  // Call luby's graph coloring algorithm.
  int c = 0;
  for( c = 0; c < nrow; c++)
  {

      //lubys_graph_coloring(c,row_offset,col_index,A_ref.colors,random,temp);
      ExecuteKernel(c, nrow, A_ref.colors);
      //std::cout << "c : " << c << std::endl;
      int left = std::count(A_ref.colors.begin(), A_ref.colors.end(), -1);
        if(left == 0)
          break;
  }

  ReleaseCLMem();

  // Calculate number of rows with the same color and save it in counter vector.
  std::vector<local_int_t> counters(c+1);
  A_ref.counters.resize(c+5);
  std::fill(counters.begin(), counters.end(), 0);
  for (local_int_t i = 0; i < nrow; ++i)
  {
    counters[A_ref.colors[i]]++;
  }

  // Calculate color offset using counter vector.
  local_int_t old = 0 , old0 = 0;
  for (int i = 1; i <= c; ++i) {
    old0 = counters[i];
    counters[i] = counters[i-1] + old;
    old = old0;
  }
  counters[0] = 0;


  for (int i = 0; i <= c; ++i) {
    A_ref.counters[i] = counters[i];
  }

  // translate `colors' into a permutation.
  for (local_int_t i=0; i<nrow; ++i) // for each color `c'
  {
    A_ref.colors[i] = counters[A_ref.colors[i]]++;
  }

  // Rearranges the reference matrix according to the coloring index.
  for(int i = 0; i < nrow; i++)
  {

	   const int currentNumberOfNonzeros = A.nonzerosInRow[A_ref.colors[i]];
     A_ref.nonzerosInRow[i] = A.nonzerosInRow[A_ref.colors[i]];
	   const double * const currentValues = A.matrixValues[A_ref.colors[i]];
	   const local_int_t * const currentColIndices = A.mtxIndL[A_ref.colors[i]];

	   double * diagonalValue = A.matrixDiagonal[A_ref.colors[i]];
	   A_ref.matrixDiagonal[i] = diagonalValue;

		//rearrange the elements in the row
     int col_indx = 0;
     for(int k = 0; k < nrow; k++)
     {
	      for(int j = 0; j < currentNumberOfNonzeros; j++)
	      {
		       if(A_ref.colors[k] == currentColIndices[j])
		       {
			        A_ref.matrixValues[i][col_indx] = currentValues[j];
			        A_ref.mtxIndL[i][col_indx++] = k;
			        break;
   	       }
        }
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
