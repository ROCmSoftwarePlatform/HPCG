
#ifndef OCL_HPP
#define OCL_HPP


#include "SparseMatrix.hpp"
#include <CL/cl.hpp>
#include "clSPARSE.h"
#include <map>
#include <string>

namespace HPCG_OCL {

class OCL {
public:
  static OCL *getOpenCL(void);
  void destoryOpenCL(void);
  cl_context getContext(void);
  cl_device_id getDeviceId(void);
  cl_program getProgram(void);
  cl_command_queue getCommandQueue(void);
  int initBuffer(SparseMatrix &A);
  cl_kernel getKernel(std::string);
  int clsparse_initCsrMatrix(const SparseMatrix h_A, clsparseCsrMatrix &d_A, int *col, int *rowoff);
  int clsparse_initDenseVector(cldenseVector &d_, int num_rows);
  int clsparse_initScalar(clsparseScalar &d_);
  int clsparse_initScalar(clsparseScalar &d_, double val);
private:
  OCL();
  ~OCL();
  void create();
  void ReleaseOpenCL();
  void BuildProgram();
  static OCL *self;
  cl_platform_id *platform;
  cl_device_id *device;
  cl_context context;
  cl_program program;
  cl_command_queue command_queue;
  std::map<std::string, cl_kernel> kernels;
};

}

#endif

