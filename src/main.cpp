
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
 @file main.cpp

 HPCG routine
 */

// Main routine of a program that calls the HPCG conjugate gradient
// solver to solve the problem, and then prints results.

#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

#include <sys/time.h>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cassert>
#ifdef HPCG_DETAILED_DEBUG
using std::cin;
#endif
using std::endl;

#include <vector>

#include "hpcg.hpp"

#include "CheckAspectRatio.hpp"
#include "GenerateGeometry.hpp"
#include "GenerateProblem.hpp"
#include "GenerateCoarseProblem.hpp"
#include "SetupHalo.hpp"
#include "CheckProblem.hpp"
#include "ExchangeHalo.hpp"
#include "OptimizeProblem.hpp"
#include "WriteProblem.hpp"
#include "ReportResults.hpp"
#include "mytimer.hpp"
#include "ComputeSPMV_ref.hpp"
#include "ComputeMG_ref.hpp"
#include "ComputeResidual.hpp"
#include "CG.hpp"
#include "CG_ref.hpp"
#include "Geometry.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "CGData.hpp"
#include "TestCG.hpp"
#include "TestSymmetry.hpp"
#include "TestNorms.hpp"
#include <CL/cl.hpp>
#include "clSPARSE.h"
#include "clSPARSE-error.h"

#ifdef __OCL__
#include "OCL.hpp"
#endif


//===extern clsparseCsrMatrix d_A;
extern cldenseVector d_p, d_Ap, d_b, d_r, d_x;
extern clsparseScalar d_alpha, d_beta, d_normr, d_minus;
extern clsparseScalar d_rtz, d_oldrtz, d_Beta, d_Alpha, d_minusAlpha, d_pAp;

extern int *fcol, *frowOff;

//extern double spmv_time;
extern float *fval, *qt_matrixValues;
extern int *col, *rowOff, *nnzInRow, *Count;
extern local_int_t *qt_mtxIndl, *qt_rowOffset, *q_mtxIndl, *q_rowOffset;
extern clsparseCsrMatrix Od_A, d_A, d_Q, d_Qt, d_A_ref;
extern int clsparse_setup(SparseMatrix & h_A);


/*!
  Main driver program: Construct synthetic problem, run V&V tests, compute benchmark parameters, run benchmark, report results.

  @param[in]  argc Standard argument count.  Should equal 1 (no arguments passed in) or 4 (nx, ny, nz passed in)
  @param[in]  argv Standard argument array.  If argc==1, argv is unused.  If argc==4, argv[1], argv[2], argv[3] will be interpreted as nx, ny, nz, resp.

  @return Returns zero on success and a non-zero value otherwise.
*/

int main(int argc, char *argv[]) {

  /*struct timeval start, stop;
  gettimeofday(&start, NULL);*/

#ifndef HPCG_NO_MPI
  MPI_Init(&argc, &argv);
#endif

  HPCG_Params params;

  HPCG_Init(&argc, &argv, params);

  // Check if QuickPath option is enabled.
  // If the running time is set to zero, we minimize all paths through the program
  bool quickPath = (params.runningTime == 0);

  int size = params.comm_size, rank = params.comm_rank; // Number of MPI processes, My process ID

#ifdef HPCG_DETAILED_DEBUG
  if (size < 100 && rank == 0) {
    HPCG_fout << "Process " << rank << " of " << size << " is alive with " << params.numThreads << " threads." << endl;
  }

  if (rank == 0) {
    char c;
    std::cout << "Press key to continue" << std::endl;
    std::cin.get(c);
  }
#ifndef HPCG_NO_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif

  local_int_t nx, ny, nz;
  nx = (local_int_t)params.nx;
  ny = (local_int_t)params.ny;
  nz = (local_int_t)params.nz;
  int ierr = 0;  // Used to check return codes on function calls

  ierr = CheckAspectRatio(0.125, nx, ny, nz, "local problem", rank == 0);
  if (ierr) {
    return ierr;
  }

  /////////////////////////
  // Problem setup Phase //
  /////////////////////////

#ifdef HPCG_DEBUG
  double t1 = mytimer();
#endif

  // Construct the geometry and linear system
  Geometry *geom = new Geometry;
  GenerateGeometry(size, rank, params.numThreads, nx, ny, nz, geom);

  ierr = CheckAspectRatio(0.125, geom->npx, geom->npy, geom->npz, "process grid", rank == 0);
  if (ierr) {
    return ierr;
  }

  // Use this array for collecting timing information
  std::vector< double > times(10, 0.0);

  double setup_time = mytimer();

  SparseMatrix A, A_ref;
  InitializeSparseMatrix(A, geom);
  // Reference matrix to store reordered sparse matrix depending on Luby's coloring order.
  InitializeSparseMatrix(A_ref, geom);

  Vector b, x, xexact;
  GenerateProblem(A, &b, &x, &xexact);
  GenerateProblem(A_ref, &b, &x, &xexact);

  SetupHalo(A);

  int numberOfMgLevels = 4; // Number of levels including first
  SparseMatrix *curLevelMatrix = &A;
  SparseMatrix *curLevelMatrix_ref = &A_ref;
  for (int level = 1; level < numberOfMgLevels; ++level) {
    GenerateCoarseProblem(*curLevelMatrix);
    curLevelMatrix = curLevelMatrix->Ac; // Make the just-constructed coarse grid the next level
    GenerateCoarseProblem(*curLevelMatrix_ref);
    curLevelMatrix_ref = curLevelMatrix_ref->Ac; // Make the just-constructed coarse grid the next level
  }

  setup_time = mytimer() - setup_time; // Capture total time of setup
  times[9] = setup_time; // Save it for reporting

  curLevelMatrix = &A;
  Vector *curb = &b;
  Vector *curx = &x;
  Vector *curxexact = &xexact;
  for (int level = 0; level < numberOfMgLevels; ++level) {
    CheckProblem(*curLevelMatrix, curb, curx, curxexact);
    curLevelMatrix = curLevelMatrix->Ac; // Make the nextcoarse grid the next level
    curb = 0; // No vectors after the top level
    curx = 0;
    curxexact = 0;
  }


  CGData data;
  InitializeSparseCGData(A, data);

  ////////////////////////////////////
  // Reference SpMV+MG Timing Phase //
  ////////////////////////////////////

  // Call Reference SpMV and MG. Compute Optimization time as ratio of times in these routines

  local_int_t nrow = A.localNumberOfRows;
  local_int_t ncol = A.localNumberOfColumns;

  Vector x_overlap, b_computed;
  InitializeVector(x_overlap, ncol); // Overlapped copy of x vector
  InitializeVector(b_computed, nrow); // Computed RHS vector


  // Record execution time of reference SpMV and MG kernels for reporting times
  // First load vector with random values
  FillRandomVector(x_overlap);

  int numberOfCalls = 10;
  if (quickPath) {
    numberOfCalls = 1;  //QuickPath means we do on one call of each block of repetitive code
  }
  double t_begin = mytimer();
  for (int i = 0; i < numberOfCalls; ++i) {
    ierr = ComputeSPMV_ref(A, x_overlap, b_computed); // b_computed = A*x_overlap
    if (ierr) {
      HPCG_fout << "Error in call to SpMV: " << ierr << ".\n" << endl;
    }
    ierr = ComputeMG_ref(A, b_computed, x_overlap); // b_computed = Minv*y_overlap
    if (ierr) {
      HPCG_fout << "Error in call to MG: " << ierr << ".\n" << endl;
    }
  }
  times[8] = (mytimer() - t_begin) / ((double) numberOfCalls); // Total time divided by number of calls.
#ifdef HPCG_DEBUG
  if (rank == 0) {
    HPCG_fout << "Total SpMV+MG timing phase execution time in main (sec) = " << mytimer() - t1 << endl;
  }
#endif

  ///////////////////////////////
  // Reference CG Timing Phase //
  ///////////////////////////////

#ifdef HPCG_DEBUG
  t1 = mytimer();
#endif
  int global_failure = 0; // assume all is well: no failures

  int niters = 0;
  int totalNiters_ref = 0;
  double normr = 0.0;
  double normr0 = 0.0;
  int refMaxIters = 50;
  numberOfCalls = 1; // Only need to run the residual reduction analysis once

  // Compute the residual reduction for the natural ordering and reference kernels
  std::vector< double > ref_times(9, 0.0);
  double tolerance = 0.0; // Set tolerance to zero to make all runs do maxIters iterations
  int err_count = 0;
  for (int i = 0; i < numberOfCalls; ++i) {
    ZeroVector(x);
    ierr = CG_ref(A, data, b, x, refMaxIters, tolerance, niters, normr, normr0, &ref_times[0], true);
    if (ierr) {
      ++err_count;  // count the number of errors in CG
    }
    totalNiters_ref += niters;
  }
  if (rank == 0 && err_count) {
    HPCG_fout << err_count << " error(s) in call(s) to reference CG." << endl;
  }
  double refTolerance = normr / normr0;

  // Call user-tunable set up function.
  double t7 = mytimer();

  /* call OptimizeProblem to all grid levels so the reference matrix is reordered
  based on Luby's color reordering algorithm*/
  clsparse_setup(A);
  A.optimizationData = &A_ref;
  OptimizeProblem(&A);
#ifdef __OCL__
  HPCG_OCL::OCL::getOpenCL()->initBuffer(A);
#endif

  t7 = mytimer() - t7;
  times[7] = t7;
#ifdef HPCG_DEBUG
  if (rank == 0) {
    HPCG_fout << "Total problem setup time in main (sec) = " << mytimer() - t1 << endl;
  }
#endif

#ifdef HPCG_DETAILED_DEBUG
  if (geom->size == 1) {
    WriteProblem(*geom, A, b, x, xexact);
  }
#endif


  //////////////////////////////
  // Validation Testing Phase //
  //////////////////////////////

#ifdef HPCG_DEBUG
  t1 = mytimer();
#endif
  TestCGData testcg_data;
  testcg_data.count_pass = testcg_data.count_fail = 0;
  TestCG(A, geom, data, b, x, testcg_data);

  TestSymmetryData testsymmetry_data;
  TestSymmetry(A, b, xexact, testsymmetry_data);

#ifdef HPCG_DEBUG
  if (rank == 0) {
    HPCG_fout << "Total validation (TestCG and TestSymmetry) execution time in main (sec) = " << mytimer() - t1 << endl;
  }
#endif

#ifdef HPCG_DEBUG
  t1 = mytimer();
#endif

  //////////////////////////////
  // Optimized CG Setup Phase //
  //////////////////////////////

  niters = 0;
  normr = 0.0;
  normr0 = 0.0;
  err_count = 0;
  int tolerance_failures = 0;

  int optMaxIters = 10 * refMaxIters;
  int optNiters = refMaxIters;
  double opt_worst_time = 0.0;

  std::vector< double > opt_times(9, 0.0);

  // Compute the residual reduction and residual count for the user ordering and optimized kernels.
  for (int i = 0; i < numberOfCalls; ++i) {
    ZeroVector(x); // start x at all zeros
    double last_cummulative_time = opt_times[0];
    ierr = CG(A, data, b, x, optMaxIters, refTolerance, niters, normr, normr0, &opt_times[0], true);
    if (ierr) {
      ++err_count;  // count the number of errors in CG
    }
    if (normr / normr0 > refTolerance) {
      ++tolerance_failures;  // the number of failures to reduce residual
    }

    // pick the largest number of iterations to guarantee convergence
    if (niters > optNiters) {
      optNiters = niters;
    }

    double current_time = opt_times[0] - last_cummulative_time;
    if (current_time > opt_worst_time) {
      opt_worst_time = current_time;
    }
  }

#ifndef HPCG_NO_MPI
  // Get the absolute worst time across all MPI ranks (time in CG can be different)
  double local_opt_worst_time = opt_worst_time;
  MPI_Allreduce(&local_opt_worst_time, &opt_worst_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif


  if (rank == 0 && err_count) {
    HPCG_fout << err_count << " error(s) in call(s) to optimized CG." << endl;
  }
  if (tolerance_failures) {
    global_failure = 1;
    if (rank == 0) {
      HPCG_fout << "Failed to reduce the residual " << tolerance_failures << " times." << endl;
    }
  }

  ///////////////////////////////
  // Optimized CG Timing Phase //
  ///////////////////////////////

  // Here we finally run the benchmark phase
  // The variable total_runtime is the target benchmark execution time in seconds

  double total_runtime = params.runningTime;
  int numberOfCgSets = int(total_runtime / opt_worst_time) + 1; // Run at least once, account for rounding

#ifdef HPCG_DEBUG
  if (rank == 0) {
    HPCG_fout << "Projected running time: " << total_runtime << " seconds" << endl;
    HPCG_fout << "Number of CG sets: " << numberOfCgSets << endl;
  }
#endif

  /* This is the timed run for a specified amount of time. */

  optMaxIters = optNiters;
  double optTolerance = 0.0;  // Force optMaxIters iterations
  TestNormsData testnorms_data;
  testnorms_data.samples = numberOfCgSets;
  testnorms_data.values = new double[numberOfCgSets];

  for (int i = 0; i < numberOfCgSets; ++i) {
    ZeroVector(x); // Zero out x
    ierr = CG(A, data, b, x, optMaxIters, optTolerance, niters, normr, normr0, &times[0], true);
    if (ierr) {
      HPCG_fout << "Error in call to CG: " << ierr << ".\n" << endl;
    }
    if (rank == 0) {
      HPCG_fout << "Call [" << i << "] Scaled Residual [" << normr / normr0 << "]" << endl;
    }
    testnorms_data.values[i] = normr / normr0; // Record scaled residual from this run
  }

  // Compute difference between known exact solution and computed solution
  // All processors are needed here.
#ifdef HPCG_DEBUG
  double residual = 0;
  ierr = ComputeResidual(A.localNumberOfRows, x, xexact, residual);
  if (ierr) {
    HPCG_fout << "Error in call to compute_residual: " << ierr << ".\n" << endl;
  }
  if (rank == 0) {
    HPCG_fout << "Difference between computed and exact  = " << residual << ".\n" << endl;
  }
#endif

  // Test Norm Results
  ierr = TestNorms(testnorms_data);

  ////////////////////
  // Report Results //
  ////////////////////

  // Report results to YAML file
  ReportResults(A, numberOfMgLevels, numberOfCgSets, refMaxIters, optMaxIters, &times[0], testcg_data, testsymmetry_data, testnorms_data, global_failure, quickPath);

  // Clean up
  free_refmatrix_m(A_ref);
  DeleteMatrix(A); // This delete will recursively delete all coarse grid data
  DeleteCGData(data);
  DeleteVector(x);
  DeleteVector(b);
  DeleteVector(xexact);
  DeleteVector(x_overlap);
  DeleteVector(b_computed);
  delete [] testnorms_data.values;

  clsparseCsrMetaDelete(&d_A);
  clReleaseMemObject(d_alpha.value);
  clReleaseMemObject(d_beta.value);
  clReleaseMemObject(d_normr.value);
  clReleaseMemObject(d_minus.value);
  clReleaseMemObject(d_rtz.value);
  clReleaseMemObject(d_oldrtz.value);
  clReleaseMemObject(d_pAp.value);
  clReleaseMemObject(d_Alpha.value);
  clReleaseMemObject(d_Beta.value);
  clReleaseMemObject(d_minusAlpha.value);
  clReleaseMemObject(d_A.col_indices);
  clReleaseMemObject(d_A.row_pointer);
  clReleaseMemObject(d_A.values);
  clReleaseMemObject(d_p.values);
  clReleaseMemObject(d_Ap.values);
  clReleaseMemObject(d_b.values);
  clReleaseMemObject(d_r.values);
  clReleaseMemObject(d_x.values);

  delete [] fval;
  delete [] fcol;
  delete [] frowOff;
  
  clReleaseMemObject ( Od_A.col_indices );
  clReleaseMemObject ( Od_A.row_pointer );
  clReleaseMemObject ( Od_A.values );
  
  clReleaseMemObject ( d_Qt.col_indices );
  clReleaseMemObject ( d_Qt.row_pointer );
  clReleaseMemObject ( d_Qt.values );
  
  clReleaseMemObject ( d_Q.col_indices );
  clReleaseMemObject ( d_Q.row_pointer );
  clReleaseMemObject ( d_Q.values );
  
  clReleaseMemObject ( d_A_ref.col_indices );
  clReleaseMemObject ( d_A_ref.row_pointer );
  clReleaseMemObject ( d_A_ref.values );

  /** Close & release resources */
  clsparseStatus status = clsparseTeardown();
  if (status != clsparseSuccess) {
    std::cout << "Problem with closing clSPARSE library."
              << " Error: " << status << std::endl;
  }

  delete [] col;
  delete [] rowOff;
  delete [] nnzInRow;
  delete [] Count;
  delete [] qt_matrixValues;
  delete [] qt_mtxIndl;
  delete [] qt_rowOffset;
  delete [] q_mtxIndl;
  delete [] q_rowOffset;

  HPCG_Finalize();

  // Finish up
#ifndef HPCG_NO_MPI
  MPI_Finalize();
#endif

#ifdef __OCL__
  HPCG_OCL::OCL::getOpenCL()->destoryOpenCL();
#endif
  /*gettimeofday(&stop, NULL);
  std::cout << "\n SPMV time:" << spmv_time;
  std::cout << "\n Total time:" << (((stop.tv_sec * 1000000) + stop.tv_usec) - ((start.tv_sec * 1000000) + start.tv_usec)) / 1000000.0;*/
  return 0;
}
