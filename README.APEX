
===========================================================
High Performance Conjugate Gradient Benchmark for APEX
===========================================================

The HPCG benchmarking used for the APEX procurement is based on the 3.0-beta
release currently being finalized by the HPCG developers.

===========================================================

== I. 	Description: ==

HPCG is a software package that performs a fixed number of multigrid preconditioned
(using a symmetric Gauss-Seidel smoother) conjugate gradient (PCG) iterations using double
precision (64 bit) floating point values.


== II. Parallelism: ==

HPCG implements MPI-based node distributed parallelism and OpenMP threading within the
MPI rank. OpenMP parallel-for regions are used for most vector and matrix operations
with the main SpMV threaded over matrix rows.


== III. How to Build: == 

(1) Create an architecture reference file in the setup directory. Examples are provided
for Linux platforms with MPI and OpenMP using a variety of compilers. For instance, name
this file Make.Linux_APEX.

(2) In the Makefile in the base directory modify the arch = variable to state Linux_APEX.

(3) Run make

(4) The xhpcg binary will be created in the bin directory


== IV.  How to Run: == 

HPCG reads the problem size definition from the hpcg.dat file found in the working
directory. The file format is:

NX, NY, NZ
TIME

The NX, NY and NZ variables define the problem size per node. The Time value specifies 
how long the benchmark should execute for. 

For APEX reference runs the mesh size per MPI rank has been defined at: 272 272 136. The
output of HPCG describes the global mesh size which takes the per-node mesh size and rank
decomposition to calculate the global sizes. Weak scaling is used to increase the size of
the mesh. A TIME value of 1800 must be used for the benchmark to report a valid result in
machine acceptance. Projected responses based on simulation or other performance models
may be run with a shorter time as needed but final acceptance of applications will
require the longer, 1800 second base run.

Global Problem Sizes Definitions:

Small - 272 272 136
Medium - 1088 1088 1088
Large (APEX Reference Problem) - 4352 8704 3264
Grand Challenge (APEX Target Problem) - 8704 17408 6528

Mapping of MPI ranks to nodes or global mesh decomposition over nodes can be modifed by
the user as required but the final mesh must meet the sizes above exactly.


== V.	Reporting Results ==

HPCG will produce a .yaml file in the directory where it is run with performance 
summaries of the data. In the "Final Summary" located at the bottom of the YAML file
is a GFLOP/s value. HPCG will also self report a VALID or INVALID result. Only VALID
results are to be provided in the RFP response.

The GFLOP/s value in the Final Summary should be reported in the APEX results
spreadsheet.

All modified source code, added Makefiles and YAML files are to be provided in with the
response.




