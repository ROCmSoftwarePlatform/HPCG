# HPCG#
This repository holds the source of HPCG - High Performance Conjugate Gradient Benchmark with hcSPARSE and clSPARSE csrmv-adaptive integration. 
### Pre-requisites ###
* Linux 
* [HCC Compiler](https://bitbucket.org/multicoreware/hcc/wiki/Home)
* AMD GPU and corresponding [driver](http://support.amd.com/en-us/download)
* [OpenCL SDK](http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/)
* [hcSPARSE](https://bitbucket.org/multicoreware/hcsparse)
* clSPARSE

### Additional information on building hcSPARSE from source ###
Follow the instructions given on hcSPARSE page. Make sure to run 
```
#!bash

make
```
 before 
```
#!bash

sudo make install
```
In case of any errors while building similar to *LLVM tile uniform failed* 
```
#!bash

export CLAMP_NOTILECHECK=ON
```

### clSPARSE ###
### Pre-requisites ###
* Download [clSPARSE](https://github.com/clMathLibraries/clSPARSE/releases/download/0.8.1.0/clSPARSE­0.8.
1.0­Linux­x64.tar.gz) for linux. 
* Update the clSPARSE include and library path in Makefile.Linux_Serial in the setup directory. (for ex. -­I /path/to/clsparse/include -L /path/to/clsparse/lib64) 
* Add path to clSPARSE library to LD_LIBRARY_PATH in the terminal. 
(example)

```
#!bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/clsparse/lib64 

```
* [OpenCL SDK](http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/). 
* Update OpenCL include and library path in Makefile.Linux_Serial in the setup directory.          (for ex. -­I /opt/AMDAPPSDK­3.0/include/  -­L  /opt/AMDAPPSDK­3.0/lib/x86_64/) 

### Steps to build and run HPCG###
* Inside HPCG root directory,
```
#!bash
    mkdir build
	cd build
   	../configure Linux_Serial
	make
	cd bin
	./xhpcg 

```
* Check the yaml file created in the directory where the HPCG executable is located for the 
output validation. At the end of the file it would have been reported whether the exeution 
was valid. 

Additional instructions to configure and build are available in the root directory.