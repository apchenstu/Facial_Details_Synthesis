make QUIET_MAKE=1 TARGET=NEHALEM DYNAMIC_ARCH=1 HOSTCC=gcc NUM_THREADS=64 BINARY=64 CC=x86_64-w64-mingw32-gcc FC=x86_64-w64-mingw32-gfortran
[INFO] : TIMER value: INT_ETIME (given by make.inc)
[INFO] : TIMER value: INT_ETIME (given by make.inc)
touch libopenblasp-r0.2.19.a
make -s -j 16 -C test all
make -s -j 16 -C utest all
make -s -j 16 -C ctest all

 OpenBLAS build complete. (BLAS CBLAS LAPACK LAPACKE)

  OS               ... WINNT             
  Architecture     ... x86_64               
  BINARY           ... 64bit                 
  C compiler       ... GCC  (command line : x86_64-w64-mingw32-gcc)
  Fortran compiler ... GFORTRAN  (command line : x86_64-w64-mingw32-gfortran)
  Library Name     ... libopenblasp-r0.2.19.a (Multi threaded; Max num-threads is 64)

To install the library, you can run "make PREFIX=/path/to/your/installation install".

├── bin
│   └── libopenblas.dll       The shared library for Visual Studio and GCC.
├── include
│   ├── cblas.h
│   ├── f77blas.h
│   ├── lapacke_config.h
│   ├── lapacke.h
│   ├── lapacke_mangling.h
│   ├── lapacke_utils.h
│   └── openblas_config.h
├── lib
│   ├── libopenblas.a         The static library. Only work with GCC.
│   └── libopenblas.dll.a     The import library for Visual Studio.
└── readme.txt
