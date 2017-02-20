# DiffusionMap
Implementation of the diffusion map (DMap) non-linear manifold learning algorithm (https://en.wikipedia.org/wiki/Diffusion_map) using Armadillo (http://arma.sourceforge.net/) [1] in C++. 

## DMap Theory - WIP [2][3][4][5]

## Functionality:
 - Operates on pairwise distance matrix, allowing for arbitrary distance measures to be employed.
 - Included code for matrix/graph alignment via a heuristic DFS scheme.
  - Distance code offers OpenMP parallelism to compute pairwise distances for a set of graphs.
 - Nyström interpolation to determine position in manifold of out of sample objects [7][8][9].
 - Python bindings for these codes via pybind11 (https://github.com/pybind/pybind11)
 
## Upcoming Features/To Do:
 - API-documentation and example scripts (C++ and Python)
 - Extend distance code into a separate submodule, allowing for a wider range of distance measures to be generated using this framework.
 - Smarter Pybind11 bindings to remove the need for "numpy.asarray" on returned matrices from functions
 - Landmark DMaps for accelerated out-of-sample embedding [10]
 - Automated bandwidth selection (determine region where full graph connectivity is maintained and select epsilon inside this region)
 

## Requirements:

LAPACK/BLAS/ARPACK libraries

Armadillo (headers and appropriate papers given)

pybind11 (given as a submodule to this package)


## Citations:

[1] Sanderson, C. Armadillo: An Open Source C++ Linear Algebra Library for Fast Prototyping and Computationally Intensive Experiments. Technical Report, NICTA, 2010.

[2] R. R. Coifman, S. Lafon, A. B. Lee, M. Maggioni, B. Nadler, F. Warner and S. W. Zucker; Proc. Natl. Acad. Sci. U. S. A., 2005, 102, 7426–7431.

[3] R. R. Coifman and S. Lafon, Appl. Comput. Harm. Anal.; 2006, 21, 5–30.

[4] R. R. Coifman, I. Kevrekidis, S. Lafon, M. Maggioni and B. Nadler; Multiscale Model. Simul., 2008, 7, 842–864.

[5] A. L. Ferguson, A. Z. Panagiotopoulos, I. G. Kevrekidis, and P. G. Debenedetti; Chem. Phys. Lett. 2011, 509, 1−11.

[6] R. Singh, J. Xu, and B. Berger; Proc. Natl. Acad. Sci. U. S. A. 2008, 105, 12763–12768.

[7] C. T. Baker and C. Baker; The numerical treatment of integral equations, Clarendon Press, Oxford, 1977, vol. 13.

[8] C. R. Laing, T. A. Frewen and I. G. Kevrekidis, Nonlinearity, 2007, 20, 2127.

[9] B. E. Sonday, M. Haataja and I. G. Kevrekidis, Phys. Rev. E, 2009, 80, 031102.

[10] A. W. Long and A.L. Ferguson (submitted 2016)
