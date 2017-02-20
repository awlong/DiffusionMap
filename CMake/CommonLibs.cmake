include(FindBLAS)
message("blas_linker_flags: ${BLAS_LINKER_FLAGS}")
message("blas_libraries: ${BLAS_LIBRARIES}")

include(FindLAPACK)
message("lapack_linker_flags: ${LAPACK_LINKER_FLAGS}")
message("lapack_libraries: ${LAPACK_LIBRARIES}")

include(FindOpenMP)
if(OPENMP_FOUND)
	message("openmp_cxx_flags: ${OpenMP_CXX_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -DENABLE_OPENMP=1")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif(OPENMP_FOUND)
