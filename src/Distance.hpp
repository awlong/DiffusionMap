//
//  Distance.hpp
//  Distance functions
//
//  Updated by Andrew Long on 08/22/17.
//  Copyright (c) 2016-2017 Andrew Long. All rights reserved.
//
//  Utilizes Armadillo (http://arma.sourceforge.net/) C++ LinAlg Library
//  Sanderson, C. Armadillo: An Open Source C++ Linear Algebra Library
//  for Fast Prototyping and Computationally Intensive Experiments.
//  Technical Report, NICTA, 2010.

#pragma once
#include <armadillo>
#include "DFSAlign.hpp"
#include "pyarma.h"
#include "Heuristic.h"
#include <pybind11/stl.h>
#define BRANCHES 12

double
dist(pyarr_d a, pyarr_d b, MatchHeuristic heur = MatchHeuristic::ISORANK)
{
	arma::mat oa = py_to_mat(a);
	arma::mat ob = py_to_mat(b);
	return dist_dfs(oa, ob, BRANCHES, heur);
}

pyarr_d
pdist(std::vector<pyarr_d> pyobjs, MatchHeuristic heur = MatchHeuristic::ISORANK)
{
	size_t N = pyobjs.size();
	arma::mat dists(N,N,arma::fill::zeros);
	std::vector<arma::mat> objs(N);
	for(size_t i = 0; i < N; ++i)
		objs.at(i) = py_to_mat(pyobjs.at(i));
// load balances by computing subdiagonal elements from the bottom of the matrix
#if defined(ENABLE_OPENMP)
		size_t n_rows = (N-1)/2;
	#pragma omp parallel for shared(dists)
		for(size_t i = 0; i < n_rows; ++i) 
		{
			for(size_t j = 0; j < N; ++j)
			{
				size_t idx_i = i, idx_j = j;
				if(idx_i >= idx_j)
				{
					idx_i = N - 2 - idx_i;
					idx_j = N - 1 - idx_j;
				}
				double d = dist_dfs(objs.at(idx_i), objs.at(idx_j), BRANCHES, heur);
				dists(idx_i, idx_j) = d;
				dists(idx_j, idx_i) = d;
			}
		}

		if(N % 2 == 0)
		{
		#pragma omp parallel for shared(dists)
			for(size_t j = n_rows+1; j < N; ++j)
			{
				double d = dist_dfs(objs.at(n_rows), objs.at(j), BRANCHES, heur);
				dists(n_rows, j) = d;
				dists(j, n_rows) = d;
			}
		}
#else	
	for(size_t i = 0; i < N; ++i)
	{
		for(size_t j=i+1; j < N; ++j)
		{
			dists(i,j) = dist_dfs(objs.at(i), objs.at(j), BRANCHES, heur);
			dists(j,i) = dists(i,j);
		}
	}
#endif
	return mat_to_py(dists);
}


pyarr_d
pdist2(std::vector<pyarr_d> p1, std::vector<pyarr_d> p2, MatchHeuristic heur = MatchHeuristic::ISORANK)
{
    std::vector<arma::mat> o1 = convert_matlist(p1);
    std::vector<arma::mat> o2 = convert_matlist(p2);
	size_t N1 = o1.size();
	size_t N2 = o2.size();
    arma::mat dists(N1,N2);
    // ensuring good parallelization
    if(N1 == 1)
    {
        #if defined(ENABLE_OPENMP)
        #pragma omp parallel for shared(dists)
        #endif
        for(size_t j = 0; j < N2; ++j)
			dists(0,j) = dist_dfs(o1.at(0), o2.at(j), BRANCHES, heur);
    }
    else if(N2 == 1)
    {
        #if defined(ENABLE_OPENMP)
        #pragma omp parallel for shared(dists)
        #endif
        for(size_t i = 0; i < N1; ++i)
			dists(i,0) = dist_dfs(o1.at(i), o2.at(0), BRANCHES, heur);

    }
    else
    {
        #if defined(ENABLE_OPENMP)
        #pragma omp parallel for shared(dists)
        #endif
	    for(size_t i = 0; i < N1; ++i)
		    for(size_t j = 0; j < N2; ++j)
			    dists(i,j) = dist_dfs(o1.at(i), o2.at(j), BRANCHES, heur);
	}
	return mat_to_py(dists);
}
