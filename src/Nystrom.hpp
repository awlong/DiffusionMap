//
//  Nystrom.h
//  Nystrom extension for interpolating out of sample points
//
//  Updated by Andrew Long on 11/30/16.
//  Copyright (c) 2016 Andrew Long. All rights reserved.
//
//  Utilizes Armadillo (http://arma.sourceforge.net/) C++ LinAlg Library
//  Sanderson, C. Armadillo: An Open Source C++ Linear Algebra Library
//  for Fast Prototyping and Computationally Intensive Experiments.
//  Technical Report, NICTA, 2010.

#pragma once

#include "DMap.h"
#include "LDMap.h"

#if defined(ENABLE_OPENMP)
#include <omp.h>
#endif


arma::vec
nystrom_sample(DMap& dmap, arma::vec d)
{
	d = arma::exp(-arma::square(d)/(2.0*dmap.get_epsilon()));
	double inv_rsum = 1.0f/arma::accu(d);
	d *= inv_rsum;
	
	arma::rowvec loc = d.t()*dmap.get_evec();
	loc /= dmap.get_eval().t();
	return loc.t();
}

arma::mat
nystrom(DMap& dmap, arma::mat dists)
{
	size_t Ndmap = dmap.get_num_samples();
	size_t Ncol = dists.n_cols, Nrow = dists.n_rows;
	// determine which size is correct for the DMap
	// want to select along columns
	if(Ndmap != Nrow && Ndmap != Ncol)
	{
		printf("Error in Nystrom: [DMap: %lu samples] [Distance Matrix: %lu x %lu]\n",Ndmap,Nrow,Ncol);
		throw std::runtime_error("Error: nystrom of distance matrix requires one distance matrix dimension to match the dmap");
	}
	if(Ndmap != Nrow)
	{
		dists = dists.t();
		Nrow = dists.n_rows;
		Ncol = dists.n_cols;
	}
	arma::mat coords(dmap.get_num_evec(), Ncol);
#if defined(ENABLE_OPENMP)
#pragma omp parallel for shared(coords)
#endif
	for(size_t i = 0; i < Ncol; ++i)
		coords.col(i) = nystrom_sample(dmap, dists.col(i));
	coords = coords.t();
	return coords;
}

pyarr_d
nystrom(DMap& dmap, pyarr_d dists)
{
	arma::mat d = py_to_mat(dists);
	arma::mat nys = nystrom(dmap,d);
	return mat_to_py(nys);
}

arma::vec
nystrom_sample(LDMap& ldmap, arma::vec d)
{
    d = arma::exp(-arma::square(d)/(2.0 * ldmap.get_epsilon()));
    double inv_rsum = 1.0f / arma::accu(d % ldmap.get_multiplicity());
    d *= inv_rsum;

    arma::rowvec loc = d.t() * ldmap.get_converted_evecs();
    return loc.t();
}

arma::mat
nystrom(LDMap& ldmap, arma::mat dists)
{
    size_t N = ldmap.get_num_samples();
    size_t Ncol = dists.n_cols, Nrow = dists.n_rows;
    if(N != Nrow && N != Ncol)
    {
        printf("Error in Nystrom: [LDMap: %lu samples] [Distance Matrix: %lu x %lu]\n", N, Nrow, Ncol);
        throw std::runtime_error("Error: nystrom of distance matrix requires one distance matrix dimension to match the number of samples used in landmark dmap construction");
    }
    if(N != Nrow)
    {
        dists = dists.t();
        Nrow = dists.n_rows;
        Ncol = dists.n_cols;
    }
    arma::mat coords(ldmap.get_num_evec(), Ncol);
#if defined(ENABLE_OPENMP)
#pragma omp parallel for shared(coords)
#endif
	for(size_t i = 0; i < Ncol; ++i)
		coords.col(i) = nystrom_sample(ldmap, dists.col(i));
	coords = coords.t();
	return coords;
}

pyarr_d
nystrom(LDMap& ldmap, pyarr_d dists)
{
    arma::mat d = py_to_mat(dists);
    arma::mat nys = nystrom(ldmap, d);
    return mat_to_py(nys);
}
