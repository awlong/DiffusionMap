//
//  DMap.h
//  DiffusionMap
//
//  Updated by Andrew Long on 08/18/17.
//  Copyright (c) 2016-2017 Andrew Long. All rights reserved.
//
//  Utilizes Armadillo (http://arma.sourceforge.net/) C++ LinAlg Library
//  Sanderson, C. Armadillo: An Open Source C++ Linear Algebra Library
//  for Fast Prototyping and Computationally Intensive Experiments.
//  Technical Report, NICTA, 2010.

#pragma once

#include "pyarma.h"
class LDMap
{
public:
	LDMap();
	void set_epsilon(double e);
	double get_epsilon() { return epsilon; }
	void set_num_evec(int n);
	int get_num_evec() { return num_evec; }
	int get_num_samples() { return N; }
	void set_dists(pyarr_d b);
	
    bool compute();
	
	pyarr_d get_evec_py();
	pyarr_d get_eval_py();
    pyarr_u get_multiplicity_py();
	arma::mat& get_evec();
	arma::vec& get_eval();
    arma::uvec& get_multiplicity();
    arma::mat& get_converted_evecs();
	void set_evec(pyarr_d evec);
	void set_eval(pyarr_d eval);
    void set_multiplicity(pyarr_u mult);
private:
	int N, num_evec;
	double epsilon;
	arma::mat eigenvec;
	arma::vec eigenval;
	arma::mat dists;
	arma::mat kernel;
    arma::uvec multiplicity;
    arma::mat converted_evecs;
	
	bool is_new_dist, is_new_kernel, is_new_solve;
	bool compute_kernel();
	bool compute_eigs();
    
    bool is_set_evals, is_set_evecs, is_set_mult;
    void normalize_evecs();
    void compute_converted_evecs();
};

