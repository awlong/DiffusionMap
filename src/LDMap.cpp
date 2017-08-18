#include "LDMap.h"
#include <string>
#include <tuple>
#include <cstdio>
#include <exception>


/****
*
* Landmark Diffusion Map
*
*
****/

LDMap::LDMap()
{
	N = 0;
	num_evec = 0;
	epsilon = 0;
	is_new_dist = false;
	is_new_kernel = false;
	is_new_solve = false;
    is_set_evecs = false;
    is_set_mult = false;
	eigenvec.clear();
	eigenval.clear();
    multiplicity.clear();
    converted_evecs.clear();
}

void LDMap::set_epsilon(double e)
{
	epsilon = e;
	is_new_kernel = true;
}

void LDMap::set_num_evec(int n)
{
	num_evec = n;
	is_new_solve = true;
}

bool LDMap::compute()
{
	if(!compute_kernel()) {
		std::cout << "Distance matrix/epsilon stayed constant, skipping kernel reconstruction" << std::endl;
	}
	if(!compute_eigs()) {
		std::cout << "Eigenvalue problem stayed constant, skipping eigenvalue computation" << std::endl;
	}
	return true;
}

arma::mat& LDMap::get_evec()
{	
	if( (is_new_dist || is_new_kernel || is_new_solve) == true)
	{
		WARNING("Warning: eigenvector results not up to date, rerun compute()");
	}
	return eigenvec;
}

pyarr_d LDMap::get_evec_py()
{	
	if( (is_new_dist || is_new_kernel || is_new_solve) == true)
	{
		WARNING("Warning: eigenvector results not up to date, rerun compute()");
	}
	return mat_to_py(eigenvec);
}

void LDMap::set_evec(pyarr_d evec)
{
	is_new_dist = is_new_kernel = is_new_solve = false;
	eigenvec = py_to_mat(evec);
	num_evec = eigenvec.n_cols;
	N = eigenvec.n_rows;
    is_set_evecs = true;
    if(is_set_mult && is_set_evals)
        compute_converted_evecs();
}


arma::vec& LDMap::get_eval()
{
	if( (is_new_dist || is_new_kernel || is_new_solve) == true)
	{
		WARNING("Warning: eigenvalue results not up to date, rerun compute()");
	}
	return eigenval;
}

pyarr_d LDMap::get_eval_py()
{
	if( (is_new_dist || is_new_kernel || is_new_solve) == true)
	{
		WARNING("Warning: eigenvalue results not up to date, rerun compute()");
	}
	return vec_to_py(eigenval);
}

void LDMap::set_eval(pyarr_d eval)
{
	is_new_dist = is_new_kernel = is_new_solve = false;
	eigenval = py_to_vec(eval);
    is_set_evals = true;
    if(is_set_mult && is_set_evecs)
        compute_converted_evecs();
}

void LDMap::set_multiplicity(pyarr_u mult)
{
    multiplicity = py_to_uvec(mult);
    is_set_mult = true;
    if(is_set_evecs && is_set_evals)
        compute_converted_evecs();
}

arma::uvec& LDMap::get_multiplicity()
{
    if(!is_set_mult)
        WARNING("Warning: multiplicities unset");
    return multiplicity;
}

pyarr_u LDMap::get_multiplicity_py()
{
    if(!is_set_mult)
        WARNING("Warning: multiplicities unset");
    return uvec_to_py(multiplicity);
}

void LDMap::set_dists(pyarr_d b)
{
	dists = py_to_mat(b);
	N = dists.n_rows;
	is_new_dist = true;
}

void LDMap::compute_converted_evecs()
{
    arma::mat mult_dmat = arma::diagmat(arma::conv_to<arma::vec>::from(multiplicity));
    converted_evecs = mult_dmat * eigenvec;
    converted_evecs.each_row() /= eigenval.t();
}

arma::mat& LDMap::get_converted_evecs()
{
    if(!(is_set_mult && is_set_evecs && is_set_evals))
        WARNING("Attempting to access unset converted eigenvectors");
    return converted_evecs;
}

void LDMap::normalize_evecs()
{
    if(!is_set_mult)
        return;
    
    arma::mat mult_evec = arma::diagmat(arma::conv_to<arma::vec>::from(multiplicity));
    mult_evec = mult_evec * eigenvec;
    for(int i = 0; i < num_evec; ++i)
    {
        double norm = arma::dot(mult_evec.col(i), eigenvec.col(i));
        eigenvec.col(i) /= sqrt(norm);
    }
}

bool LDMap::compute_kernel()
{
	if(!is_new_dist && !is_new_kernel)
		return false;
    
    arma::mat mult_dmat;
    if(!is_set_mult)
    {
        WARNING("Warning: Multiplicities not set! Assuming uniform multiplicity=1");
        mult_dmat = arma::eye(N,N);
    }
    else
    {
        mult_dmat = arma::diagmat(arma::conv_to<arma::vec>::from(multiplicity));
	}

	is_new_dist = false;
	double inv_neg_2eps = -1.0/(2.0*epsilon);
	kernel = arma::square(dists);
	kernel = arma::exp(kernel * inv_neg_2eps);
    arma::vec rsum = arma::sum(kernel * mult_dmat, 1);
    kernel = (arma::diagmat(1.0f/rsum) * kernel) * mult_dmat;
	is_new_kernel = false;

	is_new_solve = true;
	return true;
}

bool LDMap::compute_eigs()
{
	if(!is_new_solve){
		return false;
	}
	arma::sp_mat test(kernel);
	try
	{
		arma::cx_mat evecs;
		arma::cx_vec evals;
		arma::eigs_gen(evals, evecs, test, num_evec, "lm"); 
		
		arma::vec im_evals = arma::imag(evals);
		if(arma::any(im_evals)) {
			throw std::runtime_error("eigendecomposition returns imaginary eigenvalues");
		}

		eigenvec = arma::real(evecs);
		eigenval = arma::real(evals);
		
		arma::uvec sortidx = arma::stable_sort_index(eigenval,"descend");
		eigenval(arma::span::all) = eigenval(sortidx);
		eigenvec.cols(arma::span::all) = eigenvec.cols(sortidx);
        normalize_evecs();

		is_new_solve = false;
        is_set_evals = true;
        is_set_evecs = true;
        compute_converted_evecs();
		return true;
	} catch(std::exception& e) {
		WARNING("Error in eigendecomposition: " + std::string(e.what()));
	}
	return false;
}

