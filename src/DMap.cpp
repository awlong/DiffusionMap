#include "DMap.h"
#include <string>
#include <tuple>
#include <cstdio>
#include <exception>

void warn(std::string file, int line, std::string msg)
{
	std::cout << "[" << file << ":" << line << "] " << msg << std::endl;
}

#define WARNING(msg) warn(__FILE__,__LINE__,msg)

/****
*
* Base Diffusion Map
*
*
****/

DMap::DMap()
{
	N = 0;
	num_evec = 0;
	epsilon = 0;
	is_new_dist = false;
	is_new_kernel = false;
	is_new_solve = false;
	eigenvec.clear();
	eigenval.clear();
}

void DMap::set_epsilon(double e)
{
	epsilon = e;
	is_new_kernel = true;
}

void DMap::set_num_evec(int n)
{
	num_evec = n;
	is_new_solve = true;
}

bool DMap::compute()
{
	if(!compute_kernel()) {
		std::cout << "Distance matrix/epsilon stayed constant, skipping kernel reconstruction" << std::endl;
	}
	if(!compute_eigs()) {
		std::cout << "Eigenvalue problem stayed constant, skipping eigenvalue computation" << std::endl;
	}
	return true;
}

arma::mat& DMap::get_evec()
{	
	if( (is_new_dist || is_new_kernel || is_new_solve) == true)
	{
		WARNING("Warning: eigenvector results not up to date, rerun compute()");
	}
	return eigenvec;
}

void DMap::set_evec(const arma::mat& evec)
{
	is_new_dist = is_new_kernel = is_new_solve = false;
	eigenvec = evec;
	num_evec = eigenvec.n_cols;
	N = eigenvec.n_rows;
}

void DMap::set_evec(py::array evec)
{
	is_new_dist = is_new_kernel = is_new_solve = false;
	mat_np_init(eigenvec,evec);
	num_evec = eigenvec.n_cols;
	N = eigenvec.n_rows;
}


arma::vec& DMap::get_eval()
{
	if( (is_new_dist || is_new_kernel || is_new_solve) == true)
	{
		WARNING("Warning: eigenvalue results not up to date, rerun compute()");
	}
	return eigenval;
}

void DMap::set_eval(const arma::vec& eval)
{
	is_new_dist = is_new_kernel = is_new_solve = false;
	eigenval = eval;
}

void DMap::set_eval(py::array eval)
{
	is_new_dist = is_new_kernel = is_new_solve = false;
	vec_np_init(eigenval,eval);
}

void DMap::set_dists(const arma::mat& d)
{
	dists = d;
	N = dists.n_rows;
	is_new_dist = true;
}

void DMap::set_dists(py::array d)
{
	mat_np_init(dists,d);
	N = dists.n_rows;
	is_new_dist = true;
}

bool DMap::compute_kernel()
{
	if(!is_new_dist && !is_new_kernel)
		return false;
	
	is_new_dist = false;
	double inv_neg_2eps = -1.0/(2.0*epsilon);
	kernel = arma::square(dists);
	kernel = arma::exp(kernel * inv_neg_2eps);
	arma::vec rsum = arma::sum(kernel, 1);
	kernel = arma::diagmat(1.0f/rsum) * kernel;
	is_new_kernel = false;

	is_new_solve = true;
	return true;
}

bool DMap::compute_eigs()
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

		is_new_solve = false;
		return true;
	} catch(std::exception& e) {
		WARNING("Error in eigendecomposition: " + std::string(e.what()));
	}

	return false;
}

