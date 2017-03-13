#pragma once

#include <armadillo>
#include <vector>
#include <tuple>

#define MAX_ITERS 100

struct KMedNode
{
	KMedNode(size_t n, size_t k) : N(n), K(k), 
						medoids(K,arma::fill::zeros),
						multiplicity(K,arma::fill::zeros),
						assignments(N,arma::fill::zeros) { }
	size_t N;
	size_t K;
	arma::uvec medoids;
	arma::uvec multiplicity;
	arma::uvec assignments;
};


void 
assign_to_medoids(const arma::mat& dists, KMedNode& kmn)
{
	arma::mat sub_dists = dists.rows(kmn.medoids);
	kmn.multiplicity.zeros();
	for(size_t i = 0; i < kmn.N; ++i)
	{
		arma::uvec idx = arma::find(kmn.medoids==i,1);
		if(idx.n_elem == 1)
		{
			size_t loc = idx[0];
			kmn.multiplicity[loc]++;
			kmn.assignments[i] = kmn.medoids[loc];
			continue;
		}
		
		size_t min_idx = sub_dists.col(i).index_min();
		kmn.assignments[i] = kmn.medoids[min_idx];
		kmn.multiplicity[min_idx]++;
	}
}

void
update_medoids(const arma::mat& dists, KMedNode& kmn)
{
	arma::uvec set;
	for(size_t k = 0; k < kmn.K; ++k)
	{
		set = arma::find(kmn.assignments == kmn.medoids[k]);
		arma::mat sub_dist = dists(set,set);
		arma::vec row_sums = arma::sum(sub_dist,1);
		size_t min_idx = row_sums.index_min();
		if(kmn.medoids[k] != set[min_idx])
			printf("Swapping %llu for %llu\n",set[min_idx],kmn.medoids[k]);
		kmn.medoids[k] = set[min_idx];
	}
}

arma::uvec
kmpp_init(const arma::mat& dists, size_t k, const arma::uvec& seeds)
{
	arma::uvec medoids(k);
	size_t start;
	printf("Initializing random center\n");
	if(seeds.n_elem > k)
		throw std::runtime_error("kmeans++ called with seed count greater than k");
	if(seeds.n_elem > 0)
	{
		for(size_t i = 0; i < seeds.n_elem; ++i)
			medoids(i) = seeds(i);
		start = seeds.n_elem;
	} else {
		// randomly initialize one value
		auto init = arma::randi<arma::uvec>(1,arma::distr_param(0,dists.n_rows-1));
		medoids(0)=init(0);
		start = 1;
	}
	for(size_t i = start; i < k; ++i)
	{
		// Dsq(x) = min_i D(x,medoid_i)^2
		// so minimize across columns to find the min value of D(x,medoids_i)
		arma::uvec assigned = medoids(arma::span(0,i-1));
		arma::vec Dsq = arma::square(arma::min(dists.cols(assigned),1));

		arma::vec cdf = arma::cumsum(Dsq/arma::accu(Dsq));
		arma::vec r = arma::randu<arma::vec>(1);
		arma::uvec loc = arma::find(cdf > r[0],1,"first");
		if(loc.is_empty())
		{
			medoids[i] = dists.n_rows-1;
		}
		else
		{
			medoids[i] = loc[0];
		}
	}

	return medoids;
}
// uses Voronoi iteration
KMedNode
select_kmedoids(const arma::mat& dists, size_t k, const arma::uvec& seeds = arma::uvec() )
{
	int N = dists.n_rows;
	KMedNode kmn(N, k);
	
	kmn.medoids = arma::sort(kmpp_init(dists,k,seeds));
	arma::uvec old_medoids = kmn.medoids;
	arma::uvec diff, sindex;

	bool converged = false;
	for(size_t i = 0; i < MAX_ITERS && !converged; ++i)
	{
		printf("Iteration %lu\n",i);
		assign_to_medoids(dists,kmn);
		update_medoids(dists,kmn);
		kmn.medoids = arma::sort(kmn.medoids);

		diff = arma::find(old_medoids != kmn.medoids);
		if(diff.is_empty())
			converged = true;
		else
			old_medoids = kmn.medoids;
	}

	assign_to_medoids(dists,kmn);
	return kmn;
}

std::tuple<pyarr_u,pyarr_u>
kmedoids(pyarr_d dists, size_t k, pyarr_u seeds = pyarr_u(0,nullptr))
{
	arma::arma_rng::set_seed_random();
	arma::mat d = py_to_mat(dists);
	arma::uvec seed_medoid = py_to_uvec(seeds);
	seed_medoid.print("Seed Medoids");
	KMedNode kmn = select_kmedoids(d,k, seed_medoid);
	return std::make_tuple(uvec_to_py(kmn.medoids),uvec_to_py(kmn.multiplicity));
}
