// Andrew Long
// Updated: Feb 13, 2017
// DFS Matrix Alignment Codes

#ifndef __DFS_ALIGN__
#define __DFS_ALIGN__

#include <vector>
#include <queue>
#include <algorithm>
#include <functional>
#include <utility>
#include <armadillo>

#include "GridDLL.hpp"



#define DO_MAX_CUT 0
#define MAX_EVALS (N1*N1*N2*N2*n_branch)

inline double
L11_norm(const arma::mat& A, const arma::mat& B)
{
	if(A.size() != B.size())
	{
		throw std::logic_error("L1,1-norm works only for equal sized graphs");
	}
	return arma::norm(arma::vectorise(A - B),1);
}

inline void
fill_range(arma::uvec& v, long N)
{
	v.set_size(N);
	for(long i = 0; i < N; ++i)
		v(i) = i;
}


typedef std::pair<double, int> MoveIndex;

class DFSAlign
{
private:
	double min_dist, opt_dist;
	long N1, N2, Nelem;
    long n_branch;
	arma::mat M1;
	arma::mat M1_t;
	arma::mat M2;
	arma::uvec min_perm1;
	arma::uvec min_perm2;
	arma::mat min_perm;
	arma::uvec fevals;
	arma::mat score;
	GridDLL full_list;
    std::vector<GridPoint> all_moves;
    std::vector<MoveIndex> all_move_dists;
    long num_all;
    
	bool isAligned;
	
	
	void recursiveHelper(int depth, double currentD, GridPoint root, GridDLL& list, arma::uvec& perm1, arma::uvec& perm2)
	{
		if(min_dist == opt_dist)
			return;

        perm1[depth] = root.first;
        perm2[depth] = root.second;
        //TODO: if N1 != N2, you're done "matching", so it shouldn't really matter what else happens
        if(depth == N1-1)
        {
            
            if(N1 != N2)
            {
                std::vector<int> vec2;
                // fill in the rest of the values
                // unassigned N1 will be N1..N2-1
                // unassigned N2 requires identification
                arma::uvec bp = arma::sort(perm2,"ascend");
                int c = 0;
                for(int i =0; i < N1; ++i)
                {

                    while(c < (int)bp[i])
                    {
                        vec2.push_back(c++);
                    }
                    ++c;
                }
                for(int i = (int)(bp[N1-1]+1); i < N2; ++i)
                {
                    vec2.push_back(i);
                }
                c = (int)vec2.size();
                for(int i = 0; i < c; ++i)
                {
                    ++depth;
                    perm1[depth] = N1+i;
                    perm2[depth] = vec2[i];
                    currentD = computeDist(perm1,perm2,currentD,depth);
                }
                
                for(int i = 0; i < c; ++i)
                {
                    perm1[depth] = -1;
                    perm2[depth] = -1;
                    depth--;
                }
            }
            
            if(min_dist > currentD)
            {
                min_dist = currentD;
                min_perm1 = perm1;
                min_perm2 = perm2;
            }
            
#if defined (DO_MAX_CUT)
    #if DO_MAX_CUT == 1
            if(fevals(depth-1)>=(uint32_t)MAX_EVALS){
                opt_dist = min_dist;
            }
    #endif
#endif
            return;
        }
        
		list.clearOverlaps(root.first, root.second);

        long nK = list.topK(n_branch, all_moves, num_all);
        sortAllMoves(nK, perm1, perm2, currentD, depth+1);
        fevals(depth) += nK;
        int start = (int)num_all;
        num_all += nK;

        for(int i = start; i < num_all; ++i)
        {
            // already worse than the best matching
            if(all_move_dists[i].first > min_dist)
                continue;
            
            recursiveHelper(depth+1,
                            all_move_dists[i].first,
                            all_moves[all_move_dists[i].second],
                            list, perm1, perm2);

            if(min_dist == opt_dist)
                return;
        }
        num_all -= nK;
        list.relinkStack(depth);
	}
	
    void sortAllMoves(long nK, arma::uvec& perm1, arma::uvec& perm2, const double curD, int depth)
    {
        int start = (int)num_all;
        int end = (int)(num_all+nK);
        for(int i = start; i < end; ++i)
        {
            perm1[depth] = all_moves.at(i).first;
            perm2[depth] = all_moves.at(i).second;
            all_move_dists[i].first = computeDist(perm1,perm2,curD, depth);
            all_move_dists[i].second = i;
        }
        std::sort(all_move_dists.begin()+start, all_move_dists.begin()+end,
                  [](MoveIndex a, MoveIndex b) -> bool
                  {
                      return a.first < b.first;
                  });
        perm1[depth] = 0;
        perm2[depth] = 0;
    }
    
	void sortMoves(long nK, const std::vector<GridPoint>& moves, std::vector<MoveIndex>& move_dists, arma::uvec& perm1, arma::uvec& perm2, const double curD, int depth)
	{
		for(unsigned int i = 0; i < nK; ++i)
		{
			perm1[depth] = moves.at(i).first;
			perm2[depth] = moves.at(i).second;
            move_dists[i].first = computeDist(perm1,perm2,curD, depth);
            move_dists[i].second = i;
		}
		std::sort(move_dists.begin(), move_dists.begin()+nK,
			[](MoveIndex a, MoveIndex b) -> bool
				{
					return a.first < b.first;
				});
        perm1[depth] = 0;
        perm2[depth] = 0;
	}

    // NOTE: uses symmetry of the underlying M1/M2 matrices to speed up the process
    // WILL NOT WORK FOR NON-SYMMETRIC MATRICES
	double computeDist(const arma::uvec& p1, const arma::uvec& p2, const double curD, int depth)
	{
        double d = curD;
        int mdepth = depth-1;
        for(int i = 0; i <= mdepth; ++i)
        {
            d += 2*std::abs(M1_t(p1(depth),p1(i))-M2(p2(depth),p2(i)));
        }
        d += std::abs(M1_t(p1(depth),p1(depth)) - M2(p2(depth),p2(depth)));

        return d;
	}

public:
	DFSAlign(const arma::mat& m1, const arma::mat& m2, int branch = 1) 
	{
		init(m1, m2, branch);
		reset();
	}

	void init(const arma::mat& m1, const arma::mat& m2, int branch)
	{
		if(m1.n_rows <= m2.n_rows)
		{
			M1 = m1;
			M2 = m2;
		}
		else
		{
			M1 = m2;
			M2 = m1;
		}
		N1 = M1.n_rows;
		N2 = M2.n_rows;
		
		// setting up number for normalization
		Nelem = N2*(N2-1);
		if(Nelem == 0) 
			Nelem = 1;
		// optimal distance
		M1_t = M1;
		M1_t.resize(N2,N2);
        
		arma::vec av = arma::vectorise(M1_t);
		av = arma::sort(av,"descend");
		arma::vec bv = arma::vectorise(M2);
		bv = arma::sort(bv,"descend");
		opt_dist = arma::norm(av-bv,1);
		if(N1 == 1 || N2 == 1)
		{
			min_dist = opt_dist;
			isAligned = true;
			return;
		}

		av = arma::sum(M1,1);
		av = av/arma::norm(av,2);
		bv = arma::sum(M2,1);
		bv = bv/arma::norm(bv,2);

		score = av * (bv.t());
		full_list.buildFromMat(score);
		
		n_branch = branch;
		fevals = arma::uvec(N2,arma::fill::zeros);
        
        all_moves.resize(N2*N2*n_branch);
        all_move_dists.resize(N2*N2*n_branch);
        
        num_all = 0;
		isAligned = false;
	}
	
	void reset(int branch = -1)
	{
		isAligned = false;
		
		min_dist = L11_norm(M1_t,M2);

		fill_range(min_perm1,N2);
		fill_range(min_perm2,N2);	

		if(branch != -1)
        {
			n_branch = branch;
            all_moves.resize(N2*N2*2);
            all_move_dists.resize(N2*N2*n_branch);
        }
	}

	void align()
	{
		reset();

		int depth = 0;
		// grab the top BRANCH score pairs
		GridDLL current_list(full_list);
			
        int nK = (int)current_list.topK(n_branch,all_moves,0);
        num_all = nK;
        
		// loop over the top values
        arma::uvec perm1(N2,arma::fill::ones); perm1 *= -1;
        arma::uvec perm2(N2,arma::fill::ones); perm2 *= -1;

		for(int i = 0; i < nK; ++i)
		{
            if(min_dist == opt_dist)
                break;
            recursiveHelper(depth, 0, all_moves.at(i), current_list, perm1, perm2);
		}
		
		// generate permutation
		min_perm.zeros(N1,N2);
		for(int i=0; i < N1; ++i)
		{
			min_perm( min_perm1(i), min_perm2(i) ) = 1;
		}
		min_dist = arma::norm(arma::vectorise(min_perm.t() * M1 * min_perm - M2),1);
		isAligned = true;
	}
	
	double getDistance(arma::mat& perm)
	{
		double d = getDistance();
		perm = min_perm;
		return d;
	}

	double getDistance()
	{
		if(!isAligned)
			align();
		return min_dist;
	}
	
	double getNormalizedDistance()
	{
		if(!isAligned)
			align();
		return min_dist / Nelem;
	}

	arma::uvec getFEvals()
	{
		return fevals;
	}
    
    unsigned long long getTotalEvals()
    {
        return arma::sum(fevals);
    }
};


double
dist_dfs(arma::mat& a, arma::mat& b, uint branches)
{
	DFSAlign aligner(a, b, branches);
	return aligner.getNormalizedDistance();
}

#endif
