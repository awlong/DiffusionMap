// Andrew Long
// Updated: Aug 18, 2017
// Breakout of Heuristic code from Alignment code

#include "Heuristic.h"

arma::mat
isorank_score(const arma::mat& m1, const arma::mat& m2)
{
    arma::vec rsum1 = arma::sum(m1, 1);
    if(arma::sum(rsum1) > 0)
        rsum1 = rsum1 / arma::norm(rsum1, 2);
    
    arma::vec rsum2 = arma::sum(m2, 1);
    if(arma::sum(rsum2) > 0)
        rsum2 = rsum2 / arma::norm(rsum2, 2);

    arma::mat score;
    if(m1.n_rows <= m2.n_rows)
        score = rsum1 * (rsum2.t());
    else
        score = rsum2 * (rsum1.t());
    return score;
}

arma::mat
signed_isorank_score(const arma::mat& m1, const arma::mat& m2)
{
    arma::mat pos1 = m1;
    pos1.for_each([](arma::mat::elem_type& val) 
                    { val = (val >= 0 ? val : 0); });
    
    arma::mat neg1 = m1;
    neg1.for_each([](arma::mat::elem_type& val) 
                    { val = (val <= 0 ? -val : 0); });
    
    arma::mat pos2 = m2;
    pos2.for_each([](arma::mat::elem_type& val) 
                    { val = (val >= 0 ? val : 0); });
    
    arma::mat neg2 = m2;
    neg2.for_each([](arma::mat::elem_type& val) 
                    { val = (val <= 0 ? -val : 0); });

    arma::mat score = isorank_score(pos1, pos2) + isorank_score(neg1, neg2);
    return score;
}

