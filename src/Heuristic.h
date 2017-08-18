// Andrew Long
// Updated: Aug 18, 2017
// Breakout of Heuristic code from Alignment code

#ifndef __HEURISTIC__
#define __HEURISTIC__

#include <armadillo>
#include <map>
#include <pybind11/stl.h>
#include "pyarma.h"

enum class MatchHeuristic
{
    ISORANK,
    SIGNED_ISORANK
};

arma::mat isorank_score(const arma::mat& m1, const arma::mat& m2);
arma::mat signed_isorank_score(const arma::mat& m1, const arma::mat& m2);

typedef arma::mat (*heuristic_func)(const arma::mat&, const arma::mat&);

static std::map<MatchHeuristic, heuristic_func> heuristic_map = {
            { MatchHeuristic::ISORANK, isorank_score},
            { MatchHeuristic::SIGNED_ISORANK, signed_isorank_score}
       };

#endif
