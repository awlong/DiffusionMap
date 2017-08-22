//
//  Heuristic.h
//  Matching Heuristics for Graph Alignment
//
//  Updated by Andrew Long on 08/18/17.
//  Copyright (c) 2016-2017 Andrew Long. All rights reserved.
//
//  Utilizes Armadillo (http://arma.sourceforge.net/) C++ LinAlg Library
//  Sanderson, C. Armadillo: An Open Source C++ Linear Algebra Library
//  for Fast Prototyping and Computationally Intensive Experiments.
//  Technical Report, NICTA, 2010.

#ifndef __HEURISTIC__
#define __HEURISTIC__

#include <armadillo>
#include <map>
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
