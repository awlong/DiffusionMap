//
//  pyarma.h
//  binding between armadillo and numpy using pybind11
//
//  Updated by Andrew Long on 08/22/17.
//  Copyright (c) 2016-2017 Andrew Long. All rights reserved.
//
//  Utilizes Armadillo (http://arma.sourceforge.net/) C++ LinAlg Library
//  Sanderson, C. Armadillo: An Open Source C++ Linear Algebra Library
//  for Fast Prototyping and Computationally Intensive Experiments.
//  Technical Report, NICTA, 2010.
//
//  basis for armadillo python bindings taken from
//  https://github.com/antonl/varpro-blocks/
#pragma once

#include <armadillo>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

static void warn(std::string file, int line, std::string msg)
{
	std::cout << "[" << file << ":" << line << "] " << msg << std::endl;
}

#define WARNING(msg) warn(__FILE__,__LINE__,msg)

namespace py=pybind11;
typedef py::array_t<double, py::array::f_style | py::array::forcecast> pyarr_d;
typedef py::array_t<arma::uword, py::array::forcecast> pyarr_u;

inline
arma::mat py_to_mat(pyarr_d& pmat)
{
	py::buffer_info info = pmat.request();
	arma::mat amat;
	if(info.ndim == 1) {
		amat = arma::mat(reinterpret_cast<double*>(info.ptr),info.shape[0],1);
	} else {
		amat = arma::mat(reinterpret_cast<double*>(info.ptr),info.shape[0],info.shape[1]);
	}
	return amat;
}

inline
py::buffer_info mat_buffer(arma::mat &mat)
{
	py::buffer_info buffer(
		mat.memptr(),
		sizeof(double),
		py::format_descriptor<double>::format(),
		2,
		{ mat.n_rows, mat.n_cols },
		{ sizeof(double), sizeof(double) * mat.n_rows }
	);
    return buffer;
}

inline
pyarr_d mat_to_py(arma::mat &mat)
{
	return pyarr_d(mat_buffer(mat));
}

inline
arma::uvec py_to_uvec(pyarr_u &pvec)
{
	py::buffer_info info = pvec.request();
	arma::uvec avec(reinterpret_cast<arma::uword*>(info.ptr),info.shape[0]);
	return avec;
}


inline 
py::buffer_info uvec_buffer(arma::uvec &vec)
{
	py::buffer_info buffer(
		vec.memptr(),
		sizeof(arma::uword),
		py::format_descriptor<arma::uword>::format(),
		1,
		{ vec.n_elem },
		{ sizeof(arma::uword) }
	);
    return buffer;
}

inline
pyarr_u uvec_to_py(arma::uvec &vec)
{
	return pyarr_u(uvec_buffer(vec));
}

inline
arma::vec py_to_vec(pyarr_d &pvec)
{
	py::buffer_info info = pvec.request();
	if(info.ndim != 1) {
		throw std::runtime_error("py_to_vec encountered non 1D numpy array");
	}
	arma::vec avec(reinterpret_cast<double*>(info.ptr),info.shape[0]);
	return avec;
}

inline
py::buffer_info vec_buffer(arma::vec &vec)
{
	py::buffer_info buffer(
		vec.memptr(),
		sizeof(double),
		py::format_descriptor<double>::format(),
		1,
		{ vec.n_elem },
		{ sizeof(double) }
	);
    return buffer;
}

inline
pyarr_d vec_to_py(arma::vec &vec)
{
	return pyarr_d(vec_buffer(vec));
}

inline 
std::vector<arma::mat> convert_matlist(std::vector<pyarr_d> &p)
{
    size_t N = p.size();
    if(N == 0)
        throw std::runtime_error("Calling distance function with empty matrix");
    arma::mat base_obj = py_to_mat(p.at(0));
    std::vector<arma::mat> objs;
    // we have only a single object
    if(base_obj.n_rows == N && base_obj.n_cols == 1)
    {  
        // concatenate columns together
        for(size_t i = 1; i < N; ++i)
        {
            arma::mat tmp = py_to_mat(p.at(i));
            base_obj = arma::join_rows(base_obj, tmp);
        }
        objs.push_back(base_obj);
    }
    else
    {
        objs.reserve(N);
        // multiple objects in list
        for(size_t i = 0; i < N; ++i)
        {
            arma::mat tmp = py_to_mat(p.at(i));
            objs.push_back(tmp);
        }
    }
    return objs;
}
