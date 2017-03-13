//basis for armadillo python bindings taken from
//https://github.com/antonl/varpro-blocks/
#pragma once

#include <armadillo>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


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
pyarr_d mat_to_py(arma::mat &mat)
{
	py::buffer_info buffer(
		mat.memptr(),
		sizeof(double),
		py::format_descriptor<double>::format(),
		2,
		{ mat.n_rows, mat.n_cols },
		{ sizeof(double), sizeof(double) * mat.n_rows }
	);
	return pyarr_d(buffer);
}

inline
arma::uvec py_to_uvec(pyarr_u &pvec)
{
	py::buffer_info info = pvec.request();
	arma::uvec avec(reinterpret_cast<arma::uword*>(info.ptr),info.shape[0]);
	return avec;
}


inline 
pyarr_u uvec_to_py(arma::uvec &vec)
{
	py::buffer_info buffer(
		vec.memptr(),
		sizeof(arma::uword),
		py::format_descriptor<arma::uword>::format(),
		1,
		{ vec.n_elem },
		{ sizeof(arma::uword) }
	);
	return pyarr_u(buffer);
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
pyarr_d vec_to_py(arma::vec &vec)
{
	py::buffer_info buffer(
		vec.memptr(),
		sizeof(double),
		py::format_descriptor<double>::format(),
		1,
		{ vec.n_elem },
		{ sizeof(double) }
	);
	return pyarr_d(buffer);
}

/*
inline py::buffer_info vec_buffer(arma::vec &m)
{
	py::buffer_info buf(
		m.memptr(),
		sizeof(arma::vec::elem_type),
		py::format_descriptor<arma::vec::elem_type>::format(),
		1,
		{m.n_rows,},
		{sizeof(arma::vec::elem_type),}
	);
	return buf;
}


inline void vec_np_init(arma::vec &v, py::array& b)
{
	py::buffer_info info = b.request();
	if(info.format != py::format_descriptor<arma::vec::elem_type>::format() || info.ndim != 1) {
		throw std::runtime_error("incompatible buffer format");
	}

	if(info.strides[0] == info.itemsize) {
		new (&v) arma::vec(reinterpret_cast<arma::vec::elem_type*>(info.ptr),info.shape[0]);
	} else {
		throw std::runtime_error("array not contiguous");
	}
}

inline py::buffer_info mat_buffer(arma::mat &m)
{
	py::buffer_info buf(
		m.memptr(),
		sizeof(arma::mat::elem_type),
		py::format_descriptor<arma::mat::elem_type>::format(),
		2,
		{m.n_rows,m.n_cols},
		{sizeof(arma::mat::elem_type),sizeof(arma::mat::elem_type)*m.n_rows}
	);
	return buf;
}

inline void mat_np_init(arma::mat &m, py::array& b)
{
	py::buffer_info info = b.request();
	if(info.format != py::format_descriptor<arma::mat::elem_type>::format())
		throw std::runtime_error("incompatible buffer format");

	if(info.ndim == 1) {
	// Vector
		new (&m) arma::mat(reinterpret_cast<arma::mat::elem_type*>(info.ptr),info.shape[0],1);
	} else if((info.strides[0] == info.itemsize) && (info.strides[1] == (info.itemsize*info.shape[0]))) {
	// R-contiguous
		new (&m) arma::mat(reinterpret_cast<arma::mat::elem_type*>(info.ptr),info.shape[0],info.shape[1]);
	} else if((info.strides[1] == info.itemsize) && (info.strides[0] == (info.itemsize*info.shape[1]))) {
	// C-contiguous
		new (&m) arma::mat(reinterpret_cast<arma::mat::elem_type*>(info.ptr),info.shape[1],info.shape[0]);
		arma::inplace_trans(m);
	} else {
		throw std::runtime_error("array not contiguous");
	}
}
*/
