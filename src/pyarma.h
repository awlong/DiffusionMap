//basis for armadillo python bindings taken from
//https://github.com/antonl/varpro-blocks/
#pragma once

#include <armadillo>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace py=pybind11;

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
