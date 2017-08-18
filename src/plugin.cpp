/****
*
* Plugin binding
*
*
****/

#include "pyarma.h"
#include "DFSAlign.hpp"
#include "Distance.hpp"
#include "DMap.h"
#include "Nystrom.hpp"
#include "KMedoids.hpp"
#include "Heuristic.h"

#if defined(ENABLE_OPENMP)
#include <omp.h>
uint get_omp_threads() { 
	uint ret=0;
	#pragma omp parallel
	{ 
		#pragma omp single
		{ ret = omp_get_num_threads(); }	
	}
	return ret; 
}
void set_omp_threads(uint n) { omp_set_num_threads(n); }
#else
void set_omp_threads(uint n) { }
uint get_omp_threads() { return 1; }
#endif


PYBIND11_PLUGIN(PyDMap) {
	
	py::module m("PyDMap", "C++ implementation of DMap using armadillo");
    
    py::enum_<MatchHeuristic>(m, "MatchHeuristic")
        .value("ISORANK", MatchHeuristic::ISORANK)
        .value("SIGNED_ISORANK", MatchHeuristic::SIGNED_ISORANK)
        .export_values();

	m.def("dist",&dist, py::arg("a"), py::arg("b"), py::arg("heur")=MatchHeuristic::ISORANK);
	m.def("pdist",&pdist, py::arg("graphs"), py::arg("heur")=MatchHeuristic::ISORANK);
	m.def("pdist2",(pyarr_d(*)(std::vector<pyarr_d>,std::vector<pyarr_d>,MatchHeuristic))&pdist2, py::arg("a"), py::arg("b"), py::arg("heur")=MatchHeuristic::ISORANK);
	m.def("pdist2",(pyarr_d(*)(pyarr_d,std::vector<pyarr_d>,MatchHeuristic))&pdist2, py::arg("a"), py::arg("b"), py::arg("heur")=MatchHeuristic::ISORANK);
	m.def("pdist2",(pyarr_d(*)(std::vector<pyarr_d>,pyarr_d,MatchHeuristic))&pdist2, py::arg("a"), py::arg("b"), py::arg("heur")=MatchHeuristic::ISORANK);

	m.def("kmedoids",&kmedoids,py::arg("dists"),py::arg("k"),py::arg("seeds")=pyarr_u(0,nullptr));
    

	py::class_<DMap>(m,"DMap")
		.def(py::init<>())
		.def("compute",&DMap::compute)
		.def("set_epsilon",&DMap::set_epsilon)
		.def("get_epsilon",&DMap::get_epsilon)
		.def("set_num_evec",&DMap::set_num_evec)
		.def("get_num_evec",&DMap::get_num_evec)
		.def("set_dists",&DMap::set_dists)
		.def("get_evec",&DMap::get_evec_py)
		.def("set_evec",&DMap::set_evec)
		.def("get_eval",&DMap::get_eval_py)
		.def("set_eval",&DMap::set_eval)
		.def("get_num_samples",&DMap::get_num_samples);

	m.def("nystrom",(pyarr_d(*)(DMap&,pyarr_d))&nystrom);

	m.def("set_omp_threads",&set_omp_threads);
	m.def("get_omp_threads",&get_omp_threads);
	return m.ptr();
}

