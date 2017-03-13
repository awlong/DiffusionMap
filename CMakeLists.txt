cmake_minimum_required(VERSION 2.6)
project(structure)

set(CMAKE_MODULE_PATH
	"${CMAKE_CURRENT_SOURCE_DIR}/CMake"
	"${CMAKE_MODULE_PATH}"
	)

message("cmake_module_path: ${CMAKE_MODULE_PATH}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -Wall")

include(CommonLibs)
include(ArmadilloSetup)
include(PythonSetup)

message(STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})

include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/pybind11/include)
include_directories(${PROJECT_SOURCE_DIR}/armadillo/include)

file(GLOB SOURCES "src/*.cpp")

add_library(PyDMap SHARED
	${SOURCES}
	)

target_link_libraries(PyDMap ${LAPACK_LIBRARIES})

set_target_properties(PyDMap PROPERTIES PREFIX "")
set_target_properties(PyDMap PROPERTIES SUFFIX ".so")

