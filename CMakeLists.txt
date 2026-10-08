cmake_minimum_required(VERSION 3.14)
project(cellmate LANGUAGES CXX)

add_library(cellmate INTERFACE)
add_library(cellmate::cellmate ALIAS cellmate)
target_include_directories(cellmate INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(cellmate INTERFACE cxx_std_14)

option(CELLMATE_ENABLE_OPENMP "Enable OpenMP for cellmate consumers" OFF)
if(CELLMATE_ENABLE_OPENMP)
	find_package(OpenMP REQUIRED COMPONENTS CXX)
	target_link_libraries(cellmate INTERFACE OpenMP::OpenMP_CXX)
endif()

option(CELLMATE_BUILD_BENCHMARKS "Build the cellmate performance benchmarks" OFF)
if(CELLMATE_BUILD_BENCHMARKS)
	add_subdirectory(benchmarks)
endif()

option(CELLMATE_BUILD_EXAMPLES "Build the cellmate usage examples" OFF)
if(CELLMATE_BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()

option(CELLMATE_BUILD_TESTS "Build the cellmate correctness tests" OFF)
if(CELLMATE_BUILD_TESTS)
	add_subdirectory(tests)
endif()
