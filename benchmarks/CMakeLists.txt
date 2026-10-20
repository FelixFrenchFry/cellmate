cmake_minimum_required(VERSION 3.18)

# fetch Google benchmark library, without its tests or install rules
include(FetchContent)
if(POLICY CMP0135)
	cmake_policy(SET CMP0135 NEW)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/GoogleBenchmark.cmake")
FetchContent_MakeAvailable(googlebenchmark)

# build the performance benchmarks
add_executable(cellmate_benchmarks
	bench_cellmate.cpp
	bench_prepare.cpp
	scenarios/uniform_volume.cpp
	scenarios/anisotropic_volume.cpp
	scenarios/planar_surface.cpp
	scenarios/gradient_volume.cpp
	scenarios/clustered_volume.cpp
	scenarios/lattice_volume.cpp
	scenarios/spherical_surface.cpp
)
target_link_libraries(cellmate_benchmarks PRIVATE cellmate::cellmate benchmark::benchmark)
target_compile_features(cellmate_benchmarks PRIVATE cxx_std_17)
