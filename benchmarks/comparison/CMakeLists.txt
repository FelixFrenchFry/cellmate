cmake_minimum_required(VERSION 3.18)

# every profile compares sequential and parallel runs
if(NOT CELLMATE_ENABLE_OPENMP)
	message(FATAL_ERROR "the library comparison requires CELLMATE_ENABLE_OPENMP=ON")
endif()

# fetch Google benchmark library, without its tests or install rules
# (shares the pinned version with the maintained benchmarks)
include(FetchContent)
if(POLICY CMP0135)
	cmake_policy(SET CMP0135 NEW)
endif()
include("${CMAKE_CURRENT_LIST_DIR}/../GoogleBenchmark.cmake")

# fetch the compared libraries at pinned versions, without their examples, tests or install rules
# (a 'SOURCE_SUBDIR' without CMakeLists.txt only downloads the sources)
FetchContent_Declare(nanoflann
	URL https://codeload.github.com/jlblancoc/nanoflann/tar.gz/refs/tags/v1.10.0
	URL_HASH SHA256=b8ce3d4d4051a62a5ab68e0b1da54fde466f655c3e8d52ead5c470812c45f202
	SOURCE_SUBDIR include
)
FetchContent_MakeAvailable(googlebenchmark nanoflann)

# build the library comparison, reusing the workloads of the maintained benchmarks
add_executable(cellmate_comparison
	bench_comparison.cpp
	../bench_prepare.cpp
	../scenarios/uniform_volume.cpp
	../scenarios/anisotropic_volume.cpp
	../scenarios/planar_surface.cpp
	../scenarios/gradient_volume.cpp
	../scenarios/clustered_volume.cpp
	../scenarios/lattice_volume.cpp
	../scenarios/spherical_surface.cpp
)
target_include_directories(cellmate_comparison SYSTEM PRIVATE "${nanoflann_SOURCE_DIR}/include")
target_link_libraries(cellmate_comparison PRIVATE cellmate::cellmate benchmark::benchmark)
target_compile_features(cellmate_comparison PRIVATE cxx_std_17)
