cmake_minimum_required(VERSION 3.18)

# fetch Google test library, without its CTest setup or install rules
include(FetchContent)
if(POLICY CMP0135)
	cmake_policy(SET CMP0135 NEW)
endif()

set(GOOGLETEST_VERSION 1.18.0)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(gtest_build_tests OFF CACHE BOOL "" FORCE)
set(gtest_build_samples OFF CACHE BOOL "" FORCE)
if(MSVC)
	set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()
FetchContent_Declare(googletest
	URL https://codeload.github.com/google/googletest/tar.gz/refs/tags/v${GOOGLETEST_VERSION}
	URL_HASH SHA256=6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5
	SOURCE_SUBDIR googletest
)
FetchContent_MakeAvailable(googletest)

# build the correctness tests
add_executable(cellmate_tests
	test_index_compaction.cpp
	test_rebuild.cpp
	test_query.cpp
	test_query_batch.cpp
	test_point_storage.cpp
)
target_link_libraries(cellmate_tests PRIVATE cellmate::cellmate GTest::gtest_main)
target_compile_features(cellmate_tests PRIVATE cxx_std_17)

# exercise benchmark tooling only when the maintained suite is enabled
if(CELLMATE_BUILD_BENCHMARKS)
	target_sources(cellmate_tests PRIVATE
		test_benchmark_config.cpp
		../benchmarks/bench_prepare.cpp
		../benchmarks/scenarios/uniform_volume.cpp
		../benchmarks/scenarios/anisotropic_volume.cpp
		../benchmarks/scenarios/planar_surface.cpp
		../benchmarks/scenarios/gradient_volume.cpp
		../benchmarks/scenarios/clustered_volume.cpp
		../benchmarks/scenarios/lattice_volume.cpp
		../benchmarks/scenarios/spherical_surface.cpp
	)
endif()

# optional CPU target flags for backend test builds
if(CELLMATE_TEST_ARCH_FLAGS)
	target_compile_options(cellmate_tests PRIVATE ${CELLMATE_TEST_ARCH_FLAGS})
endif()

# verify that cellmate preserves strict floating-point settings in consumers
if(MSVC)
	target_compile_options(cellmate_tests PRIVATE /fp:strict)
endif()
