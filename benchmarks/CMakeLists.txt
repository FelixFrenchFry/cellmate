cmake_minimum_required(VERSION 3.18)

# fetch Google benchmark library, without its tests or install rules
include(FetchContent)
if(POLICY CMP0135)
	cmake_policy(SET CMP0135 NEW)
endif()

set(GOOGLEBENCHMARK_VERSION 1.9.4)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(googlebenchmark
	URL https://codeload.github.com/google/benchmark/tar.gz/refs/tags/v${GOOGLEBENCHMARK_VERSION}
	URL_HASH SHA256=b334658edd35efcf06a99d9be21e4e93e092bd5f95074c1673d5c8705d95c104
)
FetchContent_MakeAvailable(googlebenchmark)

# build the performance benchmarks
add_executable(cellmate_benchmarks bench_cellmate.cpp)
target_link_libraries(cellmate_benchmarks PRIVATE cellmate::cellmate benchmark::benchmark_main)
target_compile_features(cellmate_benchmarks PRIVATE cxx_std_17)
