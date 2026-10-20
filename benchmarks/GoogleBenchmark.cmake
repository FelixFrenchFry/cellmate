# declares the pinned Google benchmark library, without its tests or install rules
# (the only pin, shared by all benchmark executables, made available by the including CMakeLists.txt)
set(GOOGLEBENCHMARK_VERSION 1.9.4)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(googlebenchmark
	URL https://codeload.github.com/google/benchmark/tar.gz/refs/tags/v${GOOGLEBENCHMARK_VERSION}
	URL_HASH SHA256=b334658edd35efcf06a99d9be21e4e93e092bd5f95074c1673d5c8705d95c104
)
