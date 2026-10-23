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
FetchContent_Declare(picotree
	URL https://codeload.github.com/Jaybro/pico_tree/tar.gz/refs/tags/v1.0.0
	URL_HASH SHA256=4a36a8fe862a26716c084242e9e4547524ec73104a81797a43f4086db8530fa5
	SOURCE_SUBDIR src/pico_tree/pico_tree
)
FetchContent_Declare(octree
	URL https://codeload.github.com/jbehley/octree/tar.gz/8e3927d48d5ce61f94aab6090d77b4434f87dc89
	URL_HASH SHA256=829dc32d8cd7bda7fd86f4a480091df098c02615c8f70e36784d9940ebe178eb
	SOURCE_SUBDIR include
)
FetchContent_Declare(compactnsearch
	URL https://codeload.github.com/InteractiveComputerGraphics/CompactNSearch/tar.gz/b40afcf47fe1963b363eba2371f04b42720fcb1d
	URL_HASH SHA256=2bfe7c917bc14ad410ca41d951afe303c1822d13cee80d9846bfb9d6feb802d6
	SOURCE_SUBDIR include
)
FetchContent_Declare(ponca
	URL https://codeload.github.com/poncateam/ponca/tar.gz/refs/tags/v1.4
	URL_HASH SHA256=bcd6d16efc66445de10aa661131741f239a386e353251c9336b3ea15c33290a1
	SOURCE_SUBDIR Ponca
)
FetchContent_Declare(eigen
	URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
	URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
	SOURCE_SUBDIR Eigen
)
FetchContent_Declare(cgal
	URL https://github.com/CGAL/cgal/releases/download/v5.6.2/CGAL-5.6.2.tar.xz
	URL_HASH SHA256=458f60df8e8f1f2fdad93c8f24e1aa8f4b095cc61a14fac81b90680d7306a42e
	SOURCE_SUBDIR include
)
FetchContent_Declare(boost
	URL https://archives.boost.io/release/1.86.0/source/boost_1_86_0.tar.gz
	URL_HASH SHA256=2575e74ffc3ef1cd0babac2c1ee8bdb5782a0ee672b1912da40e5b4b591ca01f
	SOURCE_SUBDIR boost
)
FetchContent_MakeAvailable(googlebenchmark nanoflann picotree octree compactnsearch ponca eigen cgal boost)

# optionally fetch TreeNSearch, which compiles its own sources with AVX2
option(CELLMATE_COMPARISON_TREENSEARCH "Include TreeNSearch in the library comparison (requires AVX2)" ON)
if(CELLMATE_COMPARISON_TREENSEARCH)
	FetchContent_Declare(treensearch
		URL https://codeload.github.com/InteractiveComputerGraphics/TreeNSearch/tar.gz/61cf111f65c01f1d71fb5023ac72ce6bb0e8d6f9
		URL_HASH SHA256=33770430cb9fd208513cb7c7d3c321d37edd28d374af97d2199aa01c4f67ba33
		SOURCE_SUBDIR TreeNSearch/source
	)
	FetchContent_MakeAvailable(treensearch)

	# compile only the TreeNSearch implementation, with AVX2 on this target alone
	find_package(Threads REQUIRED)
	add_library(cellmate_comparison_treensearch STATIC "${treensearch_SOURCE_DIR}/TreeNSearch/source/TreeNSearch.cpp")
	target_include_directories(cellmate_comparison_treensearch SYSTEM PUBLIC "${treensearch_SOURCE_DIR}/TreeNSearch/source")
	target_include_directories(cellmate_comparison_treensearch SYSTEM PRIVATE "${treensearch_SOURCE_DIR}/TreeNSearch/extern")
	target_compile_features(cellmate_comparison_treensearch PRIVATE cxx_std_17)
	target_compile_options(cellmate_comparison_treensearch PRIVATE $<IF:$<CXX_COMPILER_ID:MSVC>,/arch:AVX2,-mavx2>)
	target_link_libraries(cellmate_comparison_treensearch PUBLIC OpenMP::OpenMP_CXX PRIVATE Threads::Threads)
endif()

# compile only the CompactNSearch implementation, with its default float type and OpenMP
add_library(cellmate_comparison_compactnsearch STATIC "${compactnsearch_SOURCE_DIR}/src/CompactNSearch.cpp")
target_include_directories(cellmate_comparison_compactnsearch SYSTEM PUBLIC "${compactnsearch_SOURCE_DIR}/include")
target_link_libraries(cellmate_comparison_compactnsearch PRIVATE OpenMP::OpenMP_CXX)

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
target_include_directories(cellmate_comparison SYSTEM PRIVATE
	"${nanoflann_SOURCE_DIR}/include"
	"${picotree_SOURCE_DIR}/src/pico_tree"
	"${octree_SOURCE_DIR}"
	"${ponca_SOURCE_DIR}"
	"${eigen_SOURCE_DIR}"
	"${cgal_SOURCE_DIR}/include"
	"${boost_SOURCE_DIR}"
)
target_link_libraries(cellmate_comparison PRIVATE cellmate::cellmate benchmark::benchmark cellmate_comparison_compactnsearch)
target_compile_features(cellmate_comparison PRIVATE cxx_std_17)

# record the measured commit, compiler and flags in the benchmark context
# encode flags as C++ byte escapes, so quotes and build-tool characters stay data
function(comparison_encode_flags result flags)
	string(HEX "${flags}" flags)
	string(REGEX REPLACE "(..)" "\\\\x\\1" flags "${flags}")
	set(${result} "${flags}" PARENT_SCOPE)
endfunction()
comparison_encode_flags(comparison_flags "${CMAKE_CXX_FLAGS}")
if(CMAKE_CONFIGURATION_TYPES)
	set(comparison_configs ${CMAKE_CONFIGURATION_TYPES})
else()
	set(comparison_configs "${CMAKE_BUILD_TYPE}")
endif()
foreach(config IN LISTS comparison_configs)
	string(TOUPPER "${config}" config_upper)
	comparison_encode_flags(config_flags " ${CMAKE_CXX_FLAGS_${config_upper}}")
	string(APPEND comparison_flags "$<$<CONFIG:${config}>:${config_flags}>")
endforeach()
target_compile_definitions(cellmate_comparison PRIVATE
	CELLMATE_COMPARISON_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
	CELLMATE_COMPARISON_CONFIG="$<CONFIG>"
	"CELLMATE_COMPARISON_FLAGS=\"${comparison_flags}\""
)

# refresh the revision on every build, writing the header only when it changes
find_package(Git QUIET)
set(comparison_revision_header "${CMAKE_CURRENT_BINARY_DIR}/comparison_revision.hpp")
add_custom_target(cellmate_comparison_revision
	COMMAND "${CMAKE_COMMAND}"
		"-DCOMPARISON_GIT_EXECUTABLE=${GIT_EXECUTABLE}"
		"-DCOMPARISON_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
		"-DCOMPARISON_REVISION_HEADER=${comparison_revision_header}"
		-P "${CMAKE_CURRENT_LIST_DIR}/ComparisonContext.cmake"
	BYPRODUCTS "${comparison_revision_header}"
	VERBATIM
)
add_dependencies(cellmate_comparison cellmate_comparison_revision)
target_include_directories(cellmate_comparison PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

if(CELLMATE_COMPARISON_TREENSEARCH)
	target_compile_definitions(cellmate_comparison PRIVATE CELLMATE_COMPARISON_TREENSEARCH)
	target_link_libraries(cellmate_comparison PRIVATE cellmate_comparison_treensearch)
endif()
