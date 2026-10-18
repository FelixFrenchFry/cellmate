# fail if CMake selects an unexpected compiler or version series, but allow patch updates
message(STATUS "CI compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CI compiler path: ${CMAKE_CXX_COMPILER}")

string(FIND "${CMAKE_CXX_COMPILER_VERSION}." "${CELLMATE_CI_COMPILER_VERSION}." compiler_version_match)
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL CELLMATE_CI_COMPILER_ID OR NOT compiler_version_match EQUAL 0)
	message(FATAL_ERROR
		"Expected ${CELLMATE_CI_COMPILER_ID} ${CELLMATE_CI_COMPILER_VERSION}.x, "
		"got ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
endif()
