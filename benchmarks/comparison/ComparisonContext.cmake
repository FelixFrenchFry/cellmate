# keep revision metadata current without recompiling unchanged builds
set(comparison_commit "unknown")
if(EXISTS "${COMPARISON_GIT_EXECUTABLE}" AND EXISTS "${COMPARISON_SOURCE_DIR}/.git")
	execute_process(COMMAND "${COMPARISON_GIT_EXECUTABLE}" rev-parse --short=12 HEAD
		WORKING_DIRECTORY "${COMPARISON_SOURCE_DIR}"
		RESULT_VARIABLE result OUTPUT_VARIABLE revision OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
	if(result EQUAL 0 AND NOT revision STREQUAL "")
		set(comparison_commit "${revision}")
	endif()
endif()
string(REPLACE "\\" "\\\\" comparison_commit "${comparison_commit}")
string(REPLACE "\"" "\\\"" comparison_commit "${comparison_commit}")
file(CONFIGURE OUTPUT "${COMPARISON_REVISION_HEADER}" CONTENT [=[
#pragma once
#define CELLMATE_COMPARISON_COMMIT "@comparison_commit@"
]=] @ONLY)
