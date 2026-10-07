# thread support for concurrent-query tests
find_package(Threads REQUIRED)

# build the correctness tests
add_executable(cellmate_tests test_search.cpp)
target_link_libraries(cellmate_tests PRIVATE cellmate::cellmate Threads::Threads)

# register the executable with CTest
add_test(NAME cellmate_tests COMMAND cellmate_tests)
