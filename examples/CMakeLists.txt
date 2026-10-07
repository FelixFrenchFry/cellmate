# build the usage example
add_executable(cellmate_example basic.cpp)
target_link_libraries(cellmate_example PRIVATE cellmate::cellmate)
