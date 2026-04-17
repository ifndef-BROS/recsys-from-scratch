cmake_minimum_required(VERSION 3.10)
project(Recommender)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(main src/main.cpp)
target_include_directories(main PRIVATE include)
