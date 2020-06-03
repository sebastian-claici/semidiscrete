CMAKE_MINIMUM_REQUIRED(VERSION 3.1)
PROJECT(semidiscrete)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fopenmp -O3 -ffast-math")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)
set(CMAKE_LibRARY_OUTPUT_DIRECTORY lib)

include_directories(${PROJECT_SOURCE_DIR}/include)

# Eigen for linear algebra
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(test-semidiscrete examples/semidiscrete.cpp)
target_link_libraries(test-semidiscrete LINK_PUBLIC Eigen3::Eigen)
