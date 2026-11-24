cmake_minimum_required(VERSION 3.16)
# C is enabled only because SuiteSparse 7's package config looks for OpenMP's C component.
project(trajectory_ba C CXX)
find_package(Ceres REQUIRED)
add_executable(trajectory_ba bundle.cpp)
target_compile_features(trajectory_ba PRIVATE cxx_std_17)
target_link_libraries(trajectory_ba PRIVATE Ceres::ceres)
