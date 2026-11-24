cmake_minimum_required(VERSION 3.16)
# C is enabled only because SuiteSparse 7's package config looks for OpenMP's C component.
project(rig_calib C CXX)
find_package(Ceres REQUIRED)
add_executable(rig_ba ba.cpp)
target_compile_features(rig_ba PRIVATE cxx_std_17)
target_link_libraries(rig_ba PRIVATE Ceres::ceres)
