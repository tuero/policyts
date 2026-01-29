cmake_minimum_required(VERSION 3.25)

project(policyts VERSION 0.9.0 LANGUAGES CXX)

# libpolicyts external lib has all the dependencies we need
find_package(libpolicyts CONFIG REQUIRED)

# Library source
add_subdirectory(src)

