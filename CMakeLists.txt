cmake_minimum_required(VERSION 3.15)
project(MyProject LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Include the Conan-generated toolchain file
include(build/conan_toolchain.cmake)

# Set the project
# set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_BUILD_TYPE Release)

# Find the packages
find_package(melon REQUIRED)
find_package(fmt REQUIRED)
find_package(pybind11 REQUIRED)

message("EXECUTE=${EXECUTE}")
if( EXECUTE )
    message("MY_VARIABLE evaluates to False")
    # Add your targets
    add_executable(a.out main.cpp)
    # Link the libraries
    target_link_libraries(a.out melon::melon)
else()
    add_compile_definitions(VAR=\"myValue\")
    message("MY_VARIABLE evaluates to True")
    pybind11_add_module(MyProject main.cpp)
    # Link the libraries
    target_link_libraries(MyProject PRIVATE melon::melon fmt::fmt)
endif()
# # Add your targets
# add_executable(main main.cpp)


# cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Debug