# cmake_minimum_required(VERSION 3.15)

# project(MyProject)

# set(CMAKE_CXX_STANDARD 17)

# find_package(fmt REQUIRED)
# find_package(range REQUIRED)
# add_subdirectory(dependencies/melon)

# add_executable(MyProject main.cpp)

# target_include_directories(MyProject PRIVATE ${FMT_INCLUDE_DIR})
# target_link_libraries(MyProject PRIVATE INTERFACE melon)
cmake_minimum_required(VERSION 3.15)
project(compressor)

find_package(ZLIB REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ZLIB::ZLIB)