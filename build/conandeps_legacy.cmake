message(STATUS "Conan: Using CMakeDeps conandeps_legacy.cmake aggregator via include()")
message(STATUS "Conan: It is recommended to use explicit find_package() per dependency instead")

find_package(melon)
find_package(pybind11)

set(CONANDEPS_LEGACY  melon::melon  pybind11_all_do_not_use )