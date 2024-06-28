########## MACROS ###########################################################################
#############################################################################################

# Requires CMake > 3.15
if(${CMAKE_VERSION} VERSION_LESS "3.15")
    message(FATAL_ERROR "The 'CMakeDeps' generator only works with CMake >= 3.15")
endif()

if(melon_FIND_QUIETLY)
    set(melon_MESSAGE_MODE VERBOSE)
else()
    set(melon_MESSAGE_MODE STATUS)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmakedeps_macros.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/melonTargets.cmake)
include(CMakeFindDependencyMacro)

check_build_type_defined()

foreach(_DEPENDENCY ${melon_FIND_DEPENDENCY_NAMES} )
    # Check that we have not already called a find_package with the transitive dependency
    if(NOT ${_DEPENDENCY}_FOUND)
        find_dependency(${_DEPENDENCY} REQUIRED ${${_DEPENDENCY}_FIND_MODE})
    endif()
endforeach()

set(melon_VERSION_STRING "0.5")
set(melon_INCLUDE_DIRS ${melon_INCLUDE_DIRS_RELEASE} )
set(melon_INCLUDE_DIR ${melon_INCLUDE_DIRS_RELEASE} )
set(melon_LIBRARIES ${melon_LIBRARIES_RELEASE} )
set(melon_DEFINITIONS ${melon_DEFINITIONS_RELEASE} )


# Only the last installed configuration BUILD_MODULES are included to avoid the collision
foreach(_BUILD_MODULE ${melon_BUILD_MODULES_PATHS_RELEASE} )
    message(${melon_MESSAGE_MODE} "Conan: Including build module from '${_BUILD_MODULE}'")
    include(${_BUILD_MODULE})
endforeach()


