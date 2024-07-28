# Avoid multiple calls to find_package to append duplicated properties to the targets
include_guard()########### VARIABLES #######################################################################
#############################################################################################
set(melon_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
conan_find_apple_frameworks(melon_FRAMEWORKS_FOUND_RELEASE "${melon_FRAMEWORKS_RELEASE}" "${melon_FRAMEWORK_DIRS_RELEASE}")

set(melon_LIBRARIES_TARGETS "") # Will be filled later


######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
if(NOT TARGET melon_DEPS_TARGET)
    add_library(melon_DEPS_TARGET INTERFACE IMPORTED)
endif()

set_property(TARGET melon_DEPS_TARGET
             APPEND PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Release>:${melon_FRAMEWORKS_FOUND_RELEASE}>
             $<$<CONFIG:Release>:${melon_SYSTEM_LIBS_RELEASE}>
             $<$<CONFIG:Release>:range-v3::range-v3;fmt::fmt>)

####### Find the libraries declared in cpp_info.libs, create an IMPORTED target for each one and link the
####### melon_DEPS_TARGET to all of them
conan_package_library_targets("${melon_LIBS_RELEASE}"    # libraries
                              "${melon_LIB_DIRS_RELEASE}" # package_libdir
                              "${melon_BIN_DIRS_RELEASE}" # package_bindir
                              "${melon_LIBRARY_TYPE_RELEASE}"
                              "${melon_IS_HOST_WINDOWS_RELEASE}"
                              melon_DEPS_TARGET
                              melon_LIBRARIES_TARGETS  # out_libraries_targets
                              "_RELEASE"
                              "melon"    # package_name
                              "${melon_NO_SONAME_MODE_RELEASE}")  # soname

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${melon_BUILD_DIRS_RELEASE} ${CMAKE_MODULE_PATH})

########## GLOBAL TARGET PROPERTIES Release ########################################
    set_property(TARGET melon::melon
                 APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                 $<$<CONFIG:Release>:${melon_OBJECTS_RELEASE}>
                 $<$<CONFIG:Release>:${melon_LIBRARIES_TARGETS}>
                 )

    if("${melon_LIBS_RELEASE}" STREQUAL "")
        # If the package is not declaring any "cpp_info.libs" the package deps, system libs,
        # frameworks etc are not linked to the imported targets and we need to do it to the
        # global target
        set_property(TARGET melon::melon
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     melon_DEPS_TARGET)
    endif()

    set_property(TARGET melon::melon
                 APPEND PROPERTY INTERFACE_LINK_OPTIONS
                 $<$<CONFIG:Release>:${melon_LINKER_FLAGS_RELEASE}>)
    set_property(TARGET melon::melon
                 APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                 $<$<CONFIG:Release>:${melon_INCLUDE_DIRS_RELEASE}>)
    # Necessary to find LINK shared libraries in Linux
    set_property(TARGET melon::melon
                 APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                 $<$<CONFIG:Release>:${melon_LIB_DIRS_RELEASE}>)
    set_property(TARGET melon::melon
                 APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                 $<$<CONFIG:Release>:${melon_COMPILE_DEFINITIONS_RELEASE}>)
    set_property(TARGET melon::melon
                 APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                 $<$<CONFIG:Release>:${melon_COMPILE_OPTIONS_RELEASE}>)

########## For the modules (FindXXX)
set(melon_LIBRARIES_RELEASE melon::melon)
