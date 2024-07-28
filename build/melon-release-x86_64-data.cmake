########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

set(melon_COMPONENT_NAMES "")
if(DEFINED melon_FIND_DEPENDENCY_NAMES)
  list(APPEND melon_FIND_DEPENDENCY_NAMES range-v3 fmt)
  list(REMOVE_DUPLICATES melon_FIND_DEPENDENCY_NAMES)
else()
  set(melon_FIND_DEPENDENCY_NAMES range-v3 fmt)
endif()
set(range-v3_FIND_MODE "NO_MODULE")
set(fmt_FIND_MODE "NO_MODULE")

########### VARIABLES #######################################################################
#############################################################################################
set(melon_PACKAGE_FOLDER_RELEASE "/home/b_1020/.conan2/p/b/melonfff8d3f95e6eb/p")
set(melon_BUILD_MODULES_PATHS_RELEASE )


set(melon_INCLUDE_DIRS_RELEASE "${melon_PACKAGE_FOLDER_RELEASE}/include")
set(melon_RES_DIRS_RELEASE )
set(melon_DEFINITIONS_RELEASE )
set(melon_SHARED_LINK_FLAGS_RELEASE )
set(melon_EXE_LINK_FLAGS_RELEASE )
set(melon_OBJECTS_RELEASE )
set(melon_COMPILE_DEFINITIONS_RELEASE )
set(melon_COMPILE_OPTIONS_C_RELEASE )
set(melon_COMPILE_OPTIONS_CXX_RELEASE )
set(melon_LIB_DIRS_RELEASE )
set(melon_BIN_DIRS_RELEASE )
set(melon_LIBRARY_TYPE_RELEASE UNKNOWN)
set(melon_IS_HOST_WINDOWS_RELEASE 0)
set(melon_LIBS_RELEASE )
set(melon_SYSTEM_LIBS_RELEASE )
set(melon_FRAMEWORK_DIRS_RELEASE )
set(melon_FRAMEWORKS_RELEASE )
set(melon_BUILD_DIRS_RELEASE )
set(melon_NO_SONAME_MODE_RELEASE FALSE)


# COMPOUND VARIABLES
set(melon_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${melon_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${melon_COMPILE_OPTIONS_C_RELEASE}>")
set(melon_LINKER_FLAGS_RELEASE
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${melon_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${melon_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${melon_EXE_LINK_FLAGS_RELEASE}>")


set(melon_COMPONENTS_RELEASE )