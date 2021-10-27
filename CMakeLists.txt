cmake_minimum_required(VERSION 3.17)

project(tensor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(tensor INTERFACE)

target_include_directories(tensor INTERFACE include/)

if (BUILD_TESTS)
  add_subdirectory(tests)
endif()

get_target_property(INCLUDE_DIR_TO_INSTALL tensor INTERFACE_INCLUDE_DIRECTORIES)
install(TARGETS tensor)
install(DIRECTORY ${INCLUDE_DIR_TO_INSTALL} TYPE INCLUDE)

