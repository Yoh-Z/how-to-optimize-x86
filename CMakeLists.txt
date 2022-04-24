cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cmake_test LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "-mavx2 -mavx -mfma")
list(APPEND CXX_FLAGS_RELEASE "-O2" "-Wno-unused")

message(STATUS "Release Build FLAGS: " ${CXX_FLAGS_RELEASE})
message(STATUS "CXX_FLAGS :" ${CMAKE_CXX_FLAGS})

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(USE_LIBRARY OFF)
message(STATUS "Compile sources into a library? ${USE_LIBRARY}")

set(BUILD_SHARED_LIBS OFF)
list(APPEND _tool conv/tool.h conv/tool.cpp)

if(USE_LIBRARY)
        add_library(Tool ${_tool})
        add_executable(test main.cpp)
        target_link_libraries(cmake_test Tool)
else()
        add_executable(cmake_test conv/main.cpp ${_tool})
endif()
