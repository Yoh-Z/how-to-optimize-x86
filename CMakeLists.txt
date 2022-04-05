cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(cmake_test LANGUAGES CXX)

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