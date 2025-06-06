﻿cmake_minimum_required(VERSION 3.8)

project(gtapplications VERSION ${GTE_VERSION_MAJOR}.${GTE_VERSION_MINOR})

option(BUILD_RELEASE_LIB "Build release library" OFF)
option(BUILD_SHARED_LIB "Build shared library" OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_definitions(-DGTE_USE_LINUX -DGTE_USE_ROW_MAJOR -DGTE_USE_MAT_VEC -DGTE_USE_OPENGL -DGTE_DISABLE_PCH)
add_compile_options(-c -Wall -Werror)
if(BUILD_RELEASE_LIB)
    add_compile_definitions(NDEBUG)
    add_compile_options(-O3)
else()
    add_compile_definitions(_DEBUG)
    add_compile_options(-g)
endif()

set(GTE_ROOT ${PROJECT_SOURCE_DIR}/..)
set(GTE_INCLUDE_DIR ${GTE_ROOT})
set(GTE_LIB_PREFIX ${GTE_ROOT}/lib/${CMAKE_BUILD_TYPE})

include_directories(${GTE_INCLUDE_DIR})

set(GTE_CPP_FILES
Application.cpp
CameraRig.cpp
Command.cpp
ConsoleApplication.cpp
Environment.cpp
GTApplications.cpp
OnIdleTimer.cpp
Timer.cpp
TrackBall.cpp
TrackCylinder.cpp
TrackObject.cpp
Window2.cpp
Window3.cpp
WindowApplication.cpp
GLX/Console.cpp
GLX/ConsoleSystem.cpp
GLX/WICFileIO.cpp
GLX/Window.cpp
GLX/WindowSystem.cpp)

if(BUILD_SHARED_LIB)
    add_library(${PROJECT_NAME} SHARED ${GTE_CPP_FILES})
    set_target_properties(${PROJECT_NAME} PROPERTIES SOVERSION ${PROJECT_VERSION_MAJOR} VERSION ${PROJECT_VERSION})
    set(GTE_LIB_DIR ${GTE_LIB_PREFIX}Shared)
else()
    add_library(${PROJECT_NAME} STATIC ${GTE_CPP_FILES})
    set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION})
    set(GTE_LIB_DIR ${GTE_LIB_PREFIX}Static)
endif()
set(LIBRARY_OUTPUT_PATH ${GTE_LIB_DIR} CACHE PATH "Library directory" FORCE)
SET(LIBRARY_OUTPUT_PATH ${GTE_LIB_DIR})
