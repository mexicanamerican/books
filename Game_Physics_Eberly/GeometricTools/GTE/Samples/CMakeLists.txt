if(COMMAND cmake_policy)
    # Allow VERSION in the project() statement.
    cmake_policy(SET CMP0048 NEW)
endif()

cmake_minimum_required(VERSION 3.8)

set(GTE_VERSION_MAJOR 7)
set(GTE_VERSION_MINOR 1)

project(Samples VERSION ${GTE_VERSION_MAJOR}.${GTE_VERSION_MINOR})

add_subdirectory(Distance)
add_subdirectory(Geometrics)
add_subdirectory(Graphics)
add_subdirectory(Imagics)
add_subdirectory(Intersection)
add_subdirectory(Mathematics)
add_subdirectory(Physics)
add_subdirectory(SceneGraphs)
