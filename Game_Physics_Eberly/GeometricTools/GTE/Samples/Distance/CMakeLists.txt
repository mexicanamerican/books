if(COMMAND cmake_policy)
    # Allow VERSION in the project() statement.
    cmake_policy(SET CMP0048 NEW)
endif()

cmake_minimum_required(VERSION 3.8)

set(GTE_VERSION_MAJOR 7)
set(GTE_VERSION_MINOR 1)

project(Distance VERSION ${GTE_VERSION_MAJOR}.${GTE_VERSION_MINOR})

add_subdirectory(DistanceAlignedBoxes)
add_subdirectory(DistanceAlignedBoxOrientedBox)
add_subdirectory(DistanceLine2Box2)
add_subdirectory(DistanceOrientedBoxConeFrustum)
add_subdirectory(DistanceOrientedBoxes)
add_subdirectory(DistancePointConvexPolyhedron)
add_subdirectory(DistancePointHyperellipsoid)
add_subdirectory(DistanceRectangleBox)
add_subdirectory(DistanceSegments3)
add_subdirectory(DistanceTriangleBox)
