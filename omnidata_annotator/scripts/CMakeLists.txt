cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(taskonomydata)

if(APPLE)
    find_package(PCL 1.8 REQUIRED)
endif()
if(NOT APPLE)
    find_package(PCL 1.7 REQUIRED)
endif()
find_package(Boost 1.48.0 REQUIRED)
find_package(PNG REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS} ${Boost_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS} ${Boost_DEFINITIONS})

add_executable (narf_interest_image.bin narf_interest_image.cpp)


if(APPLE)
    target_link_libraries (narf_interest_image.bin ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${PNG_LIBRARIES})
endif()
if(NOT APPLE)
    target_link_libraries (narf_interest_image.bin ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${PNG_LIBRARIES})
    
endif()

set(CMAKE_BUILD_TYPE Release)
