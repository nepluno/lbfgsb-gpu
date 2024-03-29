cmake_minimum_required (VERSION 3.15)

project (cuLBFGSB)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)

# Initialize the build type (Release, Debug, etc)
if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: Debug Release."
    FORCE)
endif (NOT CMAKE_BUILD_TYPE)

add_definitions (-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})

find_package(CUDA REQUIRED)

foreach(ComputeCapability 60 61 70 72 75 86)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_${ComputeCapability},code=sm_${ComputeCapability})
endforeach()

if (MSVC)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcompiler;$<$<CONFIG:Debug>:/MDd>$<$<CONFIG:Release>:/MD>)
endif()

if (CMAKE_BUILD_TYPE MATCHES Debug)
  add_definitions (-DDEBUG)
endif (CMAKE_BUILD_TYPE MATCHES Debug)

# Add directory with macros
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# Add cmake file with useful functions
include (CmakeCommon)

include_directories (${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory (culbfgsb)
add_subdirectory (examples)


