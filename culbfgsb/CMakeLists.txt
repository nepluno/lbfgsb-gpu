append_files (Headers "h" .)
append_files (Sources "cu" .)
append_files (Sources "cpp" .)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "-fpermissive")
endif()

option(BUILD_CULBFGSB_SHARED "Build Dynamically-linked CULBFGSB" OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
if(BUILD_CULBFGSB_SHARED)
add_definitions(-DCULBFGSB_SHARED)
cuda_add_library(cuLBFGSB SHARED
  ${Sources}
  ${Headers})
else(BUILD_CULBFGSB_SHARED)
cuda_add_library(cuLBFGSB STATIC
  ${Sources}
  ${Headers})
endif()
cuda_add_cublas_to_target (cuLBFGSB)

install(FILES culbfgsb.h DESTINATION ${CMAKE_BINARY_DIR}/include/culbfgsb)
