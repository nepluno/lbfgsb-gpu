append_files (Headers "h" .)
append_files (Sources "cu" .)
append_files (Sources "cpp" .)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(EXAMPLES_LIBRARIES ${EXAMPLES_LIBRARIES} cuLBFGSB)

cuda_add_executable (dsscfg ${Headers} ${Templates} ${Sources})
target_link_libraries (dsscfg ${EXAMPLES_LIBRARIES})
