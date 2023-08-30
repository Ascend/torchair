include_guard()
add_library(ascend_metadef_libs INTERFACE)

target_include_directories(ascend_metadef_libs INTERFACE ${ASCEND_SDK_HEADERS_PATH}/include/metadef)
target_include_directories(ascend_metadef_libs INTERFACE ${ASCEND_SDK_HEADERS_PATH}/include/metadef/external)
target_include_directories(ascend_metadef_libs INTERFACE ${ASCEND_SDK_HEADERS_PATH}/include/air)
target_include_directories(ascend_metadef_libs INTERFACE ${ASCEND_SDK_HEADERS_PATH}/include/air/external)

add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
        COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
)

set(fake_sources ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc)

add_library(ascend_protobuf SHARED ${fake_sources})
add_library(graph SHARED ${fake_sources})

set_target_properties(ascend_protobuf PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/stubs/unused)
set_target_properties(graph PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/stubs/unused)

target_link_libraries(ascend_metadef_libs INTERFACE
        ascend_protobuf
        graph)
