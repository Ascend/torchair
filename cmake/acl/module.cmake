include_guard()
add_library(acl_libs INTERFACE)

target_include_directories(acl_libs INTERFACE ${ASCEND_SDK_HEADERS_PATH}/include/ascendcl/external/)

add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
        COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
)

set(fake_sources ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc)

add_library(ascendcl SHARED ${fake_sources})
add_library(acl_op_compiler SHARED ${fake_sources})
add_library(acl_tdt_channel SHARED ${fake_sources})
target_link_libraries(acl_libs INTERFACE
        ascendcl
        acl_op_compiler
        acl_tdt_channel)
