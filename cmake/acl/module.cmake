add_library(acl_libs INTERFACE)

include_directories(${ASCEND_CI_BUILD_DIR}/inc/external)
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
