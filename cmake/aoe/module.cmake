include_guard()
add_library(aoe_libs INTERFACE)

add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
        COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
)

set(fake_sources ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc)

add_library(aoe_tuning SHARED ${fake_sources})
target_link_libraries(aoe_libs INTERFACE
        aoe_tuning)