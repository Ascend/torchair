include_guard()
add_library(ge_libs INTERFACE)

add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
        COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
)

set(fake_sources ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc)

add_library(ge_runner SHARED ${fake_sources})
add_library(fmk_parser SHARED ${fake_sources})
target_link_libraries(ge_libs INTERFACE
        ge_runner
        fmk_parser)

add_library(ge_local_stub SHARED ${fake_sources} ${CMAKE_CURRENT_LIST_DIR}/ge_stub.cpp)
target_compile_options(ge_local_stub PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
target_link_libraries(ge_local_stub PRIVATE ascend_metadef_libs)
target_link_libraries(ge_runner PRIVATE ge_local_stub ascend_metadef_libs)
