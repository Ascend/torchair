include_guard()
add_library(hccl_libs INTERFACE)

target_include_directories(hccl_libs INTERFACE ${ASCEND_SDK_HEADERS_PATH}/include/hccl/external/)

add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
        COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
)

set(fake_sources ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc)

add_library(hccl SHARED ${fake_sources})
target_link_libraries(hccl_libs INTERFACE
        hccl)

add_library(hccl_stub SHARED ${fake_sources} ${CMAKE_CURRENT_LIST_DIR}/hccl_stub.cpp)
target_compile_options(hccl_stub PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
target_link_libraries(hccl PRIVATE hccl_stub)
target_include_directories(hccl_stub PRIVATE ${ASCEND_SDK_HEADERS_PATH}/include/hccl/external/)
