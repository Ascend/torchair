include_guard()
add_library(torch_npu_libs INTERFACE)

add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
        COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
)

set(fake_sources ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc)

add_library(torch_npu SHARED ${fake_sources})
target_link_libraries(torch_npu_libs INTERFACE torch_npu)

add_library(torch_npu_local_stub SHARED ${CMAKE_CURRENT_LIST_DIR}/torch_npu_stub.cpp)
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/../../tools/COMPILE_FLAGS" CUSTOM_COMPILE_FLAGS)
foreach (COMPILE_FLAG ${CUSTOM_COMPILE_FLAGS})
    target_compile_options(torch_npu_local_stub PRIVATE ${COMPILE_FLAG})
endforeach (COMPILE_FLAG)
target_include_directories(torch_npu_local_stub PRIVATE ${ASCEND_SDK_HEADERS_PATH}/include/ascendcl/external/)
target_include_directories(torch_npu_local_stub PRIVATE ${TORCHAIR_SRC_DIR}/third_party/torch_npu/inc/)

target_link_libraries(torch_npu_local_stub PRIVATE torch_libs acl_libs)

target_link_libraries(torch_npu PRIVATE torch_npu_local_stub torch_libs)

target_include_directories(torch_npu_libs INTERFACE ${TORCHAIR_SRC_DIR}/third_party/ascend/acl/inc/)
target_include_directories(torch_npu_libs INTERFACE ${TORCHAIR_SRC_DIR}/third_party/)
