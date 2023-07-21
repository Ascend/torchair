include_guard()
add_library(torch_libs INTERFACE)
add_library(torch_nopython_libs INTERFACE)

if(DEFINED TORCH_INSTALLED_PATH)
    SET(TORCH_INCLUDE_DIR ${TORCH_INSTALLED_PATH}/include)
    target_link_libraries(torch_libs INTERFACE
            ${TORCH_INSTALLED_PATH}/lib/libtorch.so
            ${TORCH_INSTALLED_PATH}/lib/libtorch_python.so)
    target_link_libraries(torch_nopython_libs INTERFACE
            ${TORCH_INSTALLED_PATH}/lib/libtorch.so)
else()
    add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
            COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc
    )

    set(fake_sources ${CMAKE_CURRENT_BINARY_DIR}/_fake.cc)

    add_library(torch SHARED ${fake_sources})
    add_library(torch_python SHARED ${fake_sources})

    SET(TORCH_INCLUDE_DIR ${TORCH_SOURCE_PATH}/include/)
    target_link_libraries(torch_libs INTERFACE torch torch_python)
    target_link_libraries(torch_nopython_libs INTERFACE torch)
endif()

include_directories(${TORCH_INCLUDE_DIR})
include_directories(${TORCH_INCLUDE_DIR}/torch/csrc/api/include)
include_directories(${TORCH_INCLUDE_DIR}/torch/csrc/distributed)