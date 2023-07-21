file(GLOB_RECURSE TEST_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/testcases/*.cpp
        )

file(GLOB_RECURSE LIB_SOURCES
        ${TORCHAIR_SRC_DIR}/torchair/concrete_graph/*.cpp
        )

file(GLOB_RECURSE NPU_EXECUTOR_SOURCES
        ${TORCHAIR_SRC_DIR}/torchair/npu_graph_executor/npu_executor.cpp
        ${TORCHAIR_SRC_DIR}/torchair/npu_graph_executor/npu_graph_executor.cpp
        )

add_executable(torchair_ut_main ${TEST_SOURCES})
target_compile_options(torchair_ut_main PRIVATE
        -O0 -g --coverage -fprofile-arcs -ftest-coverage)
target_link_options(torchair_ut_main PRIVATE -Wl,--copy-dt-needed-entries)

include_directories(${TORCHAIR_SRC_DIR}/torchair/include)
set(TORCH_INCLUDE_DIR ${TORCH_INSTALLED_PATH}/include)
include_directories(${TORCH_INCLUDE_DIR})
include_directories(${TORCH_INCLUDE_DIR}/torch/csrc/api/include)
include_directories(${TORCH_INCLUDE_DIR}/torch/csrc/distributed)
include_directories(${TORCHAIR_SRC_DIR}/third_party)

file(STRINGS "${TORCHAIR_SRC_DIR}/tools/TORCH_INSTALLED_PATH" TORCH_INSTALLED_PATH)
#todo: Add ${NPU_EXECUTOR_SOURCES} once it's stub ready
add_library(torchair_ut_lib SHARED ${LIB_SOURCES})
target_compile_definitions(torchair_ut_lib PRIVATE google=ascend_private)
target_link_libraries(torchair_ut_lib PRIVATE
        _abi_compat_ge_apis
        ge_libs
        ascend_metadef_libs
        acl_libs
        ${TORCH_INSTALLED_PATH}/lib/libc10.so
        ${TORCH_INSTALLED_PATH}/lib/libtorch.so)

target_link_libraries(torchair_ut_main PRIVATE
        torchair_ut_lib
        gtest
        -lgcov -lrt -ldl -lutil -lpthread)

add_custom_target(torchair_ut ALL
        COMMAND lcov -c -q -d ${CMAKE_BINARY_DIR}/torchair/torchair -o ut_p1.coverage
        COMMAND ./torchair_ut_main
        COMMAND lcov -c -q -d . -o ut_p2.coverage
        COMMAND lcov -q -o ut.coverage -a ut_p1.coverage -a ut_p2.coverage
        DEPENDS torchair_ut_main run_torchair_py_cases
        VERBATIM)
