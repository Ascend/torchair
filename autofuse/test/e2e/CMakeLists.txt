include(e2e.cmake)

add_library(e2e SHARED
    ./e2e_load_abs_store.cpp
    ./e2e_load_sub_store.cpp
    ./e2e_softmax.cpp
    ./e2e_broadcast.cpp
    )
target_link_libraries(e2e PUBLIC
    autofuse
    )
target_include_directories(e2e PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

add_library(test_e2e OBJECT
    test_e2e_load_abs_store.cpp
    test_e2e_load_sub_store.cpp
    test_e2e_softmax.cpp
    )
target_link_libraries(test_e2e
    e2e
    autofuse
    nlohmann_json
    GTest::gtest
    )

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${ASCEND_ROOT}/tools/tikicpulib/lib)
find_package(tikicpulib QUIET)
string(REGEX MATCH "NOTFOUND$" tikicpulib_not_found "${tikicpulib_DIR}")
if("${tikicpulib_not_found}" STREQUAL "NOTFOUND")
    message(WARNING "tikicpulib not found, skip e2e calculate testcase.")
else()
    add_subdirectory(load_abs_store_expect_code)
    add_subdirectory(load_sub_store_expect_code)
    add_subdirectory(load_broadcast_store_expect_code)
endif()
