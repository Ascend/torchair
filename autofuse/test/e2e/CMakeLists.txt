add_library(e2e SHARED
    ./e2e_load_abs_store.cpp
    ./e2e_softmax.cpp
    )
target_link_libraries(e2e PUBLIC
    autofuse
    )
target_include_directories(e2e PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

add_library(test_e2e OBJECT
    test_e2e_load_abs_store.cpp
    test_e2e_softmax.cpp
    )
target_link_libraries(test_e2e
    e2e
    autofuse
    GTest::gtest
    )

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${ASCEND_ROOT}/tools/tikicpulib/lib)
find_package(tikicpulib)
add_subdirectory(load_abs_store_expect_code)
