add_executable(test_e2e_load_abs_store
    ./e2e_load_abs_store_expect_kernel.cpp
    ./e2e_load_abs_store_expect_tiling.cpp
    ./test_e2e_load_abs_store_expect_kernel.cpp
    )

target_link_libraries(test_e2e_load_abs_store
    tikicpulib_ascend910B1
    GTest::gtest GTest::Main)

gtest_discover_tests(test_e2e_load_abs_store)
