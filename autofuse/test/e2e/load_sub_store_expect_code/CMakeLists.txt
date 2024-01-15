add_executable(test_e2e_load_sub_store EXCLUDE_FROM_ALL
        ./e2e_load_sub_store_expect_kernel.cpp
        ./e2e_load_sub_store_expect_tiling.cpp
        ./test_e2e_load_sub_store_expect_kernel.cpp
)

target_link_libraries(test_e2e_load_sub_store
        tikicpulib_ascend910B1
        GTest::gtest GTest::Main)