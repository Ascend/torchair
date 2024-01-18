add_e2e_test(test_load_sub_store_codegen
    TILING load_sub_store_codegen_tiling.cpp
    CODEGEN load_sub_store_codegen.cpp
    KERNEL_SRC
        load_sub_store_kernel.cpp
        load_sub_store_tiling.cpp
        load_sub_store_tiling.h
    TEST_SRC test_e2e_load_sub_store_expect_kernel.cpp)
