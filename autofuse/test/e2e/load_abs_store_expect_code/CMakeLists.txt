add_e2e_test(test_load_abs_store_codegen
    TILING load_abs_store_codegen_tiling.cpp
    CODEGEN load_abs_store_codegen.cpp 
    KERNEL_SRC
        load_abs_store_kernel.cpp
        load_abs_store_tiling.cpp
        load_abs_store_tiling.h
    TEST_SRC test_e2e_load_abs_store_expect_kernel.cpp)
