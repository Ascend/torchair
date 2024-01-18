add_e2e_test(test_load_rmax_store_codegen
    TILING load_rmax_store_codegen_tiling.cpp
    CODEGEN load_rmax_store_codegen.cpp
    KERNEL_SRC
        load_rmax_store_kernel.cpp
        load_rmax_store_tiling.cpp
        load_rmax_store_tiling.h
    TEST_SRC test_load_rmax_store_kernel.cpp)

