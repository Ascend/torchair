add_e2e_test(test_load_broadcast_store_codegen
    TILING load_broadcast_store_codegen_tiling.cpp
    CODEGEN load_broadcast_store_codegen.cpp 
    KERNEL_SRC
        load_broadcast_store_kernel.cpp
        load_broadcast_store_tiling.cpp
        load_broadcast_store_tiling.h
    TEST_SRC test_load_broadcast_store_codegen.cpp)
