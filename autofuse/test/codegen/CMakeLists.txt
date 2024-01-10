add_library(test_codegen OBJECT
    ./test_codegen_proto.cpp
    ./test_codegen_kernel.cpp
    ./test_codegen_tiling_data.cpp
    )
target_link_libraries(test_codegen
    codegen
    GTest::gtest)
