add_library(test_codegen OBJECT
    ./test_codegen_kernel.cpp)
target_link_libraries(test_codegen
    codegen
    GTest::gtest)
