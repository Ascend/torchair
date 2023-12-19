add_library(test_ascir OBJECT
    test_ascir.cpp
    test_ascir_ops.cpp
    )
target_include_directories(test_ascir PRIVATE
    ${ASCEND_ROOT}/x86_64-linux/include
    ${ASCEND_ROOT}/opp/built-in/op_proto/inc)
target_link_libraries(test_ascir
    ascir
    #ge_compiler
    GTest::gtest)
