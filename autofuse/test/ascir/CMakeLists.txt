add_library(test_ascir OBJECT
    test_ascir.cpp
    test_ascir_ops.cpp
    test_header_generator.cpp
        test_ascir_graph.cpp
        test_cg_util.cpp
)
target_include_directories(test_ascir PRIVATE
    ${ASCEND_ROOT}/x86_64-linux/include
    ${ASCEND_ROOT}/opp/built-in/op_proto/inc)
target_link_libraries(test_ascir
    depends
    ascir
    ascir_builtin_ops
    #ge_compiler
    GTest::gtest)
