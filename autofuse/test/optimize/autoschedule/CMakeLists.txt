add_library(test_autoschedule OBJECT
    test_autoschedule_abs.cpp
    )
target_include_directories(test_autoschedule  PRIVATE
    ${ASCEND_ROOT}/x86_64-linux/include
    ${ASCEND_ROOT}/opp/built-in/op_proto/inc)

target_link_libraries(test_autoschedule
    depends
    ascir
    autoschedule
    ge_compiler
    GTest::gtest)
