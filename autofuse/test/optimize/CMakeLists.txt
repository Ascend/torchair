add_library(test_optimize OBJECT
    test_infer_output.cpp
    autoschedule/test_autoschedule.cpp
    autoschedule/test_autoschedule_softmax.cpp
    )
target_link_libraries(test_optimize
    ascir
    optimize
    e2e
    GTest::gtest)
