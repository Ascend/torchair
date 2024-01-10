add_subdirectory(autoschedule)

add_library(test_optimize OBJECT
    test_infer_output.cpp)
target_link_libraries(test_optimize
    test_autoschedule
    optimize GTest::gtest)
