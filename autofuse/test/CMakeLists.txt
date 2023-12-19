add_subdirectory(ascir)
add_subdirectory(e2e)
add_subdirectory(python)

add_executable(test_main test_main.cpp)
target_link_libraries(test_main PRIVATE
    test_ascir
    test_e2e
    GTest::Main)

gtest_discover_tests(test_main)
