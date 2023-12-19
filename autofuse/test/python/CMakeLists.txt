add_test(NAME test_pyautofuse
    COMMAND pytest ${CMAKE_CURRENT_SOURCE_DIR})
set_property(TEST test_pyautofuse
    PROPERTY ENVIRONMENT PYTHONPATH=ENV{PYTHONPATH}:$<TARGET_FILE_DIR:pyautofuse>)

