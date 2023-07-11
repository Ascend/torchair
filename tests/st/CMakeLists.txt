add_custom_command(OUTPUT ST_TESTCASES
        COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/torchair_st.py ${CMAKE_BINARY_DIR}/st
        VERBATIM)

add_custom_target(run_torchair_py_cases ALL
        COMMAND ${PYTHON_BIN_PATH} -m unittest torchair_st
        DEPENDS install_torchair ST_TESTCASES
        VERBATIM)

add_custom_target(torchair_st ALL
        COMMAND lcov -c -q -d ${CMAKE_BINARY_DIR}/torchair/torchair -o st_p1.coverage
        COMMAND lcov -o -q st.coverage -a st_p1.coverage
        DEPENDS run_torchair_py_cases
        VERBATIM)
