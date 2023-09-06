add_custom_command(OUTPUT ST_TESTCASES
        COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/torchair_st.py ${CMAKE_BINARY_DIR}/st
        COMMAND cp ${CMAKE_BINARY_DIR}/stubs/_privateuse1_backend.so ${CMAKE_BINARY_DIR}/st
        DEPENDS _privateuse1_backend
        VERBATIM)

add_custom_target(run_torchair_py_cases ALL
        COMMAND ${PYTHON_BIN_PATH} -m unittest torchair_st
        DEPENDS install_torchair ST_TESTCASES
        VERBATIM)

add_custom_target(torchair_st ALL
        COMMAND lcov -c -q -d ${CMAKE_BINARY_DIR}/torchair/torchair -o st_p1.coverage
        COMMAND lcov -o st.coverage -q -a st_p1.coverage
        DEPENDS run_torchair_py_cases
        VERBATIM)
