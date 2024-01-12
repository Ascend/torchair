add_custom_command(OUTPUT ST_TESTCASES
        COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/torchair_export_st.py ${CMAKE_BINARY_DIR}/st
        COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/torchair_st.py ${CMAKE_BINARY_DIR}/st
        COMMAND cp ${CMAKE_BINARY_DIR}/stubs/_privateuse1_backend.so ${CMAKE_BINARY_DIR}/st
        DEPENDS _privateuse1_backend
        VERBATIM)

add_custom_target(run_torchair_py_cases ALL
        COMMAND ${PYTHON_BIN_PATH} -m unittest torchair_st
        COMMAND ${PYTHON_BIN_PATH} -m unittest torchair_export_st
        DEPENDS install_torchair ST_TESTCASES
        VERBATIM)

add_custom_target(install_npu_inductor ALL
        COMMAND cd ${CMAKE_CURRENT_LIST_DIR}/npu_inductor && ${PYTHON_BIN_PATH} -m pip install -e .
        VERBATIM)

add_custom_command(OUTPUT NPU_INDUCTOR_ST_TESTCASES
        COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/npu_inductor/test/test_build_graph.py ${CMAKE_BINARY_DIR}/st
        COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/npu_inductor/test/test_syms_expr.py ${CMAKE_BINARY_DIR}/st
        VERBATIM)

add_custom_target(run_npu_inductor_py_cases ALL
        COMMAND ${PYTHON_BIN_PATH} -m unittest test_syms_expr
        COMMAND ${PYTHON_BIN_PATH} -m unittest test_build_graph
        DEPENDS install_npu_inductor NPU_INDUCTOR_ST_TESTCASES
        VERBATIM)

add_custom_target(torchair_st ALL
        COMMAND lcov -c -q -d ${CMAKE_BINARY_DIR}/torchair/torchair -o st_p1.coverage
        COMMAND lcov -o st.coverage -q -a st_p1.coverage
        DEPENDS run_torchair_py_cases run_npu_inductor_py_cases
        VERBATIM)
