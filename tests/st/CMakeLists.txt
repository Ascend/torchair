add_custom_command(OUTPUT ST_TESTCASES
        COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/torchair_st_utils.py ${CMAKE_BINARY_DIR}/st
        COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/torchair_st.py ${CMAKE_BINARY_DIR}/st
        COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/torchair_export_st.py ${CMAKE_BINARY_DIR}/st
        COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/torchair_scope_st.py ${CMAKE_BINARY_DIR}/st
        COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/torchair_gear_st.py ${CMAKE_BINARY_DIR}/st
        COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/torchair_loglevel_st.py ${CMAKE_BINARY_DIR}/st
        COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/torchair_cache_compile_st.py ${CMAKE_BINARY_DIR}/st
        COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/torchair_public_bindings_st.py ${CMAKE_BINARY_DIR}/st
        COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/allowlist_for_publicAPI.json ${CMAKE_BINARY_DIR}/st
        COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/torchair_hcom_st.py ${CMAKE_BINARY_DIR}/st
        COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/torchair_customop_st.py ${CMAKE_BINARY_DIR}/st
        COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/torchair_print_st.py ${CMAKE_BINARY_DIR}/st
        COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/torchair_explicit_order_st.py ${CMAKE_BINARY_DIR}/st
        COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/torchair_aclgraph_st.py ${CMAKE_BINARY_DIR}/st
        COMMAND cp ${CMAKE_BINARY_DIR}/stubs/_privateuse1_backend.so ${CMAKE_BINARY_DIR}/st
        COMMAND cp ${CMAKE_BINARY_DIR}/stubs/libaoe_tuning.so ${CMAKE_BINARY_DIR}/st
        COMMAND cp ${CMAKE_BINARY_DIR}/stubs/libopapi.so ${CMAKE_BINARY_DIR}/st
        DEPENDS _privateuse1_backend libaoe_tuning libopapi
        VERBATIM)

add_custom_target(run_torchair_py_cases ALL
        COMMAND ${PYTHON_BIN_PATH} -m unittest torchair_st
        COMMAND ${PYTHON_BIN_PATH} -m unittest torchair_scope_st
        COMMAND ${PYTHON_BIN_PATH} -m unittest torchair_export_st
        COMMAND ${PYTHON_BIN_PATH} -m unittest torchair_gear_st
        COMMAND ${PYTHON_BIN_PATH} -m unittest torchair_loglevel_st
        COMMAND ${PYTHON_BIN_PATH} -m unittest torchair_cache_compile_st
        COMMAND ${PYTHON_BIN_PATH} -m unittest torchair_public_bindings_st
        COMMAND ${PYTHON_BIN_PATH} -m unittest torchair_hcom_st
        COMMAND ${PYTHON_BIN_PATH} -m unittest torchair_customop_st
        COMMAND ${PYTHON_BIN_PATH} -m unittest torchair_print_st
        COMMAND ${PYTHON_BIN_PATH} -m unittest torchair_explicit_order_st
        COMMAND ${PYTHON_BIN_PATH} -m unittest torchair_aclgraph_st
        DEPENDS install_torchair ST_TESTCASES
        VERBATIM)

add_custom_target(torchair_st ALL
        COMMAND lcov -c -q -d ${CMAKE_BINARY_DIR}/torchair/torchair -o st_p1.coverage
        COMMAND lcov -o st.coverage -q -a st_p1.coverage
        DEPENDS run_torchair_py_cases
        VERBATIM)
