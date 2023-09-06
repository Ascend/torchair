add_library(_privateuse1_backend SHARED privateuse1.cpp)

file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/../../tools/COMPILE_FLAGS" CUSTOM_COMPILE_FLAGS)

target_link_libraries(_privateuse1_backend PRIVATE torch_libs)
set_target_properties(_privateuse1_backend PROPERTIES PREFIX "")
foreach (COMPILE_FLAG ${CUSTOM_COMPILE_FLAGS})
    target_compile_options(_privateuse1_backend PRIVATE ${COMPILE_FLAG})
endforeach (COMPILE_FLAG)