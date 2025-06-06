cmake_minimum_required(VERSION 3.14)

include_directories(${CMAKE_CURRENT_LIST_DIR}/include ${CMAKE_CURRENT_LIST_DIR}/)
add_subdirectory(abi_compat_ge_apis)

file(GLOB_RECURSE SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/core/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/concrete_graph/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/llm_datadist/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/cann_ir_ability/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/utils_tools/*.cpp)
foreach (CPP_SOURCE ${SOURCES})
    file(RELATIVE_PATH RELATIVE_CPP_SOURCE ${CMAKE_CURRENT_SOURCE_DIR} ${CPP_SOURCE})
    set_property(SOURCE ${CPP_SOURCE} PROPERTY COMPILE_DEFINITIONS __FILE__=\"${RELATIVE_CPP_SOURCE}\")
endforeach (CPP_SOURCE)

add_library(_torchair SHARED ${SOURCES})
set_target_properties(_torchair PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/dist/torchair/core)
target_compile_definitions(_torchair PRIVATE google=ascend_private)
foreach (COMPILE_FLAG ${CUSTOM_COMPILE_FLAGS})
    target_compile_options(_torchair PRIVATE ${COMPILE_FLAG})
endforeach (COMPILE_FLAG)

set_target_properties(_torchair PROPERTIES PREFIX "")
set_target_properties(_torchair PROPERTIES LINK_FLAGS "-Wl,-rpath,$ORIGIN/")
target_link_libraries(_torchair PRIVATE
        _abi_compat_ge_apis
        torch_libs
        ge_libs
        acl_libs
        ascend_metadef_libs
        hccl_libs)

add_subdirectory(npu_graph_executor)