cmake_minimum_required(VERSION 3.14)

include_directories(${CMAKE_CURRENT_LIST_DIR}/../include)

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
foreach (CPP_SOURCE ${SOURCES})
    file(RELATIVE_PATH RELATIVE_CPP_SOURCE ${CMAKE_CURRENT_SOURCE_DIR} ${CPP_SOURCE})
    set_property(SOURCE ${CPP_SOURCE} PROPERTY COMPILE_DEFINITIONS __FILE__=\"${RELATIVE_CPP_SOURCE}\")
endforeach (CPP_SOURCE)

add_library(_npu_graph_executor SHARED ${SOURCES})
set_target_properties(_npu_graph_executor PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/dist/torchair/core)
target_compile_definitions(_npu_graph_executor PRIVATE google=ascend_private)
foreach (COMPILE_FLAG ${CUSTOM_COMPILE_FLAGS})
    target_compile_options(_npu_graph_executor PRIVATE ${COMPILE_FLAG})
endforeach (COMPILE_FLAG)

set_target_properties(_npu_graph_executor PROPERTIES PREFIX "")
set_target_properties(_npu_graph_executor PROPERTIES LINK_FLAGS "-Wl,-Bsymbolic -Wl,-rpath,$ORIGIN/")
target_link_libraries(_npu_graph_executor PRIVATE
        _torchair
        _abi_compat_ge_apis
        torch_libs
        torch_npu_libs
        ge_libs
        acl_libs
        ascend_metadef_libs)