cmake_minimum_required(VERSION 3.14)

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
foreach (CPP_SOURCE ${SOURCES})
    file(RELATIVE_PATH RELATIVE_CPP_SOURCE ${CMAKE_CURRENT_SOURCE_DIR} ${CPP_SOURCE})
    set_property(SOURCE ${CPP_SOURCE} PROPERTY COMPILE_DEFINITIONS __FILE__=\"${RELATIVE_CPP_SOURCE}\")
endforeach (CPP_SOURCE)

add_library(_abi_compat_ge_apis SHARED ${SOURCES})
set_target_properties(_abi_compat_ge_apis PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/dist/torchair/core)
target_compile_definitions(_abi_compat_ge_apis PRIVATE google=ascend_private)
target_compile_options(_abi_compat_ge_apis PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
set_target_properties(_abi_compat_ge_apis PROPERTIES PREFIX "")
target_link_libraries(_abi_compat_ge_apis PRIVATE
        ge_libs
        acl_libs
        ascend_metadef_libs)