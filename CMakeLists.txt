cmake_minimum_required(VERSION 3.14)
project(AscendAdapter2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_FLAGS "-D_FORTIFY_SOURCE=2 -O2 -DNDEBUG -Wno-class-memaccess -Wno-float-equal -Wextra -Wno-unused-parameter -Wno-ignored-qualifiers -Wno-comment -Wno-deprecated-declarations -Wall -fPIC -fstack-protector-all -Wl,--no-as-needed -Wl,-z,relro,-z,now,-z,noexecstack -fno-common -pipe -fno-strict-aliasing -Wdate-time -Wformat=2 -Wno-shadow -Wno-undef -Wunused -Wstrict-prototypes ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-D_FORTIFY_SOURCE=2 -O2 -DNDEBUG -Wno-class-memaccess -Wno-float-equal -Wextra -Wno-unused-parameter -Wno-ignored-qualifiers -Wno-comment -Wno-deprecated-declarations -Wall -fPIC -fstack-protector-all -Wl,--no-as-needed -Wl,-z,relro,-z,now,-z,noexecstack -fno-common -pipe -fno-strict-aliasing -Wdate-time -Wformat=2 -Wno-shadow -Wno-undef -Wunused -Wdelete-non-virtual-dtor -Wnon-virtual-dtor -Wno-overloaded-virtual ${CMAKE_CXX_FLAGS}")
set(CMAKE_SKIP_RPATH TRUE)
set(TORCHAIR_SRC_DIR ${CMAKE_CURRENT_LIST_DIR})

IF (${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
    add_definitions(-Wno-builtin-macro-redefined)
ENDIF ()

if (NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/tools/COMPILE_FLAGS OR NOT EXISTS
        ${CMAKE_CURRENT_LIST_DIR}/tools/TORCH_INSTALLED_PATH OR NOT EXISTS
        ${CMAKE_CURRENT_LIST_DIR}/tools/PYTHON_BIN_PATH)
    message(FATAL_ERROR "No validate configuration found. Did you forget to configure first?")
endif ()

file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/tools/TORCH_INSTALLED_PATH" TORCH_INSTALLED_PATH)
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/tools/PYTHON_BIN_PATH" PYTHON_BIN_PATH)
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/tools/COMPILE_FLAGS" CUSTOM_COMPILE_FLAGS)

if (NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/tools/ASCEND_SDK_PATH)
    set(ASCEND_SDK_HEADERS_PATH ${CMAKE_CURRENT_LIST_DIR}/third_party/ascend)
else()
    file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/tools/ASCEND_SDK_PATH" ASCEND_SDK_HEADERS_PATH)
endif ()

include(${CMAKE_CURRENT_LIST_DIR}/cmake/nlohmann_json.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/secure_c.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/metadef/module.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/graph_engine/module.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/acl/module.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/aoe/module.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/pytorch/module.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/torch_npu/module.cmake)


add_subdirectory(codegen)
add_custom_command(OUTPUT AUTO_GENERATED_GE_RAW_OPS
        COMMAND code_generator ${CMAKE_CURRENT_LIST_DIR}/python/torchair/ge_concrete_graph/auto_generated_ge_raw_ops.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        COMMENT "Generating ge apis"
        DEPENDS code_generator
        VERBATIM)

add_custom_target(generate_ge_raw_ops ALL
        DEPENDS AUTO_GENERATED_GE_RAW_OPS
        VERBATIM)

add_custom_target(copy_pyfiles ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/python ${CMAKE_BINARY_DIR}/dist
        COMMENT "Copy pthon files"
        VERBATIM)

set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/stubs)

add_subdirectory(torchair)

add_custom_target(torchair ALL
        COMMAND cd ${CMAKE_BINARY_DIR}/dist/ && ${PYTHON_BIN_PATH} setup.py bdist_wheel
        DEPENDS _torchair _npu_graph_executor copy_pyfiles
        VERBATIM)

install(CODE "execute_process(COMMAND ${PYTHON_BIN_PATH} -m pip install ${CMAKE_BINARY_DIR}/dist/dist/torchair-0.1-py3-none-any.whl --upgrade)")
