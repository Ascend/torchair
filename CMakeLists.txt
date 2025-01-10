cmake_minimum_required(VERSION 3.14)
project(AscendAdapter2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-D_FORTIFY_SOURCE=2 -Wno-class-memaccess -Wno-float-equal -Wextra -Wno-unused-parameter -Wno-ignored-qualifiers -Wno-comment -Wno-deprecated-declarations -Wall -fPIC -fPIE -pie -s -fstack-protector-all -Wl,--no-as-needed -Wl,-z,relro,-z,now,-z,noexecstack -fno-common -pipe -fno-strict-aliasing -Wdate-time -Wformat=2 -Wno-shadow -Wno-undef -Wunused -Wdelete-non-virtual-dtor -Wnon-virtual-dtor -Wno-overloaded-virtual ${CMAKE_CXX_FLAGS}")
set(CMAKE_SKIP_RPATH TRUE)

set(ALLOWED_BUILD_TYPES "DEBUG" "GCOV" "RELEASE" CACHE STRING "Allowed build types")
set_property(CACHE ALLOWED_BUILD_TYPES PROPERTY STRINGS "DEBUG" "GCOV" "RELEASE")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RELEASE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    message("Configuring for DEBUG build")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
elseif(CMAKE_BUILD_TYPE STREQUAL "GCOV")
    message("Configuring for GCOV build")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 --coverage -fprofile-arcs -ftest-coverage")
elseif(CMAKE_BUILD_TYPE STREQUAL "RELEASE")
    message("Configuring for RELEASE build")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -DNDEBUG")
endif()

set(TORCHAIR_SRC_DIR ${CMAKE_CURRENT_LIST_DIR})

IF (${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
    add_definitions(-Wno-builtin-macro-redefined)
ENDIF ()

if (NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/tools/COMPILE_FLAGS OR NOT EXISTS
        ${CMAKE_CURRENT_LIST_DIR}/tools/TORCH_INSTALLED_PATH OR NOT EXISTS
        ${CMAKE_CURRENT_LIST_DIR}/tools/PYTHON_BIN_PATH OR NOT EXISTS
        ${CMAKE_CURRENT_LIST_DIR}/tools/TORCH_VERSION)
    message(FATAL_ERROR "No validate configuration found. Did you forget to configure first?")
endif ()

file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/tools/TORCH_INSTALLED_PATH" TORCH_INSTALLED_PATH)
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/tools/PYTHON_BIN_PATH" PYTHON_BIN_PATH)
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/tools/COMPILE_FLAGS" CUSTOM_COMPILE_FLAGS)
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/tools/TORCH_VERSION" TORCH_VERSION)

if(DEFINED TORCH_VERSION)
  add_definitions(-DTNG_TORCH_VERSION=${TORCH_VERSION})
endif()

set(ASCEND_SDK_HEADERS_PATH ${CMAKE_CURRENT_LIST_DIR}/third_party/ascend)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/metadef/module.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/graph_engine/module.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/acl/module.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/aoe/module.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/hccl/module.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/pytorch/module.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/torch_npu/module.cmake)
include_directories(${CMAKE_CURRENT_LIST_DIR}/third_party/secure_c/include)

if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/third_party/secure_c/include")
    message(FATAL_ERROR "Please execute 'git submodule update --init --recursive' to download and update third-party libraries!")
endif()

add_subdirectory(codegen)
add_custom_command(OUTPUT AUTO_GENERATED_GE_RAW_OPS
        COMMAND code_generator ${CMAKE_CURRENT_LIST_DIR}/python/torchair/_ge_concrete_graph/auto_generated_ge_raw_ops.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        COMMENT "Generating ge apis"
        DEPENDS code_generator
        VERBATIM)

add_custom_target(generate_ge_raw_ops ALL
        DEPENDS AUTO_GENERATED_GE_RAW_OPS
        VERBATIM)

add_custom_command(OUTPUT AUTO_GENERATED_GE_RAW_CUSTOM_SINGLE_OPS
        COMMAND code_generator_single ${CMAKE_CURRENT_LIST_DIR}/codegen/custom_op/auto_generated_ge_raw_custom_ops.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        COMMENT "Generating ge apis"
        DEPENDS code_generator_single
        VERBATIM)

add_custom_target(generate_ge_raw_custom_ops ALL
        DEPENDS AUTO_GENERATED_GE_RAW_CUSTOM_SINGLE_OPS
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

if (DEFINED TORCHAIR_INSTALL_DST)
    add_custom_target(install_torchair ALL
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/dist/torchair ${TORCHAIR_INSTALL_DST}
            DEPENDS _torchair _npu_graph_executor copy_pyfiles
            VERBATIM)
else()
    add_custom_target(install_torchair ALL
            COMMAND ${PYTHON_BIN_PATH} -m pip install ${CMAKE_BINARY_DIR}/dist/dist/torchair-0.1-py3-none-any.whl --force-reinstall
            DEPENDS torchair
            VERBATIM)
endif()

install(CODE "execute_process(COMMAND ${PYTHON_BIN_PATH} -m pip install ${CMAKE_BINARY_DIR}/dist/dist/torchair-0.1-py3-none-any.whl --upgrade)")
