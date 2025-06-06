cmake_minimum_required(VERSION 3.14)
project(TorchairTest)

set(TORCHAIR_SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/../)

set(CMAKE_CXX_STANDARD 17)

file(STRINGS "${TORCHAIR_SRC_DIR}/tools/TORCH_INSTALLED_PATH" TORCH_INSTALLED_PATH)
set(ASCEND_SDK_HEADERS_PATH ${TORCHAIR_SRC_DIR}/third_party/ascend)

file(STRINGS "${TORCHAIR_SRC_DIR}/tools/TORCH_VERSION" TORCH_VERSION)
if(DEFINED TORCH_VERSION)
  add_definitions(-DTNG_TORCH_VERSION=${TORCH_VERSION})
endif()

set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/stubs)

include(${TORCHAIR_SRC_DIR}/cmake/metadef/module.cmake)
include(${TORCHAIR_SRC_DIR}/cmake/graph_engine/module.cmake)
include(${TORCHAIR_SRC_DIR}/cmake/acl/module.cmake)
include(${TORCHAIR_SRC_DIR}/cmake/aoe/module.cmake)
include(${TORCHAIR_SRC_DIR}/cmake/hccl/module.cmake)
include(${TORCHAIR_SRC_DIR}/cmake/pytorch/module.cmake)
include(${TORCHAIR_SRC_DIR}/cmake/torch_npu/module.cmake)
include_directories(${TORCHAIR_SRC_DIR}/torchair/include)

add_subdirectory(${TORCHAIR_SRC_DIR} torchair)

add_subdirectory(${TORCHAIR_SRC_DIR}/third_party/googletest gtest)
include_directories(${TORCHAIR_SRC_DIR}/third_party/googletest/googletest/include)
include_directories(${TORCHAIR_SRC_DIR}/third_party/secure_c/include)



set(CMAKE_CXX_FLAGS "-D_FORTIFY_SOURCE=2 -Wno-class-memaccess -Wno-float-equal -Wextra -Wno-unused-parameter -Wno-ignored-qualifiers -Wno-comment -Wno-deprecated-declarations -Wall -fPIC -fstack-protector-all -Wl,--no-as-needed -Wl,-z,relro,-z,now,-z,noexecstack -fno-common -pipe -fno-strict-aliasing -Wdate-time -Wformat=2 -Wno-shadow -Wno-undef -Wunused -Wdelete-non-virtual-dtor -Wnon-virtual-dtor -Wno-overloaded-virtual ${CMAKE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lgcov --coverage -fprofile-arcs -ftest-coverage ")

add_subdirectory(privateuse1)
add_subdirectory(aoe)
add_subdirectory(reg_op_stub)
add_subdirectory(utils_tools)
add_subdirectory(ut)
add_subdirectory(st)
