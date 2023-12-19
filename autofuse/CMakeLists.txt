cmake_minimum_required(VERSION 3.16.1)
project(autofuse)

enable_testing()

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)

set(ASCEND_ROOT $ENV{ASCEND_AICPU_PATH}) # e.g. /usr/local/Ascend/ascend-toolkit/latest
set(ASCEND_OPENSDK $ENV{ASCEND_CUSTOM_PATH}/opensdk/opensdk) # e.g. /usr/local/Ascend/latest/opensdk/opensdk
set(CMAKE_FIND_ROOT_PATH
    ${ASCEND_OPENSDK}/cmake
    ${ASCEND_OPENSDK}/gtest
    ${ASCEND_OPENSDK}/json)

find_package(nlohmann_json) # For inja
find_package(GTest REQUIRED)

find_package(air)
find_package(metadef)
find_package(runtime)
find_package(msprof)
find_package(platform)

include(GoogleTest)
set(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE PRE_TEST)

# Common depends
add_library(depends INTERFACE)
target_link_directories(depends INTERFACE
    ${ASCEND_ROOT}/x86_64-linux/devlib
    ${ASCEND_OPENSDK}/lib/stub)

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(thirdparty)
