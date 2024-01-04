cmake_minimum_required(VERSION 3.14)

add_executable(code_generator codegen.cpp)
target_include_directories(code_generator BEFORE PRIVATE ${CMAKE_CURRENT_LIST_DIR}/monkey_include)
target_include_directories(code_generator BEFORE PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../third_party/ascend/include/ops/op_proto/inc)

target_compile_options(code_generator PRIVATE -std=c++11)
target_link_libraries(code_generator PRIVATE stdc++)

add_executable(code_generator_asc codegen_asc.cpp)
target_include_directories(code_generator_asc BEFORE PRIVATE ${CMAKE_CURRENT_LIST_DIR}/monkey_include)
target_include_directories(code_generator_asc BEFORE PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../third_party/ascend/include/ops/op_proto/inc)

target_compile_options(code_generator_asc PRIVATE -std=c++11)
target_compile_definitions(code_generator_asc PRIVATE -DCODEGEN_ASC_OPS_H=all_ops.h)
target_link_libraries(code_generator_asc PRIVATE stdc++)