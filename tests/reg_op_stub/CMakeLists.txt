add_library(reg_op_stub SHARED reg_op_stub.cpp)
set(TORCHAIR_SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/../../)
target_include_directories(reg_op_stub PRIVATE ${TORCHAIR_SRC_DIR}/third_party/ascend/include/metadef/external/)
target_include_directories(reg_op_stub PRIVATE ${TORCHAIR_SRC_DIR}/third_party/ascend/include/air/external/)
target_link_libraries(ge_runner PRIVATE reg_op_stub)
