add_library(libaoe_tuning SHARED aoe_stub.cpp)

set(TORCHAIR_SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/../../)

file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/../../tools/COMPILE_FLAGS" CUSTOM_COMPILE_FLAGS)

include_directories(${TORCHAIR_SRC_DIR}/third_party/ascend/include/metadef/external/)
include_directories(${TORCHAIR_SRC_DIR}/third_party/ascend/include/air/external/)

target_link_libraries(libaoe_tuning PRIVATE torch_libs)
set_target_properties(libaoe_tuning PROPERTIES PREFIX "")
foreach (COMPILE_FLAG ${CUSTOM_COMPILE_FLAGS})
    target_compile_options(libaoe_tuning PRIVATE ${COMPILE_FLAG})
endforeach (COMPILE_FLAG)