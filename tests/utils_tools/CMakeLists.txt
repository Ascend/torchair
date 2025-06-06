add_library(libopapi SHARED opapi_fake.cpp)

set(TORCHAIR_SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/../../)

file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/../../tools/COMPILE_FLAGS" CUSTOM_COMPILE_FLAGS)

set_target_properties(libopapi PROPERTIES PREFIX "")
foreach (COMPILE_FLAG ${CUSTOM_COMPILE_FLAGS})
    target_compile_options(libopapi PRIVATE ${COMPILE_FLAG})
endforeach (COMPILE_FLAG)