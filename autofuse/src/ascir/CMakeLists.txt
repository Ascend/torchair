add_library(ascir SHARED
    ascir_utils.cpp
    ascir.cpp
    )

target_include_directories(ascir PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ascir
    #ge_compiler # Currently only for ascir_utils to Dump graph
    graph graph_base)
