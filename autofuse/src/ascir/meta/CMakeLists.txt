add_library(ascir_meta SHARED
        ascir_utils.cpp
        ascir.cpp
        ascir_ops_utils.cc
        cg_context.cpp
)

target_include_directories(ascir_meta PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ascir_meta PUBLIC
        graph graph_base
)
