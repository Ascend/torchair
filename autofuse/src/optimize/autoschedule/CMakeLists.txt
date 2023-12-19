add_library(autoschedule
    autoschedule.cpp)

target_include_directories(autoschedule PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(autoschedule
    graph
    ascir
    graph_base)
