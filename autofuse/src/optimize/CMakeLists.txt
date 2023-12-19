add_library(optimize SHARED
    optimize.cpp
    )
target_include_directories(optimize PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    )
target_link_libraries(optimize PUBLIC
    ascir
    )
