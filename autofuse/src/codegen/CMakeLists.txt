add_library(codegen SHARED
    codegen_kernel.cpp
    codegen.cpp
    )
target_include_directories(codegen PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    )
target_link_libraries(codegen PUBLIC
    ascir
    )

