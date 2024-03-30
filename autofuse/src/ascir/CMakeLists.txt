add_subdirectory(meta)
add_subdirectory(generator)

add_library(ascir INTERFACE)

target_link_libraries(ascir
        INTERFACE
        ascir_meta
        ascir_ops_headers
)

