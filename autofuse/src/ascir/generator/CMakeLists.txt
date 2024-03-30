add_library(ascir_register STATIC
        ascir_register.cpp
        ascir_registry.cpp
        generator.cpp
)
target_compile_options(ascir_register PRIVATE "-fPIC")
target_link_libraries(ascir_register graph_base)

add_library(ascir_builtin_ops SHARED ascir_builtin_ops.cpp)
target_link_libraries(ascir_builtin_ops PRIVATE ascir_register)

add_executable(ascir_builtin_ops_header_generator builtin_ops_generator_main.cpp)
target_link_libraries(ascir_builtin_ops_header_generator PRIVATE ascir_builtin_ops)

set(ops_header_dir ${CMAKE_BINARY_DIR}/ascir_builtin_ops)
set_target_properties(ascir_builtin_ops_header_generator PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${ops_header_dir})
add_custom_command(
        OUTPUT ${ops_header_dir}/ascir_ops.h
        DEPENDS ascir_builtin_ops_header_generator
        COMMAND ${ops_header_dir}/ascir_builtin_ops_header_generator ${ops_header_dir}/ascir_ops.h
)
add_custom_target(ascir_builtin_ops_header ALL DEPENDS ${ops_header_dir}/ascir_ops.h)

add_library(ascir_ops_headers INTERFACE)
target_include_directories(ascir_ops_headers INTERFACE ${ops_header_dir})
add_dependencies(ascir_ops_headers ascir_builtin_ops_header)