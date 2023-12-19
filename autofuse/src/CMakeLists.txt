add_subdirectory(ascir)
add_subdirectory(optimize)
add_subdirectory(codegen)
add_subdirectory(optimize/autoschedule)

add_subdirectory(python)

add_library(autofuse SHARED autofuse.cpp)
target_link_libraries(autofuse ascir optimize codegen)

