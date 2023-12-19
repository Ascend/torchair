find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

Python3_add_library(pyautofuse MODULE
    pyascir.cpp pyascir_types.cpp
    pyautofuse.cpp)
target_compile_options(pyautofuse PRIVATE -pedantic)
target_link_libraries(pyautofuse PUBLIC autofuse)
