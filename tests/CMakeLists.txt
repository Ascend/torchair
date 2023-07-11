cmake_minimum_required(VERSION 3.14)
project(TorchairTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-D_FORTIFY_SOURCE=2 -Wno-class-memaccess -Wno-float-equal -Wextra -Wno-unused-parameter -Wno-ignored-qualifiers -Wno-comment -Wno-deprecated-declarations -Wall -fPIC -fstack-protector-all -Wl,--no-as-needed -Wl,-z,relro,-z,now,-z,noexecstack -fno-common -pipe -fno-strict-aliasing -Wdate-time -Wformat=2 -Wno-shadow -Wno-undef -Wunused -Wdelete-non-virtual-dtor -Wnon-virtual-dtor -Wno-overloaded-virtual ${CMAKE_CXX_FLAGS}")

set(TORCHAIR_SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/../)

file(STRINGS "${TORCHAIR_SRC_DIR}/tools/TORCH_INSTALLED_PATH" TORCH_INSTALLED_PATH)

add_subdirectory(${TORCHAIR_SRC_DIR} torchair)
add_subdirectory(ut)
add_subdirectory(st)
