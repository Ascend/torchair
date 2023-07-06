include(FetchContent)
if (CI_PKG_SERVER)
    FetchContent_Declare(
            secure_c
            URL ${CI_PKG_SERVER}/libs/securec/v1.1.10.tar.gz
    )
else ()
    FetchContent_Declare(
            secure_c
            URL https://gitee.com/openeuler/libboundscheck/repository/archive/v1.1.10.tar.gz
    )
endif ()
FetchContent_GetProperties(secure_c)
if (NOT secure_c_POPULATED)
    FetchContent_Populate(secure_c)
    include_directories(${secure_c_SOURCE_DIR}/include)
    file(GLOB SEC_C_SRCS ${secure_c_SOURCE_DIR}/src/*.c)
    add_library(c_sec SHARED ${SEC_C_SRCS})
endif ()