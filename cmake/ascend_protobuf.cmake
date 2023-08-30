include(FetchContent)
if (CI_PKG_SERVER)
    FetchContent_Declare(
            ascend_protobuf
            URL ${CI_PKG_SERVER}/libs/protobuf/v3.13.0.tar.gz
    )
else ()
    FetchContent_Declare(
            ascend_protobuf
            URL https://gitee.com/mirrors/protobuf_source/repository/archive/v3.13.0.tar.gz
            URL_HASH MD5=53ab10736257b3c61749de9800b8ce97

    )
endif ()
FetchContent_GetProperties(ascend_protobuf)
if (NOT ascend_protobuf_POPULATED)
    FetchContent_Populate(ascend_protobuf)
    message("ascend_protobuf_SOURCE_DIR:"${ascend_protobuf_SOURCE_DIR})
    include_directories(${ascend_protobuf_SOURCE_DIR}/src)
endif ()