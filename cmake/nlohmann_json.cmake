include(FetchContent)
if (CI_PKG_SERVER)
    FetchContent_Declare(
            nlohmann_json
            URL https://ascend-cann.obs.myhuaweicloud.com/json/repository/archive/json-v3.10.1.zip
    )
else ()
    FetchContent_Declare(
            nlohmann_json
            URL https://ascend-cann.obs.myhuaweicloud.com/json/repository/archive/json-v3.10.1.zip
    )
endif ()
FetchContent_GetProperties(nlohmann_json)
if (NOT nlohmann_json_POPULATED)
    FetchContent_Populate(nlohmann_json)
    include_directories(${nlohmann_json_SOURCE_DIR}/include)
endif ()
