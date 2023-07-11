include(FetchContent)
if(TF_PKG_SERVER)
  FetchContent_Declare(
          googletest
          URL ${TF_PKG_SERVER}/libs/ge_gtest/release-1.8.1.tar.gz
          URL_HASH MD5=2e6fbeb6a91310a16efe181886c59596
  )
else()
  FetchContent_Declare(
          googletest
          URL https://gitee.com/mirrors/googletest/repository/archive/release-1.8.1.tar.gz
          URL_HASH MD5=2e6fbeb6a91310a16efe181886c59596
  )
endif()

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
