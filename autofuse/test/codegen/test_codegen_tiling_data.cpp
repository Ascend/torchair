#include "gtest/gtest.h"

#include "codegen_tiling_data.h"

class TestCodegenTilingData : public testing::Test, public codegen::TilingData {
 protected:
  TestCodegenTilingData() : codegen::TilingData("TestKernel", "TestTilingData") {}
};

TEST_F(TestCodegenTilingData, ClassBegin) {
  EXPECT_EQ(this->ClassBegin(), "BEGIN_TILING_DATA_DEF(TestTilingData)");
}

TEST_F(TestCodegenTilingData, ClassEnd) {
  EXPECT_EQ(this->ClassEnd(), "END_TILING_DATA_DEF;");
}

TEST_F(TestCodegenTilingData, DataFieldDefine) {
  ascir::SizeVar s0{0, "s0"};
  EXPECT_EQ(this->DataFieldDefine(s0), "TILING_DATA_FIELD_DEF(uint32_t, s0);");
}

TEST_F(TestCodegenTilingData, ClassRegister) {
  EXPECT_EQ(this->ClassRegister(), "REGISTER_TILING_DATA_CLASS(TestKernel, TestTilingData)");
}
