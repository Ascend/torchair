#ifndef __CODEGEN_TILING_DATA_H__
#define __CODEGEN_TILING_DATA_H__

#include <string>

#include "ascir.h"

namespace codegen {

class TilingData {
 public:
  explicit TilingData(const std::string& kernel_name, const std::string& class_name = "TilingData");
  std::string Generate(const ascir::ImplGraph& graph);

 protected:
  std::string class_name;
  std::string kernel_name;

  static const std::string MacrosAndIncludes;

  std::string ClassBegin();
  std::string DataFieldDefine(const ascir::SizeVar& size);
  std::string ClassEnd();
  std::string ClassRegister();
};
};

#endif
