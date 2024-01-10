#include "codegen_common.h"

#include <sstream>

std::string codegen::CamelToLowerSneak(const std::string &str) {
  std::stringstream ss;
  bool is_first = true;

  for (auto c : str) {
    if (isupper(c)) {
      if (is_first) {
        is_first = false;
      } else {
        ss << "_";
      }
      ss << char(tolower(c));
    } else {
      ss << c;
    }
  }

  return ss.str();
}
