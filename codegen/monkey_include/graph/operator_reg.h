#ifndef CODEGEN_OPERATOR_REG_H_
#define CODEGEN_OPERATOR_REG_H_

#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <iostream>
#include <unordered_set>
#include <regex>

const std::string k4Space = "    ";
const char kEnd = '\n';

const std::string kTensorTypeUnknown = "TensorType.TT_UNKNOWN";
const std::string kTensorTypeAll = "TensorType.TT_ALL";
const std::string kTensorTypeQuantified = "TensorType.TT_QUANTIFIED";
const std::string kTensorTypeOrdinary = "TensorType.TT_ORDINARY";
const std::string kTensorTypeBasic = "TensorType.TT_BASIC";
const std::string kTensorTypeNumber = "TensorType.TT_NUMBER";
const std::string kTensorTypeRealNumber = "TensorType.TT_REAL_NUMBER";
const std::string kTensorTypeComplex = "TensorType.TT_COMPLEX";
const std::string kTensorTypeInteger = "TensorType.TT_INTEGER";
const std::string kTensorTypeSigned = "TensorType.TT_SIGNED";
const std::string kTensorTypeUnsigned = "TensorType.TT_UNSIGNED";
const std::string kTensorTypeFloating = "TensorType.TT_FLOATING";
const std::string kTensorTypeIndexNumber = "TensorType.TT_INDEX_NUMBER";
const std::string kTensorTypeUnary = "TensorType.TT_UNARY";
const std::string kTensorTypeFloat = "TensorType.TT_FLOAT";

const std::vector<std::pair<std::string, std::string>> kTensorTypeMap{
  {"QuantifiedType", kTensorTypeQuantified},
  {"OrdinaryType", kTensorTypeOrdinary},
  {"BasicType", kTensorTypeBasic},
  {"RealNumberType", kTensorTypeRealNumber},
  {"ComplexDataType", kTensorTypeComplex},
  {"IntegerDataType", kTensorTypeInteger},
  {"SignedDataType", kTensorTypeSigned},
  {"UnsignedDataType", kTensorTypeUnsigned},
  {"FloatingDataType", kTensorTypeFloating},
  {"IndexNumberType", kTensorTypeIndexNumber},
  {"UnaryDataType", kTensorTypeUnary},
  {"NumberType", kTensorTypeNumber},
  {"FLOAT", kTensorTypeFloat},
  {"ALL", kTensorTypeAll},
};

std::string Brack(const std::string& v) { return R"(")" + v + R"(")"; }

std::string GetSig(const std::string& name) {
  const static std::unordered_set<std::string> kPythonReserved = {
    "False", "None",     "True",  "and",    "as",   "assert", "async",  "await",    "break",
    "class", "continue", "def",   "del",    "elif", "else",   "except", "finally",  "for",
    "from",  "global",   "if",    "import", "in",   "is",     "lambda", "nonlocal", "not",
    "or",    "pass",     "raise", "return", "try",  "while",  "with",   "yield"};
  if (kPythonReserved.count(name)) {
    return name + "_changed_as_is_python_key";
  }
  return name;
}

enum DataType {
  DT_FLOAT = 0,            // float type
  DT_FLOAT16 = 1,          // fp16 type
  DT_INT8 = 2,             // int8 type
  DT_INT16 = 6,            // int16 type
  DT_UINT16 = 7,           // uint16 type
  DT_UINT8 = 4,            // uint8 type
  DT_INT32 = 3,            //
  DT_INT64 = 9,            // int64 type
  DT_UINT32 = 8,           // unsigned int32
  DT_UINT64 = 10,          // unsigned int64
  DT_BOOL = 12,            // bool type
  DT_DOUBLE = 11,          // double type
  DT_STRING = 13,          // string type
  DT_DUAL_SUB_INT8 = 14,   // dual output int8 type
  DT_DUAL_SUB_UINT8 = 15,  // dual output uint8 type
  DT_COMPLEX64 = 16,       // complex64 type
  DT_COMPLEX128 = 17,      // complex128 type
  DT_QINT8 = 18,           // qint8 type
  DT_QINT16 = 19,          // qint16 type
  DT_QINT32 = 20,          // qint32 type
  DT_QUINT8 = 21,          // quint8 type
  DT_QUINT16 = 22,         // quint16 type
  DT_RESOURCE = 23,        // resource type
  DT_STRING_REF = 24,      // string ref type
  DT_DUAL = 25,            // dual output type
  DT_VARIANT = 26,         // dt_variant type
  DT_BF16 = 27,            // bf16 type
  DT_UNDEFINED = 28,       // Used to indicate a DataType field has not been set.
  DT_INT4 = 29,            // int4 type
  DT_UINT1 = 30,           // uint1 type
  DT_INT2 = 31,            // int2 type
  DT_UINT2 = 32,           // uint2 type
  DT_MAX                   // Mark the boundaries of data types
};

class TensorType {};

using Int = int;
using Float = float;
using Bool = bool;
using String = std::string;
using Type = DataType;
using Tensor = TensorType;

using ListInt = std::vector<int>;
using ListType = std::vector<Type>;
using ListListInt = std::vector<std::vector<int>>;
using ListFloat = std::vector<float>;
using ListListFloat = std::vector<std::vector<float>>;
using ListBool = std::vector<bool>;
using ListListBool = std::vector<std::vector<bool>>;
using ListString = std::vector<std::string>;
using ListListString = std::vector<std::vector<std::string>>;

template <typename T>
bool IsEnd(const T& v) {
  return false;
}

template <>
bool IsEnd(const char& v) {
  return v == kEnd;
}

class Code : public std::stringstream {
 public:
  template <typename T>
  Code& operator<<(const T& v) {
    if (new_line_) {
      ss_ << indent_ << v;
    } else {
      ss_ << v;
    }
    new_line_ = IsEnd(v);
    return *this;
  }

  void Indent() { indent_ += k4Space; }

  void Dedent() {
    if (indent_.size() >= 4U) {
      indent_ = indent_.substr(0, indent_.size() - 4U);
    }
  }

  std::string str() { return ss_.str(); }

 private:
  std::stringstream ss_;
  std::string indent_;
  bool new_line_ = false;
};

struct InputDef {
  std::string name;
  std::string sig;
  bool is_dynamic = false;
  bool is_optional = false;
  std::string tensorType = "TensorType.TT_UNKNOWN";
  std::string tensorTypeStr;
  std::string TypeIndicator() const {
    if (is_dynamic) {
      return "List[Tensor]";
    }
    if (is_optional) {
      return "Optional[Tensor]";
    }
    return "Tensor";
  }
  void GenSig(Code& ss) const { ss << sig << ": " << TypeIndicator(); }
  void GenCode(Code& ss) const {
    ss << k4Space << Brack(name) << ": " << sig << ",";
  }
  void GenIrDefCode(Code& ss) const {
    if (is_dynamic) {
      ss << k4Space << ".dynamic_input(" << Brack(name) << ", " << Brack(tensorTypeStr) << ") \\";
      return;
    }
    if (is_optional) {
      ss << k4Space << ".optional_input(" << Brack(name) << ", " << Brack(tensorTypeStr) << ") \\";
      return;
    }
    ss << k4Space << ".input(" << Brack(name) << ", " << Brack(tensorTypeStr) << ") \\";
    return;
  }
};

struct OutputDef {
  std::string name;
  std::string sig;
  bool is_dynamic = false;
  std::string tensorTypeStr;
  void GenSig(Code& ss) const {
    if (is_dynamic) {
      ss << "size_of_" << sig << ": int";
    }
  }
  void GenCode(Code& ss) const {
    if (is_dynamic) {
      ss << "(" << Brack(name) << ", size_of_" << sig << "),";
    } else {
      ss << Brack(name) << ",";
    }
  }

  void GenIrDefCode(Code& ss) const {
    if (is_dynamic) {
      ss << k4Space << ".dynamic_output(" << Brack(name) << " , " << Brack(tensorTypeStr) << ") \\";
    } else {
      ss << k4Space << ".output(" << Brack(name) << " , " << Brack(tensorTypeStr) << ") \\";
    }
  }
};

namespace {
// enum ListValueType{
//   VT_LIST_NONE = 0;
//   VT_LIST_STRING = 1;
//   VT_LIST_INT = 2;
//   VT_LIST_FLOAT = 3;
//   VT_LIST_BOOL = 4;
//   VT_LIST_BYTES = 5;
//   VT_LIST_TENSOR_DESC = 6;
//   VT_LIST_TENSOR = 7;
//   VT_LIST_GRAPH = 8;
//   VT_LIST_NAMED_ATTRS = 9;
//   VT_LIST_DATA_TYPE = 10;
// }
inline std::string GetListValueType(const std::string& proto) {
  if (proto == "list.i") {
    return "2";
  }
  if (proto == "list.f") {
    return "3";
  }
  if (proto == "list.b") {
    return "4";
  }
  if (proto == "list.t") {
    return "7";
  }
  if (proto == "list.dt") {
    return "10";
  }
  return "0";
}
}  // namespace

struct AttrDef {
  std::string name;
  std::string sig;
  std::string type;
  std::string proto;
  void GenSig(Code& ss) const { ss << sig << ": " << type; }
  std::string GenAttrCode() const {
    if (proto == "i") {
      return "Int";
    }
    if (proto == "f") {
      return "Float";
    }
    if (proto == "b") {
      return "Bool";
    }
    if (proto == "dt") {
      return "DataType";
    }
    if (proto == "list.i") {
      return "ListInt";
    }
    if (proto == "list.f") {
      return "ListFloat";
    }
    if (proto == "list.b") {
      return "ListBool";
    }
    if (proto == "list.dt") {
      return "ListDataType";
    }
    if (proto == "s") {
      return "Str";
    }
    if (proto == "list.s") {
      return "ListStr";
    }
    if (proto == "list_list_int") {
      return "ListListInt";
    }
    if (proto == "list_list_float") {
      return "ListListFloat";
    }
  }

  void GenCode(Code& ss) const {
    ss << k4Space << Brack(name) << ": attr." << GenAttrCode() << "(" << sig << "),";
  }

  void GenIrDefCode(Code& ss) const {
    ss << k4Space << ".required_attr(" << Brack(name) << ", attr." << GenAttrCode() << ") \\";
  }
};

struct AttrDefWithDefault : public AttrDef {
  std::string value;
  void GenSig(Code& ss) const { ss << sig << ": " << type << "=" << value; }
  void GenIrDefCode(Code& ss) const {
    ss << k4Space << ".attr(" << Brack(name) << ", attr." << GenAttrCode() << "(" << value << ")) \\";
  }
};

struct OpDef {
  static std::vector<OpDef> defs;
  std::string err;
  std::string op;
  std::string doc;
  std::vector<InputDef> inputs;
  std::vector<OutputDef> outputs;
  std::vector<AttrDef> attrs;
  std::vector<AttrDefWithDefault> attrs_with_default;

  void GenWrap(Code& ss) {
    const static auto kDynamicStr = [](const std::vector<InputDef>& inputs) {
      std::stringstream ss;
      if (inputs.empty()) {
        return ss.str();
      }
      for (size_t i = 0U; i < inputs.size() - 1U; ++i) {
        ss << (inputs[i].is_dynamic ? "True" : "False") << ", ";
      }
      ss << (inputs.rbegin()->is_dynamic ? "True" : "False");
      return ss.str();
    };
    const static auto kOptionalStr = [](const std::vector<InputDef>& inputs) {
      std::stringstream ss;
      if (inputs.empty()) {
        return ss.str();
      }
      for (size_t i = 0U; i < inputs.size() - 1U; ++i) {
        ss << (inputs[i].is_optional ? "True" : "False") << ", ";
      }
      ss << (inputs.rbegin()->is_optional ? "True" : "False");
      return ss.str();
    };
    const static auto kTensorTypeStr = [](const std::vector<InputDef>& inputs) {
      std::stringstream ss;
      if (inputs.empty()) {
        return ss.str();
      }
      for (size_t i = 0U; i < inputs.size() - 1U; ++i) {
        ss << inputs[i].tensorType << ", ";
      }
      ss << inputs.rbegin()->tensorType;
      return ss.str();
    };

    bool needTensorType = false;
    for (auto it = inputs.begin(); it != inputs.end(); ++it) {
      if (it->tensorType != kTensorTypeUnknown) {
        needTensorType = true;
        break;
      }
    }
    if (needTensorType) {
      ss << "@auto_convert_to_tensor([" << kDynamicStr(inputs) << "], [" << kOptionalStr(inputs)
         << "], inputs_tensor_type=[" << kTensorTypeStr(inputs) << "])" << kEnd;
    } else {
      ss << "@auto_convert_to_tensor([" << kDynamicStr(inputs) << "], [" << kOptionalStr(inputs) << "])" << kEnd;
    }
  }

  bool IsDynamicOutput() const {
    for (auto& output : outputs) {
      if (output.is_dynamic) {
        return true;
      }
    }
    return false;
  }

  void GenSig(Code& ss) {
    GenWrap(ss);
    ss << "def " << (IsDynamicOutput() ? "_" : "") << op << "(";
    for (auto& input : inputs) {
      input.GenSig(ss);
      ss << ", ";
    }
    ss << "*, ";
    for (auto& output : outputs) {
      if (output.is_dynamic) {
        output.GenSig(ss);
        ss << ", ";
      }
    }
    for (auto& attr : attrs) {
      attr.GenSig(ss);
      ss << ", ";
    }
    for (auto& attr : attrs_with_default) {
      attr.GenSig(ss);
      ss << ", ";
    }
    ss << "dependencies=[], node_name=None):" << kEnd;
  }

  void GenDoc(Code& ss) const {
    ss << kEnd;
    ss << "# This api is auto-generated from IR " << op << kEnd;
  }

  void GenReturn(Code& ss) const {
    ss << "return";
    if (outputs.empty()) {
      return;
    }
    ss << " ";
    const size_t num = outputs.size();
    for (size_t i = 0U; i < num - 1U; i++) {
      ss << outputs[i].name << ", ";
    }
    ss << outputs[num - 1].name;
  }

  void GenNewLine(Code& ss) {
    ss.Dedent();
    ss << kEnd;
    ss.Indent();
  }

  void GenCode(Code& ss) {
    ss.Indent();
    ss << R"(""")" << doc << R"(""")" << kEnd;

    GenNewLine(ss);
    ss << "# process inputs" << kEnd;
    ss << "inputs = {" << kEnd;
    for (auto& input : inputs) {
      input.GenCode(ss);
      ss << kEnd;
    }
    ss << "}" << kEnd;

    GenNewLine(ss);
    ss << "# process attrs" << kEnd;
    ss << "attrs = {" << kEnd;
    for (auto& attr : attrs) {
      attr.GenCode(ss);
      ss << kEnd;
    }
    for (auto& attr : attrs_with_default) {
      attr.GenCode(ss);
      ss << kEnd;
    }
    ss << "}" << kEnd;

    GenNewLine(ss);
    ss << "# process outputs" << kEnd;
    ss << "outputs = [" << kEnd;
    for (auto& output : outputs) {
      output.GenCode(ss);
      ss << kEnd;
    }
    ss << "]" << kEnd;

    GenNewLine(ss);
    ss << "return ge_op(" << kEnd;
    ss << k4Space << "op_type=" << Brack(op) << "," << kEnd;
    ss << k4Space << "inputs=inputs," << kEnd;
    ss << k4Space << "attrs=attrs," << kEnd;
    ss << k4Space << "outputs=outputs," << kEnd;
    ss << k4Space << "dependencies=dependencies," << kEnd;
    ss << k4Space << "ir=";
    GenIrDef(ss);
    ss << ")" << kEnd;
    ss.Dedent();
  }

  void GenIrDef(Code& ss) {
    Code buffer;
    buffer << "IrDef(" << Brack(op) << ") \\" << kEnd;
    buffer.Indent();
    for (auto& input : inputs) {
      input.GenIrDefCode(buffer);
      buffer << kEnd;
    }

    for (auto& attr : attrs) {
      attr.GenIrDefCode(buffer);
      buffer << kEnd;
    }
    for (auto& attr : attrs_with_default) {
      attr.GenIrDefCode(buffer);
      buffer << kEnd;
    }

    for (auto& output : outputs) {
      output.GenIrDefCode(buffer);
      buffer << kEnd;
    }
    std::string temp = buffer.str();
    // 删除尾部的反斜杠
    temp.pop_back();
    temp.pop_back();
    temp.pop_back();
    ss << temp << kEnd;
  }

  void Gen(Code& ss) {
    std::unordered_set<std::string> input_names;
    for (auto& input : inputs) {
      input_names.insert(input.name);
      input.sig = GetSig(input.name);
    }
    for (auto& output : outputs) {
      output.sig = GetSig(output.name);
    }
    for (auto& attr : attrs) {
      if (input_names.count(attr.name)) {
        attr.sig = attr.name + "_changed_as_duplicate_with_input";
      } else {
        attr.sig = GetSig(attr.name);
      }
    }
    for (auto& attr : attrs_with_default) {
      if (input_names.count(attr.name)) {
        attr.sig = attr.name + "_changed_as_duplicate_with_input";
      } else {
        attr.sig = GetSig(attr.name);
      }
    }

    GenDoc(ss);
    GenSig(ss);
    GenCode(ss);
  }

  const std::string& Err() const { return err; }
};

std::vector<OpDef> OpDef::defs;

template <typename T>
struct Value;

template <>
struct Value<int> {
  static std::string ToString(int v) { return std::to_string(v); }
  static std::string Type() { return "int"; }
  static std::string Proto() { return "i"; }
};

template <>
struct Value<float> {
  static std::string ToString(float v) { return std::to_string(v); }
  static std::string Type() { return "float"; }
  static std::string Proto() { return "f"; }
};

template <>
struct Value<bool> {
  static std::string ToString(bool v) { return v ? "True" : "False"; }
  static std::string Type() { return "bool"; }
  static std::string Proto() { return "b"; }
};

template <>
struct Value<std::string> {
  static std::string ToString(const std::string& v) { return "\"" + v + "\""; }
  static std::string Type() { return "str"; }
  static std::string Proto() { return "s"; }
};

template <>
struct Value<Type> {
  static std::string ToString(const Type& v) {
    if (v == Type::DT_FLOAT) return "DataType.DT_FLOAT";
    if (v == Type::DT_INT32) return "DataType.DT_INT32";
    if (v == Type::DT_UINT8) return "DataType.DT_UINT8";
    if (v == Type::DT_INT16) return "DataType.DT_INT16";
    if (v == Type::DT_INT8) return "DataType.DT_INT8";
    if (v == Type::DT_STRING) return "DataType.DT_STRING";
    if (v == Type::DT_INT64) return "DataType.DT_INT64";
    if (v == Type::DT_BOOL) return "DataType.DT_BOOL";
    if (v == Type::DT_UINT16) return "DataType.DT_UINT16";
    if (v == Type::DT_RESOURCE) return "DataType.DT_RESOURCE";
    if (v == Type::DT_VARIANT) return "DataType.DT_VARIANT";
    if (v == Type::DT_UINT32) return "DataType.DT_UINT32";
    if (v == Type::DT_UINT64) return "DataType.DT_UINT64";
    return std::to_string(v);
  }
  static std::string Type() { return "int"; }
  static std::string Proto() { return "dt"; }
};

template <>
struct Value<Tensor> {
  static std::string ToString(const Tensor& v) { return "None"; }
  static std::string Type() { return "Any"; }
  static std::string Proto() { return "t"; }
};

template <typename T>
struct Value<std::vector<T>> {
  static std::string ToString(const std::vector<T>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
      if (i > 0) s += ", ";
      s += Value<T>::ToString(v[i]);
    }
    s += "]";
    return s;
  }
  static std::string Type() { return "List[" + Value<T>::Type() + "]"; }
  static std::string Proto() { return "list." + Value<T>::Proto(); }
};

template <typename T>
struct Value<std::vector<std::vector<T>>> {
  static std::string ToString(const std::vector<std::vector<T>>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
      if (i > 0) s += ", ";
      s += Value<std::vector<T>>::ToString(v[i]);
    }
    s += "]";
    return s;
  }
  static std::string Type() { return "List[" + Value<std::vector<T>>::Type() + "]"; }
  static std::string Proto() { return "list_list_" + Value<T>::Type(); }
};

AttrDef BuildAttrDef(const std::string& name, const std::string& type, const std::string& proto) {
  AttrDef def;
  def.name = name;
  def.type = type;
  def.proto = proto;
  return def;
}

template <typename T>
AttrDefWithDefault BuildAttrDefWithDefault(const std::string& name, const std::string& type, const std::string& proto,
                                           const T& v) {
  AttrDefWithDefault def;
  def.name = name;
  def.type = type;
  def.proto = proto;
  def.value = Value<T>::ToString(v);
  return def;
}

class OpDefBuilder {
 public:
  explicit OpDefBuilder(const std::string& name) {
    const static std::unordered_set<std::string> kBypass = {
      "Const", "Data", "Constant", "Cast", "Variable", "VariableV2", "Placeholder", "PlaceholderV2",
      "PlaceholderWithDefault"};

    if (kBypass.count(name)) {
      def_.err = "Deformed prototype";
    }
    def_.op = name;
  }
  OpDefBuilder& Bypass() { return *this; }
  OpDefBuilder& Unsupported(const std::string& err) {
    if (def_.err.empty()) {
      def_.err = err;
    }
    return *this;
  }
  OpDefBuilder &Input(const std::string &input, const std::string &tensorType, bool is_dynamic = false,
                      bool is_optional = false) {
    InputDef def;
    def.name = input;
    def.is_dynamic = is_dynamic;
    def.is_optional = is_optional;
    def.tensorType = GenTensorTypeStr(tensorType);
    def.tensorTypeStr = GenTensorTypeIrStr(tensorType);
    
    def_.inputs.push_back(def);
    return *this;
  }
  OpDefBuilder& Output(const std::string& input, const std::string &tensorType, bool is_dynamic = false) {
    OutputDef def;
    def.name = input;
    def.is_dynamic = is_dynamic;
    def.tensorTypeStr = GenTensorTypeIrStr(tensorType);
    def_.outputs.push_back(def);
    return *this;
  }

  OpDefBuilder& Attr(const std::string& attr, const std::string& type, const std::string& proto) {
    def_.attrs.push_back(BuildAttrDef(attr, type, proto));
    return *this;
  }

  template <typename T>
  OpDefBuilder& AttrWithDefault(const std::string& attr, const std::string& type, const std::string& proto,
                                const T& v) {
    def_.attrs_with_default.push_back(BuildAttrDefWithDefault(attr, type, proto, v));
    return *this;
  }

  OpDefBuilder& Record(const std::string& str) {
    ss_ << str << "\\n" << kEnd;
    return *this;
  }

  int Build() {
    def_.doc = ss_.str();
    OpDef::defs.push_back(def_);
    return 0;
  }

 private:
  std::string GenTensorTypeStr(const std::string &s) {
    if (s.find("DT_") != std::string::npos) {
      std::regex regex("\\s+");
      auto _s = std::regex_replace(s, regex, "");
      if (_s == "TensorType({DT_INT32,DT_INT64})" || _s == "TensorType({DT_INT64,DT_INT32})") {
        return kTensorTypeIndexNumber;
      }
      return kTensorTypeUnknown;
    }

    for (auto it = kTensorTypeMap.begin(); it != kTensorTypeMap.end(); ++it) {
      if (s.find(it->first) != std::string::npos) {
        return it->second;
      }
    }

    return kTensorTypeUnknown;
  }

  std::string GenTensorTypeIrStr(const std::string &str) {
    size_t start = str.find("{") + 1;
    size_t end = str.rfind("}");
    if (start != std::string::npos && end != std::string::npos && start < end) {
        return str.substr(start, end - start);
    } else {
        return "";
    }
  }

  OpDef def_;
  std::stringstream ss_;
  std::string err_;
};

#define REG_OP_COUNTER2(type, counter) \
  static auto g_register_kernel_##counter = OpDefBuilder(#type).Record("REG_OP(" #type ")")
#define REG_OP_COUNTER(type, counter) REG_OP_COUNTER2(type, counter)
#define REG_OP(type) REG_OP_COUNTER(type, __COUNTER__)

#define CONCATENATE_STR(...) #__VA_ARGS__

#define INPUT(x, t) Input(#x, #t).Record(".INPUT(" CONCATENATE_STR(x, t) ")")
#define DYNAMIC_INPUT(x, t) Input(#x, #t, true).Record(".DYNAMIC_INPUT(" CONCATENATE_STR(x, t) ")")
#define OPTIONAL_INPUT(x, t) Input(#x, #t, false, true).Record(".OPTIONAL_INPUT(" CONCATENATE_STR(x, t) ")")
#define OUTPUT(x, t) Output(#x, #t).Record(".OUTPUT(" CONCATENATE_STR(x, t) ")")
#define DYNAMIC_OUTPUT(x, t) Output(#x, #t, true).Record(".DYNAMIC_OUTPUT(" CONCATENATE_STR(x, t) ")")
#define ATTR(x, T, ...)                                                    \
  AttrWithDefault(#x, Value<T>::Type(), Value<T>::Proto(), T(__VA_ARGS__)) \
    .Record(".ATTR(" CONCATENATE_STR(x, T, __VA_ARGS__) ")")
#define REQUIRED_ATTR(x, T) \
  Attr(#x, Value<T>::Type(), Value<T>::Proto()).Record(".REQUIRED_ATTR(" CONCATENATE_STR(x, T) ")")
#define OP_END_FACTORY_REG(x) Build();

#define DATATYPE(...) Bypass()
#define OPTIONAL_OUTPUT(...) Unsupported("optional output")
#define GRAPH(...) Unsupported("graph")
#define DYNAMIC_GRAPH(...) Unsupported("dynamic graph")

namespace ge {
static auto UNKNOWN_SHAPE = {-1};
static auto UNKNOWN_RANK = {-2};
}  // namespace ge

#endif  // CODEGEN_OPERATOR_REG_H_