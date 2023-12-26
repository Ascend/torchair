#include "ascir_utils.h"

#include <sstream>

using namespace ascir::utils;

void ascir::utils::DumpGraph(const ascir::Graph &graph, const std::string &name) {
}

static std::string IdentifierToStr(ascir::Identifier id) {
  if (id == ascir::ID_NONE) {
      return "(nil)";
  } else {
      return std::to_string(id);
  }
}

static std::string DtypeToStr(ge::DataType dtype) {
  const char *TypeName[] = {
    [ge::DT_FLOAT] = "float32",
    [ge::DT_FLOAT16] = "float16",
  };

  if (dtype >= sizeof(TypeName) / sizeof(TypeName[0])) {
      return "unknown";
  }
  return TypeName[dtype];
};

static std::string ApiTypeToStr(ascir::ApiType compute_type) {
  const char *TypeName[] = {
    [ascir::API_TYPE_BUFFER] = "Buffer",
    [ascir::API_TYPE_COMPUTE] = "Compute",
  };

  if (compute_type >= sizeof(TypeName) / sizeof(TypeName[0])) {
      return "unknown";
  }
  return TypeName[compute_type];
};

static std::string ComputeUnitToStr(ascir::ComputeUnit compute_unit) {
  const char *TypeName[] = {
    [ascir::UNIT_NONE] = "None",
    [ascir::UNIT_MTE] = "MTE",
    [ascir::UNIT_SCALAR] = "Scalar",
    [ascir::UNIT_VECTOR] = "Vector",
    [ascir::UNIT_CUBE] = "Cube",
  };

  if (compute_unit >= sizeof(TypeName) / sizeof(TypeName[0])) {
      return "unknown";
  }
  return TypeName[compute_unit];
}

static std::string AllocTypeToStr(ascir::AllocType alloc_type) {
  const char *TypeName[] = {
    [ascir::ALLOC_TYPE_GLOBAL] = "Global",
    [ascir::ALLOC_TYPE_BUFFER] = "Buffer",
    [ascir::ALLOC_TYPE_QUEUE] = "Queue",
  };

  if (alloc_type >= sizeof(TypeName) / sizeof(TypeName[0])) {
      return "unknown";
  }
  return TypeName[alloc_type];
}

static std::string MemHardwareToStr(ascir::MemHardware mem_hardware) {
  const char *TypeName[] = {
    [ascir::MEM_HARDWARE_GM] = "GM",
    [ascir::MEM_HARDWARE_UB] = "UB",
  };

  if (mem_hardware >= sizeof(TypeName) / sizeof(TypeName[0])) {
      return "unknown";
  }
  return TypeName[mem_hardware];
}

static std::string PositionToStr(ascir::Position position) {
  const char *TypeName[] = {
    [ascir::POSITION_GM] = "GM",
    [ascir::POSITION_VECIN] = "VECIN",
    [ascir::POSITION_VECOUT] = "VECOUT",
  };

  if (position >= sizeof(TypeName) / sizeof(TypeName[0])) {
      return "unknown";
  }
  return TypeName[position];
}

static std::stringstream &AxisListStr(std::stringstream& ss, const ascir::Graph &graph, const std::vector<ascir::AxisId> &axis_list) {
  for (auto axis_id : axis_list) {
    ss << graph.axis[axis_id].name << ", ";
  }
  return ss;
}

static std::string SizeExprStr(const ascir::Graph &graph, const ascir::SizeExpr &size_expr) {
  std::stringstream ss;

  if (size_expr.is_zero) {
    ss << "0";
    return ss.str();
  }

  if (size_expr.nums.size() == 0) {
    ss << "1";
  } else {
    ss << graph.size_var[size_expr.nums[0]].name;
    for (int i = 1; i < size_expr.nums.size(); ++i) {
      ss << '*' << graph.size_var[size_expr.nums[i]].name;
    }
  }

  for (auto den : size_expr.dens) {
    ss << "/" << graph.size_var[den].name;
  }

  return ss.str();
}

static std::stringstream &SizeExprListStr(std::stringstream &ss, const ascir::Graph &graph,
                                          const std::vector<ascir::SizeExpr> &size_expr_list) {
  for (auto size_expr: size_expr_list) {
    ss << SizeExprStr(graph, size_expr) << ", ";
  }
  return ss;
}

static std::stringstream &GraphNameStr(std::stringstream &ss, const ascir::Graph &graph) {
  ge::AscendString graph_name;
  graph.GetName(graph_name);
  ss << "Graph: " << graph_name.GetString() << std::endl;
  return ss;
}

static std::stringstream &GraphSizeStr(std::stringstream& ss, const ascir::Graph &graph) {
  ss << "Sizes:" << std::endl;
  for (auto size_var : graph.size_var()) {
    if (size_var.type == size_var.SIZE_TYPE_VAR) {
      ss << "  " << size_var.name << ": VAR" << std::endl;
    } else {
      ss << "  " << size_var.name << ": CONST(" << size_var.value << ")" << std::endl;
    }
  }

  return ss;
}

static std::stringstream &GraphAxisStr(std::stringstream &ss, const ascir::Graph &graph) {
  ss << "Axis:" << std::endl;
  for (auto axis : graph.axis()) {
    ss << "  " << axis.name << ": " << SizeExprStr(graph, axis.size) << ", ";

    const char *axis_type_str[] = {
      [ascir::Axis::AXIS_TYPE_ORIGINAL] = "ORIGINAL",
      [ascir::Axis::AXIS_TYPE_BLOCK_OUTER] = "BLOCK_OUT",
      [ascir::Axis::AXIS_TYPE_BLOCK_INNER] = "BLOCK_IN",
      [ascir::Axis::AXIS_TYPE_TILE_OUTER] = "TILE_OUT",
      [ascir::Axis::AXIS_TYPE_TILE_INNER] = "TILE_IN",
      [ascir::Axis::AXIS_TYPE_MERGED] = "MERGED"
    };
    ss << axis_type_str[axis.type];

    if (axis.from.size() > 0) {
      ss << ", from: ";
      for (auto from_axis : axis.from) {
        ss << graph.axis[from_axis].name << ", ";
      }
    }

    ss << std::endl;
  }
  return ss;
}

static std::stringstream &NodeAttrStr(std::stringstream &ss, const ascir::Graph &graph, ascir::NodeView &node,
                                      bool verbose = false) {
  // Node sched axis
  ss << "    .axis = "
     << "{";
  for (auto axis_id : node.attr.sched.axis()) {
    ss << graph.axis[axis_id].name << ", ";
  }
  ss << "}" << std::endl;

  if (verbose) {
    ss << "    .api:" << std::endl;
    ss << "      .type = " << ApiTypeToStr(node.attr.api.type) << std::endl;
    ss << "      .unit = " << ComputeUnitToStr(node.attr.api.unit) << std::endl;
  }

  return ss;
}

static std::stringstream &NodeInputStr(std::stringstream &ss, const ascir::Graph &graph, ascir::TensorPtr& input, bool verbose) {
  auto input_name = (*input).GetOwnerNode()->GetOpDesc()->GetInputNameByIndex((*input).GetIdx());
  if (input->Owner() == nullptr) {
    ss << "    ." << input_name << " = "
       << "(nil)" << std::endl;
  } else {
    auto peer_name = input->Owner()->GetName();
    auto peer_output_name = input->Owner()->GetOpDesc()->GetOutputNameByIndex(input->Index());
    ss << "    ." << input_name << " = " << peer_name << "." << peer_output_name << std::endl;
  }

  return ss;
}

static std::stringstream &NodeOutputStr(std::stringstream &ss, const ascir::Graph &graph, ascir::TensorView& output, bool verbose) {
  auto output_name = (*output).GetOwnerNode()->GetOpDesc()->GetOutputNameByIndex((*output).GetIdx());
  auto dtype = output->GetOwnerNode()->GetOpDesc()->GetOutputDesc(output->GetIdx()).GetDataType();
  ss << "    ." << output_name << ".dtype = " << DtypeToStr(dtype) << std::endl;

  ss << "    ." << output_name << ".axis = " << "{";
  AxisListStr(ss, graph, output.axis());
  ss << "}" << std::endl;

  ss << "    ." << output_name << ".repeats = " << "{";
  SizeExprListStr(ss, graph, output.repeats());
  ss << "}" << std::endl;

  ss << "    ." << output_name << ".strides = " << "{";
  SizeExprListStr(ss, graph, output.strides());
  ss << "}" << std::endl;

  ss << "    ." << output_name << ".vectorized_axis = " << "{";
  AxisListStr(ss, graph, output.vectorized_axis());
  ss << "}" << std::endl;

  if (verbose) {
      ss << "    ." << output_name << ".mem:" << std::endl;
      ss << "      .tensor_id = " << output.mem.tensor_id << std::endl;
      ss << "      .alloc_type = " << AllocTypeToStr(output.mem.alloc_type) << std::endl;
      ss << "      .hardware = " << MemHardwareToStr(output.mem.hardware) << std::endl;
      ss << "      .position = " << PositionToStr(output.mem.position) << std::endl;

      if (output.mem.alloc_type == ascir::ALLOC_TYPE_BUFFER) {
        ss << "    ." << output_name << ".buf:" << std::endl;
        ss << "      .id = " << output.buf.id << std::endl;
      } else if (output.mem.alloc_type == ascir::ALLOC_TYPE_QUEUE) {
        ss << "    ." << output_name << ".que:" << std::endl;
        ss << "      .id = " << output.que.id << std::endl;
        ss << "      .depth = " << output.que.depth << std::endl;
        ss << "      .buf_num = " << output.que.buf_num << std::endl;
      }

      ss << "    ." << output_name << ".opt:" << std::endl;
      ss << "      .ref_tensor = " << IdentifierToStr(output.opt.ref_tensor) << std::endl;
      ss << "      .merge_scope = " << IdentifierToStr(output.opt.merge_scope) << std::endl;
  }

  return ss;
}

std::string ascir::utils::DebugStr(ascir::Graph &graph, bool verbose) {
  std::stringstream ss;

  GraphNameStr(ss, graph);
  GraphSizeStr(ss, graph);
  GraphAxisStr(ss, graph);

  ss << "Nodes:" << std::endl;
  for (auto node : graph.GetAllNodes()) {
    // Node name and exec_order
    ss << "  " << node->GetName() << ": " << node->GetType() << " (" << node.attr.sched.exec_order << ")" << std::endl;

    NodeAttrStr(ss, graph, node, verbose);

    // Node inputs
    for (auto input : node.inputs()) {
      NodeInputStr(ss, graph, input, verbose);
    }

    // Node outputs
    for (auto output: node.outputs()) {
      NodeOutputStr(ss, graph, output, verbose);
    }
  }

  return ss.str();
}

std::string ascir::utils::DebugHintGraphStr(ascir::HintGraph &graph) {
  return DebugStr(graph, false);
}

std::string ascir::utils::DebugImplGraphStr(ascir::ImplGraph &graph) {
  return DebugStr(graph, true);
}
