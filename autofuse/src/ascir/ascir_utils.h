#ifndef ASCIR_UTILS_HPP
#define ASCIR_UTILS_HPP

#include <string>
#include "ascir.h"

namespace ascir::utils {
/**
 * @brief Dumps the graph to a pbtxt file
 */
void DumpGraph(const ascir::Graph &graph, const std::string &dump_name);

/**
 * @brief Prints nodes info in graph.
 */
std::string DebugStr(const ascir::Graph &graph, bool verbose = false);

std::string DebugHintGraphStr(const ascir::HintGraph &graph);
std::string DebugImplGraphStr(const ascir::ImplGraph &graph);

std::string IdentifierToStr(ascir::Identifier id);
};  // namespace ascir::utils

#endif
