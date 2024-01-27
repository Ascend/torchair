#ifndef AOE_EXTERNAL_AOE_H
#define AOE_EXTERNAL_AOE_H

#include <map>
#include "ge/ge_api.h"
#include "graph/ascend_string.h"

namespace Aoe {
using AoeStatus = int32_t;

extern "C" AoeStatus AoeInitialize(const std::map<ge::AscendString, ge::AscendString> &globalOptions);

extern "C" AoeStatus AoeFinalize();

extern "C" AoeStatus AoeCreateSession(uint64_t &sessionKey);

extern "C" AoeStatus AoeDestroySession(uint64_t sessionKey);

extern "C" AoeStatus AoeSetGeSession(uint64_t sessionKey, ge::Session *geSession);

extern "C" AoeStatus AoeSetDependGraphs(uint64_t sessionKey, const std::vector<ge::Graph> &dependGraphs);

extern "C" AoeStatus AoeSetDependGraphsInputs(uint64_t sessionKey, const std::vector<std::vector<ge::Tensor>> &inputs);

extern "C" AoeStatus AoeSetTuningGraph(uint64_t sessionKey, const ge::Graph &tuningGraph);

extern "C" AoeStatus AoeSetTuningGraphInput(uint64_t sessionKey, const std::vector<ge::Tensor> &input);

extern "C" AoeStatus AoeTuningGraph(uint64_t sessionKey, const std::map<ge::AscendString, ge::AscendString> &tuningOptions);

} // namespace Aoe
#endif