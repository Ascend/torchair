#include "aoe.h"

namespace Aoe {
AoeStatus AoeInitialize(const std::map<ge::AscendString, ge::AscendString> &globalOptions) {
  return 0;
}

AoeStatus AoeFinalize() {
  return 0;
}

AoeStatus AoeCreateSession(uint64_t &sessionKey) {
  return 0;
}

AoeStatus AoeDestroySession(uint64_t sessionKey) {
  return 0;
}

AoeStatus AoeSetGeSession(uint64_t sessionKey, ge::Session *geSession) {
  return 0;
}

AoeStatus AoeSetDependGraphs(uint64_t sessionKey, const std::vector<ge::Graph> &dependGraphs) {
  return 0;
}

AoeStatus AoeSetDependGraphsInputs(uint64_t sessionKey, const std::vector<std::vector<ge::Tensor>> &inputs) {
  return 0;
}

AoeStatus AoeSetTuningGraph(uint64_t sessionKey, const ge::Graph &tuningGraph) {
  return 0;
}

AoeStatus AoeSetTuningGraphInput(uint64_t sessionKey, const std::vector<ge::Tensor> &input) {
  return 0;
}

AoeStatus AoeTuningGraph(uint64_t sessionKey, const std::map<ge::AscendString, ge::AscendString> &tuningOptions) {
  return 0;
}

}