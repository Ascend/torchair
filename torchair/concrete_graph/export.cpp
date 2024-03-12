#include <cstdarg>

#include "concrete_graph.h"

#include "graph/types.h"
#include "ge/ge_api.h"

#include "checker.h"
#include "compat_apis.h"
#include "executor.h"
#include "graph_data.h"
#include "logger.h"
#include "session.h"
#include "utils.h"

namespace tng {
namespace ep {
  Status Export(const void *serialized_proto, size_t proto_size,
                const std::map<ge::AscendString, ge::AscendString> &options) {
    TNG_LOG(INFO) << "Creating concrete graph from proto with size " << proto_size;
    TNG_ASSERT_NOTNULL(serialized_proto, "Given serialized proto is nullptr.");
    TNG_ASSERT(ge::IntegerChecker<int32_t>::Compat(proto_size), "Proto size %zu exceed 2G limit.", proto_size);

    static ge::AscendString export_path_key("export_path_dir");
    static ge::AscendString export_name_key("export_name");
    std::string save_air_path;
    auto iter_path = options.find(export_path_key);
    TNG_ASSERT(iter_path != options.end(), "Export_path is none when export graph");
    save_air_path += iter_path->second.GetString();
    save_air_path += "/";
    auto iter_name = options.find(export_name_key);
    TNG_ASSERT(iter_name != options.end(), "Export_name is none when export graph");
    save_air_path += iter_name->second.GetString();

    TNG_LOG(INFO) << "export air file path and name is : " << save_air_path;

    ge::GraphPtr graph = nullptr;
    TNG_RETURN_IF_ERROR(compat::ParseGraphFromArray(serialized_proto, proto_size, graph));
    TNG_ASSERT_NOTNULL(graph);
    TNG_ASSERT(graph->SaveToFile(save_air_path.c_str()) == ge::GRAPH_SUCCESS);

    return Status::Success();
}
}  // namespace ep
}  // namespace tng
