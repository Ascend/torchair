#include <cstdarg>

#include "concrete_graph.h"

#include "external/graph/types.h"
#include "framework/common/ge_types.h"
#include "ge/ge_api.h"
#include "ge_ir.pb.h"
#include "graph/detail/model_serialize_imp.h"
#include "graph/model_serialize.h"
#include "graph/utils/graph_utils_ex.h"

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

    auto graph_data = std::make_unique<GraphData>();

    TNG_ASSERT(graph_data->graph_def.ParseFromArray(serialized_proto, proto_size));
    TNG_LOG(INFO) << "Graph parsed successfully and " << graph_data->graph_def.op_size() << " ops parsed.";

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
    TNG_RETURN_IF_ERROR(compat::ConvertGraphDefToAir(graph_data->graph_def, graph_data->graph, save_air_path.c_str()));

    TNG_LOG(INFO) << DebugString(*graph_data);

    TNG_LOG(INFO) << "Concrete graph from proto with size " << proto_size << " created.";

    return Status::Success();
}
}  // namespace ep
}  // namespace tng
