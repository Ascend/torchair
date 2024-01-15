#include "load_sub_store_tiling.h"
#ifndef __CCE_KT_TEST__
#include "register/op_def_registry.h"
#endif

extern "C" void GetTiling(optiling::TilingData& tiling_data) {
    tiling_data.set_block_dim(48);
    tiling_data.set_z0b_size(2);
    tiling_data.set_z1t_size(4);
}
#ifndef __CCE_KT_TEST__
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  TilingData tiling;
  const gert::Shape& size_var_shape = context->GetInputShape(0)->GetOriginShape();
  tiling.set_block_dim(48);
  tiling.set_s0(size_var_shape.GetDim(0));
  tiling.set_s1(size_var_shape.GetDim(1));
  tiling.set_s2(size_var_shape.GetDim(2));

  GetTiling(tiling);
  context->SetBlockDim(tiling.get_block_dim());
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    return GRAPH_SUCCESS;
}
}

namespace ops {
class LoadAbsStore : public OpDef {
public:
    explicit LoadAbsStore(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(LoadAbsStore);
}
#endif
