# 整网运行报错“xxx op dtype is not same”

## 问题现象描述

在图模式场景下进行整网推理时，算子出现了如下类似的报错：

```bash
E89999:[PID:260559] 2024-11-06-15:44:43.218.474 op[FloorDiv_1] op dtype is not same, type1:DT_FLOAT, type2:DT_INT64[FUNC:CheckTwoInputDtypeSame] 
       TraceBack (most recent call last):
       Verifying FloorDiv_1 failed.[FUNC:InferShapeAndType][FILE:infershape_pass.cc][LINE:129]
       Call InferShapeAndType for node:FloorDiv_1(FloorDiv) failed[FUNC:Infer][FILE:infershape_pass.cc][LINE:117]
       process pass InferShapePass on node:FloorDiv_1 failed, ret:4294967295[FUNC:RunPassesOnNode][FILE:base_pass.cc][LINE:563]
       [Call][PreRun] Failed, graph_id:0, session_id:0.[FUNC:CompileGraph][FILE:graph_manager.cc][LINE:4467]
       [Compile][Graph]Compile graph failed, error code:1343225857, session_id:0, graph_id:0.[FUNC:CompileGraph][FILE:ge_api.cc]
```

## 可能原因

-   Meta推导的输出dtype不符合预期。
-   Meta推导符合预期，但GE算子不支持Meta推导的dtype类型。
-   GE算子的dtype推导不符合预期。

## 处理步骤

1.  此类报错通常在CANN侧，一般在图编译阶段，可获取GE的dump图与TorchAir的dump图进行比较，dump操作参见[关键数据获取](定界与定位流程.md#关键数据获取)。
2.  经过比较，发现TorchAir的dump图dtype类型正确，但GE的dump图在InferShape阶段，FloorDiv输入的dtype类型已经不一致。

    ![](figures/Snap1.png)

3.  根据此图出问题的节点，依次往上面的节点排查，确认dtype不一致的原因。发现在GE侧，float类型是由floor算子推导而来。
4.  此时回到TorchAir的dump图，查看期望的输入/输出dtype，发现floor算子的期望输出类型是int类型。此时确认原因：用户期望的输入dtype和GE侧InferShape推导的dtype并不一致，此时需要用户在算子converter转换阶段处理这种差异，对输入插入cast保证dtype类型一致，脚本修改样例如下：

    ```python
    @register_fx_node_ge_converter(math.floor)
    # 实现converter
    def converter_math_floor(self: Union[Number, Tensor],
                           meta_outputs: TensorSpec = None):
        if not isinstance(self, Tensor):
            return math.floor(self)
        self = dtype_promote(self, target_dtype=meta_outputs.dtype)
        return ge.Floor(self)
    ```
