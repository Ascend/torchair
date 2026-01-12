# 图编译报错“it should be less than input shape size xxx”

## 问题现象描述

在多卡环境下，执行全网的图模式推理时，出现以下错误：

```bash
E89999: [PID: 8383] 2025-06-28-17:38:17.580.416 op[Transpose], attr[perm], has wrong value[2], it should be less than input shape size[2][FUNC:TransposeCommonInferShape][FILE:transformation_ops.cc][LINE:1039]
        TraceBack (most recent call last):
       Call InferShapeAndType for node:Transpose(Transpose) failed[FUNC:Infer][FILE:infershape_pass.cc][LINE:118]
       process pass InferShapePass on node:Transpose failed, ret:4294967295[FUNC:RunPassesOnNode][FILE:base_pass.cc][LINE:565]
       [Call][PreRun] Failed, graph_id:0, session_id:0.[FUNC:CompileGraph][FILE:graph_manager.cc][LINE:4652]
       [Compile][Graph]Compile graph failed, error code:1343225857, session_id:0, graph_id:0.[FUNC:CompileGraph][FILE:ge_api.cc][LINE:1343]
```

## 可能的原因

-   脚本中用户实际transpose的Tensor不符合预期，需要修改脚本
-   Meta推导和GE的InferShape推导类型不一致，导致GE算子侧transpose操作不符合预期

## 处理步骤

1.  此类报错通常出现在CANN算子的InferShape阶段，可获取GE的dump图与TorchAir的dump图进行比较，dump操作参见[关键数据获取](定界与定位流程.md#关键数据获取)。
2.  经过比较，发现allgather推导的输出shape不同，TorchAir推导的输出shape为\[4, 256, 5120\]，而GE算子推导的输出shape为\[1024, 5120\]。
3.  单击[Link](https://www.hiascend.com/support)联系技术支持，了解GE算子底层实现，发现两者的shape推导实现确实存在差异，需要在converter侧做适配。
4.  在推理脚本中实现converter时，插入Reshape操作，代码如下，仅供参考，以保证GE算子推导的shape与PyTorch推导的shape一致。

```python
@register_fx_node_ge_converter(torch.ops.npu_define.allgather_in_tensor.default)
    def converter_allgather_in_tensor(
           output_size: Union[List[int], Tensor],
           input_tensor: torch.Tensor,
           tag: str,
           rank_list,
           group_size: int,
           meta_outputs: Any = None,):   
    """allgather_in_tensor(SymInt[] output_size, Tensor input, str tag, int[] ranks, int group_size) -> Tensor"""
       group_name = get_group_name_and_record(tag, rank_list, group_size)
       res = ge.HcomAllGather(input_tensor, rank_size=group_size, group=group_name, fusion=0)
       output_size = dtype_promote(output_size, target_dtype=DataType.DT_INT64)
       return ge.Reshape(res, output_size)
```
