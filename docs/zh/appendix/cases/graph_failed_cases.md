# 入图失败定界与定位

## 定界与定位流程

当PyTorch网络在昇腾NPU上以图模式运行出现了入图失败（如断图），可按[图1](#fig1)进行问题定位。

整体定位流程大致如下，请根据实际情况逐一排查。

1.  判断PyTorch模型在单算子模式下是否运行成功。

    若失败可能存在部分算子在NPU上不支持，优先打通单算子模式，具体可参考[torch\_npu](https://www.hiascend.com/document/detail/zh/Pytorch/700/index/index.html)资料；若成功进入[步骤2](#step2)。

2.  判断图模式backend采用Eager后端是否运行成功。若失败可能FX图本身执行存在问题，若成功进入[步骤3](#step3)。<a id="step2"></a>

3.  判断图模式backend采用NPU后端是否运行成功。若失败可能是FX成图存在断图问题，需要检视PyTorch脚本分析断图原因；若成功进入[步骤4](#step4)。<a id="step3"></a>

4.  如果FX成图没问题，需要判断问题出现在TorchAir的converter转换阶段还是底层算子编译、执行阶段。<a id="step4"></a>

    请打开TorchAir的C++和Python侧debug日志，根据报错提示和具体的失败堆栈信息，自行分析和解决问题。若无法解决，获取日志后您可以单击[Link](https://www.hiascend.com/support)联系技术支持。

**图 1**  入图问题分析流程  
![](../../figures/graph_cases_flowchart.png "入图问题分析流程")<a id="fig1"></a>

## 关键数据获取

入图失败后，一般需要借助如下信息辅助问题的定界与定位，请用户根据实际情况获取。

|关键数据|说明|
|--|--|
|TorchAir的Python侧日志|参考TorchAir Python层日志打印，在PyTorch脚本中添加logger.setLevel(logging.DEBUG)，查看debug日志。|
|TorchAir的C++侧日志|参考TorchAir C++层日志打印，设置环境变量export TNG_LOG_LEVEL=0，查看C++日志。|
|TorchAir dump图|参考图结构dump功能，在PyTorch脚本中设置config.debug.graph_dump.type="pbtxt" ，查看TorchAir dump图信息。|
|GE dump图|参考《CANN 环境变量参考》中的“DUMP_GE_GRAPH”章节，设置环境变量DUMP_GE_GRAPH，查看GE的dump图信息。|
|CANN侧plog日志|参考《CANN 环境变量参考》中的“ASCEND_GLOBAL_LOG_LEVEL”章节，设置环境变量export ASCEND_GLOBAL_LOG_LEVEL=0开启plog debug日志。参考《CANN 环境变量参考》中的“ASCEND_SLOG_PRINT_TO_STDOUT”章节，设置环境变量export ASCEND_SLOG_PRINT_TO_STDOUT=1开启日志打印。|

## 整网运行报错“xxx op dtype is not same”

### 问题现象描述

在图模式场景下进行整网推理时，算子出现了如下类似的报错：

```txt
E89999:[PID:260559] 2024-11-06-15:44:43.218.474 op[FloorDiv_1] op dtype is not same, type1:DT_FLOAT, type2:DT_INT64[FUNC:CheckTwoInputDtypeSame] 
       TraceBack (most recent call last):
       Verifying FloorDiv_1 failed.[FUNC:InferShapeAndType][FILE:infershape_pass.cc][LINE:129]
       Call InferShapeAndType for node:FloorDiv_1(FloorDiv) failed[FUNC:Infer][FILE:infershape_pass.cc][LINE:117]
       process pass InferShapePass on node:FloorDiv_1 failed, ret:4294967295[FUNC:RunPassesOnNode][FILE:base_pass.cc][LINE:563]
       [Call][PreRun] Failed, graph_id:0, session_id:0.[FUNC:CompileGraph][FILE:graph_manager.cc][LINE:4467]
       [Compile][Graph]Compile graph failed, error code:1343225857, session_id:0, graph_id:0.[FUNC:CompileGraph][FILE:ge_api.cc]
```

### 可能原因

-   Meta推导的输出dtype不符合预期。
-   Meta推导符合预期，但GE算子不支持Meta推导的dtype类型。
-   GE算子的dtype推导不符合预期。

### 处理步骤

1.  此类报错通常在CANN侧，一般在图编译阶段，可获取GE的dump图与TorchAir的dump图进行比较，dump操作参见[关键数据获取](#关键数据获取)。
2.  经过比较，发现TorchAir的dump图dtype类型正确，但GE的dump图在InferShape阶段，FloorDiv输入的dtype类型已经不一致。

    ![](../../figures/Snap1.png)

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

## 自定义算子入图报错“Check output size failed”

### 问题现象描述

图模式场景下使用自定义算子推理时，CANN出现如下报错信息：

```txt
RuntimeError: E19025: [PID: 44349] 2024-12-05-16:19:12.399.912 Input tensor is invalid. Reason: The Output memory provided by the user, plus 64 bytes for data alignment, is smaller than op_size in the model, which is an illegal behavior. Output size=8192 , op_size=16416. 
       TraceBack (most recent call last):
       Check output size failed, index 0, user size 8192, op size 16416.[FUNC:ConstructZeroCopyIoActiveBaseAddrs][FILE:davinci_model.cc][LINE:5728]
       Assert ((ConstructZeroCopyIoActiveBaseAddrs(zero copy_output_indexes_,output_index_to_allocation_ids_, output_data.blobs,output_tensor,false,output_in_dex_to_active_mem_base_addrs )) == ge::SUCCESS) failed[FUNC:UpdateAllNodeArgs][FILE:davinci_model.cc][LINE:5797]
       Assert ((UpdateAllNodeArgs(input_data, output_data,input_tensor, output_tensor)) == ge::SUCCESS) failed[FUNC:CopyModelData][FILE:davinci_model.cc][LINE:487]
       Assert (((ExecuteModelAsync(model_id, stream, async_mode, input_tensor, output_tensor))) == ge::SUCCESS) failed[FUNC:ExecuteModel][FILE:model_manager.cc][LINE:1923]
       GraphManager ExecuteGraphWithStreamAsync failed,session id = 0, graph id = 16, stream = 0xaaab012b740.[FUNC:ExecuteGraphWithStreamAsync][FILE:inner_session.cc][LINE:625]
       Execute graph with stream async failed, error code:1343225857, session_id:0, graph_id:16, stream:0xaaab012b740 .[FUNC:ExecuteGraphWithStreamAsync][FILE:ge_api.cc][LINE:851]
```

### 可能原因

-   PyTorch侧的数据大小和实际申请的目标数据大小不一致。
-   PyTorch侧申请的内存大小不满足算子实际的输出大小。

### 处理步骤

1.  此类报错通常在CANN算子中，原因是数据的存储size不足，因此优先查看GE的dump图，dump操作参见[关键数据获取](#关键数据获取)。
2.  通过分析图信息，发现报错算子申请的实际size和报错日志中的输出Tensor的size不一致，实际输出的是4096的bfloat16类型数据，GE输出的是4096的float32类型数据，因此申请的size不够导致报错。
3.  由于数据类型不一致，需要查看该算子的Meta推导和GE推导过程，发现两种推导类型不一致。查看算子定义，发现Meta推导应该是float32类型，而不是bfloat16类型。
4.  确认Meta推导有问题后，需对此特殊类型做处理。

    在脚本中使用如下代码，仅供参考，请以实际需求为准。目的是保证Meta推导类型正确，内存大小申请正确。

    ```python
    @impl(m, "npu_rms_norm_backward")
    def npu_rms_norm_backward_meta():
        return (torch.empty_like(self, dtype=self.dtype), torch.empty_like(gamma, dtype=torch.float32))
    ```

## 自定义算子入图Meta注册报错“tensor's device must be 'meta', got xxx instead”

### 问题现象描述

图模式场景下执行带有自定义算子的推理脚本时，出现如下的报错日志：

```txt
torch._dynamo.exc.TorchRuntimeError: Failed running call_function custom_define.npu_custom_batch_matmul_cce(*(FakeTensor(..., device='npu:4', size=(3072, 2048), dtype=torch.int8), FakeTensor(..., device='npu:4', size=(2048, 4096), dtype=torch.int8), FakeTensor(..., device='npu:4', size=(4096,), dtype=torch.int64)), **{}): tensor's device must be `meta`, got cpu instead
```

### 可能的原因

Meta注册时构造的Tensor类型不符合要求。

### 处理步骤

1.  此类报错通常问题出现在Dynamo编译阶段，该阶段自定义算子主要的代码实现就是Meta注册。
2.  根据报错提示，先检查Meta注册代码，代码形如下方，可以发现确实返回了CPU Tensor。

    ```python
    @impl(m, "npu_custom_batch_matmul_cce", "Meta") 
    def npu_custom_batch_matmul_cce_meta(a, b, scale):   
        return torch.zeros(a.shape[0], b.shape[1])
    ```

3.  将返回的Tensor指定device为"meta"，问题即可解决。

    ```python
    @impl(m, "npu_custom_batch_matmul_cce", "Meta") 
    def npu_custom_batch_matmul_cce_meta(a, b, scale): 
        return torch.zeros(a.shape[0], b.shape[1], device="meta")
    ```

## 开启固定权重类输入地址功能后出现精度问题

### 问题现象描述

图模式场景下执行推理脚本时，若开启[固定权重类输入地址功能](../../ascend_ir/features/advanced/frozen_parameter.md)（config.experimental\_config.frozen\_parameter=True），发现图模式和Eager模式下模型精度不一致，图模式下精度存在问题，关闭该功能后正常。

### 可能的原因

-   parameter对象的输入不符合预期，原始数据有误。
-   模型中存在非连续的parameter对象，导致计算的结果错误。

### 处理步骤

1.  出现精度问题后，首先排查是否是Dynamo框架导致FX图有误。

    先验证Eager模式执行效果，若执行后的模型精度正确，那么原因就是NPU后端图模式导致。

2.  确认问题来源于图模式后，分别对比图模式下data dump数据（[算子data dump功能](../../ascend_ir/features/advanced/data_dump.md)）和Eager模式下data dump数据（[算子data dump功能（Eager模式）](../../ascend_ir/features/basic/data_dump_eager.md)）。对比后发现原始输入存在差异，说明图模式处理输入参数时已经存在问题。
3.  为进一步确认问题原因，开启TorchAir的Python侧日志，观察其处理输入的日志，发现该模型的输入input 0连续性不符合预期，是非连续的，如下所示

    ```txt
    [DEBUG] TORCHAIR(2250956,python):2025-02-06 15:44:44 runtime inputs
    [DEBUG] TORCHAIR(2250956,python):2025-02-06 15:44:44    input 0: <class 'torch.Tensor'>(torch.Size([896, 1152]), torch.int8, contiguous=False)
    [DEBUG] TORCHAIR(2250956,python):2025-02-06 15:44:44    input 1: <class 'torch.Tensor'>(torch.Size([1152]), torch.bfloat16, contiguous=True)
    [DEBUG] TORCHAIR(2250956,python):2025-02-06 15:44:44    input 2: <class 'torch.Tensor'> (torch.Size([1]), torch.float32, contiguous=True)
    [DEBUG] TORCHAIR(2250956,python):2025-02-06 15:44:44    input 3: <class 'torch.Tensor'> (torch.Size([1, 896]), torch.int8, contiguous=True)
    [INFO] TORCHAIR(2250956,python):2025-02-06 15:44:44 input process func is:
    ```

4.  修改推理脚本，将非连续输入parameter对象转为连续输入。

    非连续输入无法直接传入GE进行计算，因此先确认该输入在模型中的位置，在脚本中使用“contiguous\(\)”完成转换。

## 图编译报错“it should be less than input shape size\[xxx\]”

### 问题现象描述

在多卡环境下，执行全网的图模式推理时，出现以下错误：

```txt
E89999: [PID: 8383] 2025-06-28-17:38:17.580.416 op[Transpose], attr[perm], has wrong value[2], it should be less than input shape size[2][FUNC:TransposeCommonInferShape][FILE:transformation_ops.cc][LINE:1039]
        TraceBack (most recent call last):
       Call InferShapeAndType for node:Transpose(Transpose) failed[FUNC:Infer][FILE:infershape_pass.cc][LINE:118]
       process pass InferShapePass on node:Transpose failed, ret:4294967295[FUNC:RunPassesOnNode][FILE:base_pass.cc][LINE:565]
       [Call][PreRun] Failed, graph_id:0, session_id:0.[FUNC:CompileGraph][FILE:graph_manager.cc][LINE:4652]
       [Compile][Graph]Compile graph failed, error code:1343225857, session_id:0, graph_id:0.[FUNC:CompileGraph][FILE:ge_api.cc][LINE:1343]
```

### 可能的原因

-   脚本中用户实际transpose的Tensor不符合预期，需要修改脚本
-   Meta推导和GE的InferShape推导类型不一致，导致GE算子侧transpose操作不符合预期

### 处理步骤

1.  此类报错通常出现在CANN算子的InferShape阶段，可获取GE的dump图与TorchAir的dump图进行比较，dump操作参见[关键数据获取](#关键数据获取)。
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

## 运行时报错“torch.xxx ge\_converter is not implemented!”

### 问题描述

运行时可能遇到未支持的Converter，报错信息为："torch.xxx ge\_converter is not implemented!"。

### 解决方案

参考[Converter补齐](https://gitcode.com/ascend/torchair/blob/master/CONTRIBUTING.md#converter%E8%A1%A5%E9%BD%90)，补齐缺少的Converter即可。

## 自定义算子入图报错“unsupported operator”

### 问题现象描述

使能TorchAir图模式后，出现如下报错：

```txt
torch._dynamo.exc.Unsupported: unsupported operator: npu.custom.default (see https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs/edit#heading=h.64r4npvq0w0 for how to fix)
```

### 可能原因

算子没有实现Meta推导函数，无法入图。

### 处理步骤

参考[自定义算子入图](../../custom_op_graph/custom_op_graph.md)章节完成Meta推导函数实现。

## 自定义算子入图报错“Found a custom \(non-ATen\) operator”

### 问题现象描述

使能TorchAir图模式后，出现如下报错：

```txt
RuntimeError: Found a custom (non-ATen) operator that either mutates or its inputs: npu::npu_xp_inplace_custom.. Getting these operators to work with functionalization requires some extra work. For mutable ops you need to register a corresponding out-of-place variant of the op, and you also need to register a Functionalization kernel that performs some boilerplate, telling functionalization to map from the mutable op to the out-of-place op. See a more complete example of how to do this at https://gist.github.com/bdhirsh/7dadbf6296f8f7d1abcf4c482f438aaa. Please file a GitHub issue if you run into any problems.
```

### 可能原因

算子为In-place类算子，但是没有实现函数化转换。

### 处理步骤

参考[自定义算子入图](../../custom_op_graph/custom_op_graph.md)章节中In-place算子样例，完成函数化转换（将In-place算子替换为非In-place算子）。

## 自定义算子入图报错“torch.\_C.\_dispatch\_tls\_local\_exclude\_set\(\)”

### 问题现象描述

在自定义算子入图过程中，出现了如下类似的报错：

```txt
assert not torch._C._dispatch_tls_local_exclude_set().has(AssertionError:xx)
```

### 可能原因

该报错为PyTorch原生错误，通常发生在PyTorch算子通过torch.library.Library接口（介绍参考[PyTorch官网](https://docs.pytorch.org/docs/stable/library.html#torch.library.Library)）注册时，同时又没有实现Meta推导函数。

### 处理步骤

参考[自定义算子入图](../../custom_op_graph/custom_op_graph.md)章节完成Meta推导函数实现。










