# 自定义算子入图报错“Check output size failed”

## 问题现象描述

图模式场景下使用自定义算子推理时，CANN出现如下报错信息：

```bash
RuntimeError: E19025: [PID: 44349] 2024-12-05-16:19:12.399.912 Input tensor is invalid. Reason: The Output memory provided by the user, plus 64 bytes for data alignment, is smaller than op_size in the model, which is an illegal behavior. Output size=8192 , op_size=16416. 
       TraceBack (most recent call last):
       Check output size failed, index 0, user size 8192, op size 16416.[FUNC:ConstructZeroCopyIoActiveBaseAddrs][FILE:davinci_model.cc][LINE:5728]
       Assert ((ConstructZeroCopyIoActiveBaseAddrs(zero copy_output_indexes_,output_index_to_allocation_ids_, output_data.blobs,output_tensor,false,output_in_dex_to_active_mem_base_addrs )) == ge::SUCCESS) failed[FUNC:UpdateAllNodeArgs][FILE:davinci_model.cc][LINE:5797]
       Assert ((UpdateAllNodeArgs(input_data, output_data,input_tensor, output_tensor)) == ge::SUCCESS) failed[FUNC:CopyModelData][FILE:davinci_model.cc][LINE:487]
       Assert (((ExecuteModelAsync(model_id, stream, async_mode, input_tensor, output_tensor))) == ge::SUCCESS) failed[FUNC:ExecuteModel][FILE:model_manager.cc][LINE:1923]
       GraphManager ExecuteGrapWithStreamhAsync failed,session id = 0, graph id = 16, stream = 0xaaab012b740.[FUNC:ExecuteGraphWithStreamAsync][FILE:inner_session.cc][LINE:625]
       Execute graph with stream async failed, error code:1343225857, session_id:0, graph_id:16, stream:0xaaab012b740 .[FUNC:ExecuteGraphWithStreamAsync][FILE:ge_api.cc][LINE:851]
```

## 可能原因

-   PyTorch侧的数据大小和实际申请的目标数据大小不一致。
-   PyTorch侧申请的内存大小不满足算子实际的输出大小。

## 处理步骤

1.  此类报错通常在CANN算子中，原因是数据的存储size不足，因此优先查看GE的dump图，dump操作参见[关键数据获取](定界与定位流程.md#关键数据获取)。
2.  通过分析图信息，发现报错算子申请的实际size和报错日志中的输出Tensor的size不一致，实际输出的是4096的bfloat16类型数据，GE输出的是4096的float32类型数据，因此申请的size不够导致报错。
3.  由于数据类型不一致，需要查看该算子的Meta推导和GE推导过程，发现两种推导类型不一致。查看算子定义，发现Meta推导应该是float32类型，而不是bfloat16类型。
4.  确认Meta推导有问题后，需对此特殊类型做处理。

    在脚本中使用如下代码，仅供参考，请以实际需求为准。目的是保证Meta推导类型正确，内存大小申请正确。

    ```python
    @impl(m, "npu_rms_norm_backward")
    def npu_rms_norm_backward_meta():
        return (torch.empty_like(self, dtype=self.dtype), torch.empty_like(gamma, dtype=torch.float32))
    ```

