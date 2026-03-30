# SuperKernel调试调优方法

## 执行异常/超时

### 功能说明

当SuperKernel执行异常时，用户可结合ERROR级别的日志报错信息，定位到发生异常的SuperKernel及其内部具体子算子。

### 定位方法
当SuperKernel执行报错时，可按如下方法进行定位：
1. 识别报错算子是否为SuperKernel。在plog中搜索关键字```fault kernel_name```，若检索出的kernel_name以```te_superkernel```开头, 通过该关键字可以搜索出异常SuperKernel的```kernel_name```、```stream_id```、```task_id```等信息。

2. 确定报错的SuperKernel算子。在plog中搜索关键字```origin_op_name```，可以检索出异常SuperKernel的融合信息，包括用户框定的```scope name```、SuperKernel内的起始算子和结束算子信息。
   
3. 确定SuperKernel内具体报错的子算子。在plog中搜索关键字```symbol name```，可检索出报错子算子的符号名，例如```te_rmsnorm_xxx```，从而定位到SuperKernel内实际报错的子算子为```te_rmsnorm_xxx```。


> 日志级别和日志路径的相关配置，请参考[[日志参考](https://hiascend.com/document/redirect/CannCommunitylogref)]


### 定位案例
当使能SuperKernel执行失败时，可先搜索```fault kernel_name```，若可以搜索到相关日志信息，则表示当前报错算子发生在SuperKernel里。搜索结果示例如下：
```
[ERROR] RUNTIME(1567697,python3):2026-03-28-11:32:18.888.239 [davinci_kernel_task.cc:1639]1568661 PrintErrorInfoForDavinciTask:[DFX_INFO]Aicore kernel execute failed, device_id=0, stream_id=70, report_stream_id=73, task_id=0, flip_num=0, fault kernel_name=te_superkernel_911c4c5214b52a01ea224923a6130b1ff609f8f8759f8dc118fe65b846e4573e, fault kernel info ext=none, program id=1, hash=9430089888170931922.
[ERROR] RUNTIME(1567697,python3):2026-03-28-11:32:18.953.295 [stream.cc:1507]1568661 GetError:[DFX_INFO]Aicore kernel execute failed, device_id=0, stream_id=70, report_stream_id=73, task_id=0, flip_num=0, fault kernel_name=te_superkernel_911c4c5214b52a01ea224923a6130b1ff609f8f8759f8dc118fe65b846e4573e, fault kernel info ext=none, program id=1, hash=9430089888170931922.
```
继续搜索```origin_op_name```，定位具体报错的SuperKernel。搜索结果示例如下：
```
[ERROR] GE(1567697,python3):2026-03-28-11:32:18.888.346 [error_tracking.cc:177]1568661 ErrorTrackingCallback: ErrorNo: 4294967295(failed)Error happened, origin_op_name [sk_sp1_start_TransQuantParamV2_end_QuantBatchMatmulV3_2], op_name [sk_sp1_start_TransQuantParamV2_end_QuantBatchMatmulV3_2], task_id 0, stream_id 70.
```
上述日志中，可以确定报错位置在用户传入scopename为```sp1```的SuperKernel内，且该SuperKernel起始算子为TransQuantParamV2，结束算子为QuantBatchMatmulV3_2。
继续搜索```symbol name```，可进一步定位到SuperKernel内部子算子信息。搜索结果示例如下：
```
[ERROR] IDEDD(1567697,python3):2026-03-28-11:32:18.893.754 [kernel_info_collector.cpp:534][tid:1568661] [Dump][Exception] coreId=3, coreType=0, startPC=12c044e03000, currentPC=12c044e098b8, offset=68b8, symbol name=te_quantbatchmatmulv3_8ab205c8644b12fc36c5d343da847e383ccea495f1f6c6a4f562d0abcf218010__kernel0_middle_split3, symbol size=1914, symbol offset=6800
```
其中，```symbol name=te_quantbatchmatmulv3_8ab205c8644b12fc36c5d343da847e383ccea495f1f6c6a4f562d0abcf218010__kernel0_middle_split3```即为SuperKernel内部异常的子算子。