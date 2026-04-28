# 多流并发死锁检测功能

## 功能简介

本功能可以自动化检测[多流并发](../advanced/multi_stream.md)场景下可能存在的死锁风险。  
当多个流上的通信类算子同时进入就绪状态，且它们所请求的 AIV 核心数量总和超过了硬件的总可用数，NPU 调度器将无法同时满足所有任务的资源分配需求，从而形成死锁——即每个任务都在等待其他任务释放核心资源，但没有任何任务能够获得足够的资源以开始运行。

## 使用约束

本功能支持的产品型号参见[使用说明](../../overview.md#使用说明)。

## 使用方法

通过npugraph\_ex的options配置，示例如下，仅供参考不支持直接拷贝运行，参数说明参见下表。

```python
import torch
model = torch.compile(npu_mode, backend="npugraph_ex", options={"deadlock_check": True}, dynamic=False)
```

**表 1**  参数说明

|参数名|参数说明|
|--|--|
|deadlock_check|是否开启死锁检测功能，bool类型。False（默认值）：不开启死锁检测。True：开启死锁检测。|

## 产物说明

若存在死锁风险，则在脚本执行同级目录/torch_compile_debug/run_时间戳-pid_进程号/npugraph_ex/model__xx/，输出如下结果：

```txt
graph_1_id_281464657589552_rank_2_pid_9096_ts_20260408072201427037.json                           // 原始dump json
graph_1_id_281464657589552_rank_2_pid_9096_ts_20260408072201427037_filtered.json                  // 过滤掉计算算子后的json
graph_1_id_281464657589552_rank_2_pid_9096_ts_20260408072201427037_filtered_deadlock_check.json   // 死锁检测结果json
```

此时会有打屏日志提示：

```txt
[RESULT] Deadlock risk detected! 56 conflicting pair(s):

  CONFLICT: [stream316 TaskId=4] aiv_all_gather_bfloat16_t (32 AIVEC)  <->  [stream317 TaskId=5] MoeDistributeDispatchV2_e3e2549baf4795bf56c350e970d2218a_1060 (24 AIVEC)  sum=56 > 48
  CONFLICT: [stream316 TaskId=4] aiv_all_gather_bfloat16_t (32 AIVEC)  <->  [stream317 TaskId=9] MoeDistributeCombineV2_37d3570b1cb4463a4348af4b433cf23c_32 (24 AIVEC)  sum=56 > 48
  CONFLICT: [stream316 TaskId=20] aiv_all_gather_bfloat16_t (32 AIVEC)  <->  [stream317 TaskId=17] MoeDistributeDispatchV2_e3e2549baf4795bf56c350e970d2218a_1060 (24 AIVEC)  sum=56 > 48
```

例如第一行日志提示，在流316上taskid为4的占用32个vector核的aiv_all_gather_bfloat16_t算子和在流317上taskid为5的占用24个vector核的MoeDistributeDispatchV2_e3e2549baf4795bf56c350e970d2218a_1060算子有死锁风险，因为他们的占用总核数大于了设备的总核数48。

死锁的task统计在graph_1_id_281464657589552_rank_2_pid_9096_ts_20260408072201427037_filtered_deadlock_check.json中。
