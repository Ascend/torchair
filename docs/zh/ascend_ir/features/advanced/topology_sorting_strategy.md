# 图模式编译节点遍历选项

## 功能简介

推理场景下进行算子编译时，可以设置不同的图遍历模式，对静态图内存使用有不同的影响。用户需根据实际情况自行设置。

## 使用约束

本功能仅适用于GE图模式场景。

## 使用方法

该功能通过[torchair.get\_npu\_backend](../../api/torchair/get_npu_backend.md)中compiler\_config配置，示例如下，仅供参考不支持直接拷贝运行，参数说明参见下表。

```python
import torch_npu
import torchair 
config = torchair.CompilerConfig()
# 图模式编译的遍历策略配置
config.experimental_config.topology_sorting_strategy = "DFS"
npu_backend = torchair.get_npu_backend(compiler_config=config)
opt_model = torch.compile(model, backend=npu_backend)
```

**表 1**  参数说明

|参数名|说明|
|--|--|
|topology_sorting_strategy|图执行时可以设置不同的图遍历顺序。<br>DFS（默认值）：Depth First Search，深度优先遍历策略。<br>BFS：Breadth First Search，广度优先遍历策略。<br>RDFS：Reverse DFS，反向深度优先遍历策略。<br>StableRDFS：稳定拓扑序策略，针对图里已有的算子，不会改变其计算顺序；针对图里新增的算子，使用RDFS遍历策略。<br>若同一通信域内的通信算子间未通过控制边显式约束依赖，选择非稳定排序算法（如 DFS、BFS、RDFS）可能导致多卡间执行顺序不一致，从而引发通信死锁或精度问题。虽然StableRDFS旨在最大程度上保留原图的时序，但在图融合或图优化导致拓扑结构变更时，其稳定性仍可能失效。因此，推荐通过显式控制边来保障执行顺序的绝对一致性，而 StableRDFS仅作为在无法修改图结构时的辅助补偿方案。|
