# 算子data dump功能（Eager模式）

## 功能简介

Eager模式下，dump图计算过程中的输入、输出数据，用于后续问题定位和分析，如算子运行性能或精度问题。

## 使用约束

开启该功能后，GE图模式相关的功能配置均不生效。

## 使用方法

该功能通过[torchair.get\_npu\_backend](../../api/torchair/get_npu_backend.md)中compiler\_config配置，示例如下，仅供参考不支持直接拷贝运行，参数说明参见下表。

```python
import torch, torch_npu, torchair
config = torchair.CompilerConfig()
# Eager模式下数据dump功能
config.debug.data_dump.type = "npy"
config.debug.data_dump.path = "./test"
# 若只dump'add_1'和'sub'算子，n.name为Node类实例化的name属性
config.debug.data_dump.filter = lambda n: n if n.name in ['add_1', 'sub'] else None
npu_backend = torchair.get_npu_backend(compiler_config=config)
opt_model = torch.compile(model, backend=npu_backend)
```

**表 1**  参数说明

|参数名|参数说明|
|--|--|
|data_dump.type|指定dump文件类型，字符串类型。默认为None，表示不导出dump数据。若设置，当前仅支持npy格式。|
|data_dump.filter|用户自定义过滤函数，保留满足函数条件的内容。<br>输入：PyTorch中的Node类的实例n。<br>输出：PyTorch中的Node类的实例n或者None。<br>默认值为None，表示不过滤任何内容。<br>说明：Node类实例化的各属性（如name、target等）获取方法主要通过图结构dump功能获取。<br>以常见的name属性为例，获取方法如下：<br>1. 先以py格式dump图信息。config.debug.graph_dump.type = "py"<br>2. 在当前执行路径下生成dynamo_*.py，示例如下，搜索关键词“FX Code”，其后面字段对应n.name属性信息。<br># File "/home/a.py", line 32, in forward    x=x+y<br>## FX Code: **add_1**: torch.float32[s0, s0]npu:0 = torch.ops.aten.add.Tensor(add: torch.float32[s0, s0]npu:0, arg2_1: torch.float32[s0, s0]npu:0)<br>Add_1_0 = ge.Add(Add_0, arg2_1_0, node_name="Add_1")<br># File "/home/a.py", line 36, in forward    x=x-1<br>## FX Code: **sub**: torch.float32[s0, s0]npu:0 = torch.ops.aten.sub.Tensor(mul_1: torch.float32[s0, s0]npu:0, 1)<br>Sub_0 = ge.Sub(Cast_1_0, ge.Const(1, dtype=0), node_name="Sub")|
|data_dump.path|设置dump文件生成的路径，字符串型。可选配置，如果不设置，默认为当前执行路径。<br>请确保参数中指定的路径真实存在，并且运行用户具有读取和写入权限。|


## 产物说明

开启数据dump功能后，得到$\{op\_type\}-$\{aten\_ir\}.$\{param\_type\}$\{param\_idx\}$\{timestamp\}.npy文件。其中$\{op\_type\}为算子类型，$\{aten\_ir\}为ATen算子名，$\{param\_type\}为参数输入/输出类型，$\{param\_idx\}为输入/输出参数的索引（默认从0开始），$\{timestamp\}为时间戳，$\{world\_size\}和$\{global\_rank\}指集合通信中的world\_size以及global\_rank信息。

产物路径默认为当前执行目录，目录结构如下：

```txt
├ torch_test.py                            // PyTorch脚本
│   ├── worldsize${world_size}_global_rank${global_rank}  // 若只涉及单卡则表示为“worldsize1_global_rank0”
│       ├── graph_1                     // ${global_rank}所在进程的子图名称
│           ├── 0                       // 首次dump的结果文件，通过step id标记，step id从0开始 
│               ├── add_1-aten.add.Tensor.INPUT.0.20251017071211190258.npy          // filter过滤函数dump的结果文件
│               ├── add_1-aten.add.Tensor.INPUT.1.20251017071211190258.npy 
│               ├── add_1-aten.add.Tensor.OUTPUT.0.20251017071211190258.npy
│               ├── sub-aten.sub.Tensor.INPUT.0.20251017071211202083.npy
│               ├── sub-aten.sub.Tensor.INPUT.1.20251017071211202083.npy
│               ├── sub-aten.sub.Tensor.OUTPUT.0.20251017071211202083.npy
│        ......
│           ├── n                       // 第n+1次dump的结果文件，通过step id标记
│               ├── xxx.npy   
│        ......
│   ├── worldsize${world_size}_global_rank${global_rank}
│       ├── graph_1
│           ├── 0
│               ├── xxx.npy
```

