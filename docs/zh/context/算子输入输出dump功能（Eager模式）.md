# 算子输入输出dump功能（Eager模式）

## 功能简介

Eager模式下，dump图计算过程中的输入、输出数据，用于后续问题定位和分析，如算子运行性能或精度问题。

## 使用约束

使用该功能时，NPU图编译后端中与图模式相关的配置均不生效，具体配置参见[CompilerConfig类](CompilerConfig类.md)。

## 使用方法

该功能通过[torchair.get\_npu\_backend](get_npu_backend.md)中compiler\_config配置，示例如下，仅供参考不支持直接拷贝运行，参数说明如下表。

```
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

<table><thead>
  <tr>
    <th>参数名</th>
    <th>参数说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>data_dump.type</td>
    <td>指定dump文件类型，字符串类型。<br>
     <li>默认为None，表示不导出dump数据。</li>
     <li>若设置，当前仅支持npy格式。</li>
    </td>
  </tr>
  <tr>
    <td>data_dump.filter</td>
    <td>用户自定义过滤函数，保留满足函数条件的内容。<br>
    输入：PyTorch中的<a href="https://github.com/pytorch/pytorch/blob/bf5aa9e42eb4049aad56264dacefd638233924b5/torch/fx/node.py">Node类</a>的实例n。<br>
    输出：PyTorch中的<a href="https://github.com/pytorch/pytorch/blob/bf5aa9e42eb4049aad56264dacefd638233924b5/torch/fx/node.py">Node类</a>的实例n或者None。<br>
    默认值为None，表示不过滤任何内容。<br>
    说明：Node类实例化的各属性（如name、target等）获取方法主要通过<a href="图结构dump功能.md">图结构dump功能</a>获取。<br>
    以常见的name属性为例，获取方法如下：<br>
     <ol>
     <li>先以py格式dump图信息。<br>
         <pre><code class="language-python">config.debug.graph_dump.type = "py"</pre></code>
     </li>
     <li>在当前执行路径下生成dynamo_*.py，示例如下，搜索关键词“FX Code”，其后面字段对应n.name属性信息。<br>
         <pre><code class="language-bash"># File "/home/a.py", line 32, in forward    x=x+y<br>## FX Code: add_1: torch.float32[s0, s0]npu:0 = torch.ops.aten.add.Tensor(add: torch.float32[s0, s0]npu:0, arg2_1: torch.float32[s0, s0]npu:0)<br>Add_1_0 = ge.Add(Add_0, arg2_1_0, node_name="Add_1")<br># File "/home/a.py", line 36, in forward    x=x-1<br>## FX Code: sub: torch.float32[s0, s0]npu:0 = torch.ops.aten.sub.Tensor(mul_1: torch.float32[s0, s0]npu:0, 1)<br>Sub_0 = ge.Sub(Cast_1_0, ge.Const(1, dtype=0), node_name="Sub")<br></code></pre>
      </li>
     </ol>
    </td>
  </tr>
  <tr>
    <td>data_dump.path</td>
      <td>设置dump文件生成的路径，字符串型。可选配置，如果不设置，默认为当前执行路径。<br><strong>说明</strong>：请确保该参数指定的路径确实存在，并且运行用户具有读、写操作权限。
    </td>
  </tr>
</tbody>
</table>

## 产物说明

开启数据dump功能后，得到\$\{op\_type\}-\$\{aten\_ir\}.\$\{param\_type\}\$\{param\_idx\}\$\{timestamp\}.npy文件。其中\$\{op\_type\}为算子类型，\$\{aten\_ir\}为ATen算子名，\$\{param\_type\}为参数输入/输出类型，\$\{param\_idx\}为输入/输出参数的索引（默认从0开始），\$\{timestamp\}为时间戳，\$\{world\_size\}和\$\{global\_rank\}指集合通信中的world\_size以及global\_rank信息。

产物路径默认为当前执行目录，目录结构如下：

```
├ torch_test.py                          // PyTorch脚本
│   ├── worldsize${world_size}_global_rank${global_rank}  // 若只涉及单卡则表示为“worldsize1_global_rank0”
│       ├── graph_1                     // ${global_rank}所在进程的子图名称
│           ├── 0                       // 首次dump的结果文件，通过step id标记，step id从0开始 
│               ├── add_1-aten.add.Tensor.INPUT.0.20251017071211190258.npy     // filter过滤函数dump结果
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

