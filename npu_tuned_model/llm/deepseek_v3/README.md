# DeepseekV3

本sample主要是DeepseekV3模型在npu上的推理适配点介绍，使用transformers==4.40.0版本，基于DeepseekV3开源算法[modeling_deepseek.py](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/blob/main/modeling_deepseek.py)进行迁移。

---

# 1. Quick Start：执行样例

本sample的目录下提供了手动Tensor并行及DeepseekV3推理的执行样例参考

## 1.1. 环境准备
**安装对应的transformers版本与numpy版本**

```shell
pip3 install transformers==4.40.0
pip3 install numpy==1.24.4
```

**安装torch_npu**

相关介绍与安装步骤参考：[torch_npu](https://gitee.com/ascend/pytorch)

> torch_npu是支持PyTorch框架运行在昇腾NPU上的适配插件，提供AI模型执行在昇腾NPU上的基础能力。
> 同时我们使用其torchair子模块提供的torch.compile后端，加速模型执行。相关介绍参考：[torchair](https://gitee.com/ascend/torchair)
> 版本配套关系参考：[附录2：版本配套表](#5-附录2版本配套表)

**安装MindSpeed**

相关介绍与安装步骤参考：[MindSpeed](https://gitee.com/ascend/MindSpeed)

> MindSpeed 是针对华为昇腾设备的大模型加速库。我们主要使用其提供的GMM融合算子，加速DeepseekV3模型MOE专家计算部分的执行

```shell
git clone https://gitee.com/ascend/MindSpeed.git
pip3 install -e MindSpeed
```

**设置环境变量**

```shell
cann_path=/usr/local/Ascend # 昇腾cann包安装目录
source ${cann_path}/latest/bin/setenv.bash
export ASCEND_HOME_PATH=${cann_path}/latest
export WORLD_SIZE=8 # 多卡推理时使用的卡数，默认为8
```

## 1.2. 权重准备

下载[DeepseekV3权重](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/tree/main)到本地路径`model_path_origin`

**手动切分与调整权重**

本sample对DeepseekV3开源模型进行了部分结构调整，需要对原始权重进行切分与调整。通过执行`scripts/split_weight.py`进行切分与调整，涉及参数如下：
- 已提供权重切分脚本`scripts/split_weight.py`
- `model_path_origin`为原始完整权重路径
- `model_path_tp`为切分与调整后的权重路径
- `world_size`为切分权重的份数，即多卡推理时使用的卡数。默认为8，即默认执行8卡切分与调整

```shell
python3 scripts/split_weight.py --model-path model_path_origin --output-path model_path_tp --world-size world_size
```

## 1.3. DeepseekV3多卡推理

基于已切分好的权重路径`model_path_tp`，可通过已提供的推理启动脚本`scripts/infer.sh`进行多卡推理:
```shell
cd scripts
bash infer.sh model_path_tp
```
**需要注意**
默认的推理脚本通过调用推理执行脚本`scripts/infer.py`实现：
- 提供8卡推理
- 默认输入padding到1024，最大输出token数为32，decode为单batch推理。
- 默认使用`eager`单算子模式推理
- 相关参数说明参考[附录1：环境变量说明](#4-附录1环境变量说明)

---

# 2. 目录结构

本sample目录结构与文件介绍如下：
- `engine`目录：涉及通用模型执行引擎`model_run.py`
  - `model_run.py`：模型执行引擎，包含模型初始化、模型加载、tokenizer初始化、模型推理等通用基类方法。
- `scripts`目录：涉及当前DeepseekV3模型执行涉及的相关脚本
  - `models`目录：涉及模型脚本
    - `configuration_deepseek.py`：DeepseekV3模型配置config
    - `modeling_deepseek.py`：DeepseekV3模型脚本
  - `runner_deepseek.py`：基于通用模型执行引擎进行继承，适配当前模型所需各项内容
  - `split_weight.py`：权重切分工具
  - `infer.py`：DeepSeekV3模型推理执行脚本
  - `infer.sh`：DeepSeekV3模型推理启动脚本

---

# 3. 模型迁移、适配与优化

[模型迁移指导](https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/ptmoddevg/trainingmigrguide/PT_LMTMOG_0002.html)

## 3.1. 权重切分与手动Tensor并行

推理时我们需要对模型权重进行切分，使得内存占用小于device可用内存。本sample以Tensor并行为例，对DeepseekV3的权重进行了手动切分。

手动Tensor并行主要涉及以下几步：

- 定义切分后的模型权重，涉及Attenion层、MOE层以及前后处理阶段。分别针对DeepseekV3Attention、DeepseekV3MLP、DeepseekV3DecoderLayer三个类
- 对模型权重在不同轴上进行切分，各卡用于独立计算。
  - 针对Attenion层的切分，具体实现可参考`scripts/split_weight.py`中的`split_w_attn`函数。
  - 针对MOE层的切分，具体实现可参考`scripts/split_weight.py`中的`split_w_moe`与`split_w_dense`函数。
  - 针对前后处理的切分，具体实现可参考`scripts/split_weight.py`中的`split_w`函数。
- 在各个切分模块，完成各卡独立计算后插入allreduce算子，用于整合多卡上并行获得的计算结果

## 3.2. 性能优化

**注**：在modeling_deepseek.py中，被修改的原函数都加了‘__’前缀，可用于对比修改后的函数变化。deepseek结构中的非MOE部分与Llama类似，通用优化点可参考[Llama](https://gitee.com/ascend/torchair/tree/master/npu_tuned_model/llm/llama)的改动，如固定kv cache大小、cos/sin优化、Add+RMSNorm融合、全量优化LM Head计算量。本sample重点展示其余改动点。

### 3.2.1. 算法优化

#### 3.2.1.1. DeepseekV2低秩压缩优化
参考[DeepseekV2论文](https://arxiv.org/pdf/2405.04434)中提及的低秩压缩方法，本sample对`DeepseekV3Attention`类进行修改
- 将原始实现中的`kv_b_proj`拆分成`kv_b_proj_w_k`与`kv_b_proj_w_v`，权重切分方式参考`scripts/split_weight.py`
- 相关计算过程在`DeepseekV3Attention`类`forward`方法中体现

### 3.2.2. 算子融合
#### 3.2.2.1. GMM使能&&Routing优化

DeepseekV3原始的MOE实现比较朴素，for循环处理每个专家，单独计算expert_num个FFN，计算效率较低。MOE路由分发体现在`__DeepseekV3MoE`类中，专家计算过程体现在`DeepSeekV3MLP`类。

CANN提供了[GroupedMatmul](https://gitee.com/ascend/MindSpeed/blob/master/docs/ops/gmm.md)算子，可以同时计算多个专家，提高计算和搬运效率。为了使能GroupedMatmul算子，我们需修改Routing逻辑，构造对应输入。

- 整体上进行`DeepseekV3MoE`重构
  - 路由专家计算过程，主要涉及`DeepSeekV3MLP`类，适配GroupedMatmul算子并修改为`DeepSeekV3MLPGMM`类
    - 路由专家的权重在`DeepSeekV3MLPGMM`中进行了专家合并，合并为了一个weight，体现在`scripts/split_weight.py`中。进行tensor并行切分时，同时对`DeepSeekV3MLPGMM`进行切分
  - 共享专家依旧沿用`DeepSeekV3MLP`类

- 专家路由基础流程参考[GMM使能&&Routing优化](https://gitee.com/ascend/torchair/blob/master/npu_tuned_model/llm/mixtral/README.md)实现，体现在`DeepseekV3MoE`类中的`moe_infer_normal`函数
- 同时，可通过使能CANN提供的torch_npu moe_routing相关算子进行优化，体现在`DeepseekV3MoE`类中的`moe_infer_fusion`函数。
  - 可通过设置`self.npu_routing_kernel=True`类使能，默认为True
  - 可通过使能[torch_npu.npu_moe_init_routing](https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/apiref/apilist/ptaoplist_000780.html)替换基础流程中专家排布获取环节
  - 可通过使能[torch_npu.npu_moe_compute_expert_tokens](https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/apiref/apilist/ptaoplist_000782.html)替换基础流程中专家获得token数计算环节
  - 可通过使能[torch_npu.npu_moe_finalize_routing](https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/apiref/apilist/ptaoplist_000781.html)替换基础流程中专家计算完成后的重新排布环节，用于获得最终输出

#### 3.2.2.2. MoeGate亲和优化

DeepseekV3原始的`MoeGate`实现中，通过`torch.zeros_like`与`scatter`算子来进行`group_mask`获取：
```python
group_mask = torch.zeros_like(group_scores)  # [n, n_group]
group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
```

在本sample中，通过`one_hot`与`sum`进行等价替换，实现npu亲和的性能优化
```python
def one_hot(tensor, num_classes):
    index = torch.arange(0, num_classes, dtype=tensor.dtype, device=tensor.device)
    return (
        tensor.view([*tensor.shape, 1]) == index.view([1] * tensor.ndim + [num_classes])
    ).to(torch.float32)

group_mask = one_hot(group_idx, self.n_group)  # [n, n_group]
group_mask = torch.sum(group_mask, dim=1)  # [n, n_group]
```

#### 3.2.2.3. MLP合并优化

原始`DeepseekV3MLP`实现中，存在`gate_proj`、`up_proj`与`down_proj`三个matmul运算，可通过将`gate_proj`与`up_proj`进行合并整合计算，提升整体计算效率。
- 整体上进行`DeepseekV3MLP`重构，原始实现体现在`__DeepseekV3MLP`类中
- 权重切分过程中，需要额外对`gate_proj`与`up_proj`的权重进行合并，体现在`scripts/split_weight.py`中

### 3.2.3. 图模式适配

在图模式适配过程中，需要**注意**：

 - 需先保证模型在npu上的eager模式功能正常和精度正确，然后再进行图模式的迁移和适配。

考虑到LLM prefill阶段，query的seq length经常是变化的；decode阶段，seq length通常是固定的。本sample通过提前引入输入padding，将输入padding到预设长度，同时以静态图的方式执行Prefill & Decode。

CompilerConfig配置参考[torchair资料](https://www.hiascend.com/document/detail/zh/Pytorch/60RC2/modthirdparty/torchairuseguide/torchair_0021.html)

- torchair提供了NPU的图构造/图编译/图执行能力。相关能力全部集成到NPU图的后端，在使用torch.compile接口时，指定NPU图后端来使能。同时提供了开关和config控制图的编译和执行流程。
- 在使用NPU图后端的时候，torchair提供了静态图和动态图两种图执行的能力。根据dynamic参数决定是否走动态图。

具体实现中
- 通过使能`EXE_MODE="dynamo"`的环境变量，可以触发`engine/model_runner.py`中的`graph_compile`函数进行图模式编译。
- 在此基础上，使能aclgraph需设置环境变量：export ENABLE_ACLGRAPH=1，会切换到mode="reduce-overhead"下完成静态图下沉。

### 3.2.4. HCCL使能AIV

利用Device的AI Vector Core计算单元来加速AllReduce，可参考[HCCL_OP_EXPANSION_MODE环境变量](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha003/apiref/envref/envref_07_0088.html)

```shell
export HCCL_OP_EXPANSION_MODE=AIV
```
---

# 4. 附录1：环境变量说明

<table>
  <tr><th>类别归属</th><th>环境变量</th><th>说明</th></tr>

  <tr><td rowspan="6">环境配置项</td></tr>
  <tr><td><code>WORLD_SIZE</code></td><td>多卡执行时，用于声明可使用的卡数</td></tr>
  <tr><td><code>LOCAL_RANK</code></td><td>每个进程在整体通信域中感知到的rank_id</td></tr>
  <tr><td><code>RANK_ID</code></td><td>每个进程在整体通信域中感知到的rank_id</td></tr>
  <tr><td><code>MASTER_ADDR</code></td><td>HCCL通信域中master主机IP</td></tr>
  <tr><td><code>MASTER_PORT</code></td><td>HCCL通信域中master主机的通信端口</td></tr>

  <tr><td rowspan="7">模型基础配置项</td></tr>
  <tr><td><code>MODEL_NAME</code></td><td>模型名</td></tr>
  <tr><td><code>MODEL_DIR</code></td><td>权重路径，必须配置到模型权重所在文件夹</td></tr>
  <tr><td><code>INPUT_MAX_LEN</code></td><td>本sample默认将输入padding到固定长度进行执行</td></tr>
  <tr><td><code>MAX_NEW_TOKENS</code></td><td>用于配置最多decode生成字符个数</td></tr>
  <tr><td><code>BATCH_SIZE</code></td><td>默认执行prefill-1batch, decode-nBatch模式。可通过设置该环境变量，使能decode多batch推理，默认为1</td></tr>
  <tr><td><code>TOKENIZER_MODE</code></td><td>可使用不同的tokenizer，用于生成不同的prompt进行推理。支持default与chat两种，默认为default</td></tr>

  <tr><td rowspan="3">执行模式配置</td></tr>
  <tr><td><code>EXE_MODE</code></td><td>用于区分图模式与单算子模式。eager表示单算子模式，dynamo表示图模式。默认为单算子模式</td></tr>
  <tr><td><code>ENABLE_ACLGRAPH</code></td><td>用于aclgraph与GE两种图模式的使能。1表示aclgraph，0表示GE。默认为GE。</td></tr>

  <tr><td rowspan="3">调测配置项</td></tr>
  <tr><td><code>HCCL_DETERMINISTIC</code></td><td>可设置该环境变量为true，用于使能多卡间的确定性计算。默认为false</td></tr>
  <tr><td><code>HCCL_OP_EXPANSION_MODE</code></td><td>利用Device的AI Vector Core计算单元来加速AllReduce。与确定性计算HCCL_DETERMINISTIC互斥</td></tr>
</table>

# 5. 附录2：版本配套表

<a id="version_match"></a>
| TorchAir版本 | PyTorch版本 | torch_npu版本 | CANN版本 | Python版本
| ----------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| master（主线） | 2.1.0 | 在研版本 | 在研版本 | Python3.8.x<br/>Python3.9.x<br/>Python3.10.x | 
| 6.0.rc3 | 2.1.0 | 6.0.rc3 | 8.0.rc3 | Python3.8.x<br/>Python3.9.x<br/>Python3.10.x | 
| 6.0.rc2 | 2.1.0 | 6.0.rc2 | 8.0.rc2 | Python3.8.x<br/>Python3.9.x<br/>Python3.10.x | 
| 6.0.rc1 | 2.1.0 | 6.0.rc1 | 8.0.rc1 | Python3.8.x<br/>Python3.9.x<br/>Python3.10.x | 
