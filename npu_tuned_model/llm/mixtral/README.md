# Mixtral

本模块主要是Mixtral模型在npu上的推理适配点介绍，使用transformers==4.40.0版本，基于其中的[modeling_mixtral.py](https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/mixtral/modeling_mixtral.py)进行迁移。

# 手动Tensor并行

Mixtral 8*7B参数量为47B，Mixtral 8\*22B参数量为142B，FP16/BF16权重内存占用分别为94G和284G。推理时我们需要对模型权重进行切分，使得内存占用小于device可用内存。当前sample以Tensor并行为例，对Mixtral的权重进行了手动切分。

手动Tensor并行主要涉及三步：

- 定义切分后的模型权重，涉及MixtralAttention和MixtralBlockSparseTop2MLP两个类

```python
  # scripte/models/nodeling_mixtral.py
  # class MixtralAttention
  # 原始代码
  self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
  self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
  self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
  self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
  # 新代码
  self.hidden_size_per_rank = self.hidden_size // self.world_size
  self.num_heads_per_rank = config.num_attention_heads // self.world_size
  self.num_key_value_heads_per_rank = config.num_key_value_heads // self.world_size
  self.q_proj = nn.Linear(self.hidden_size, self.num_heads_per_rank * self.head_dim, bias=False)
  self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads_per_rank * self.head_dim, bias=False)
  self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads_per_rank * self.head_dim, bias=False)
  self.o_proj = nn.Linear(self.num_heads_per_rank* self.head_dim, self.hidden_size, bias=False)
  
  # class MixtralBlockSparseTop2MLP
  # 原始代码
  self.ffn_dim = config.intermediate_size
  # 新代码
  self.world_size = int(os.getenv("WORLD_SIZE", "1"))
  self.ffn_dim = config.intermediate_size // self.world_size
```

- 切分模型权重，可参考scripts/split_weight.py中的split_w函数，将Attention层的q/k/v Weight在N轴上切分成TP份，将MOE层每个专家中的w1/w3 weight在N轴上切分TP份，w2在K轴上切分成TP份

```python
# scripts/split_weight.py
def split_w(src_model, dst_model, world_size, local_rank):
    dst_model.lm_head.weight.data = src_model.lm_head.weight.data
    dst_model.model.embed_tokens.weight.data = src_model.model.embed_tokens.weight.data
    dst_model.model.norm.weight.data = src_model.model.norm.weight.data
    q_dim = dst_model.model.layers[0].self_attn.num_heads_per_rank * \
                dst_model.model.layers[0].self_attn.head_dim
    k_dim = dst_model.model.layers[0].self_attn.num_key_value_heads_per_rank * \
                dst_model.model.layers[0].self_attn.head_dim
    
    for i, block in enumerate(src_model.model.layers):
        # Attention层q/k/v Weight在N轴上进行切分
        dst_model.model.layers[i].self_attn.q_proj.weight.data = \
            block.self_attn.q_proj.weight.data[local_rank * q_dim: (local_rank + 1) * q_dim, :].contiguous()
        dst_model.model.layers[i].self_attn.k_proj.weight.data = \
            block.self_attn.k_proj.weight.data[local_rank * k_dim: (local_rank + 1) * k_dim, :].contiguous()
        dst_model.model.layers[i].self_attn.v_proj.weight.data = \
            block.self_attn.v_proj.weight.data[local_rank * k_dim: (local_rank + 1) * k_dim, :].contiguous()
        # Attention层proj Weight在K轴上进行切分
        dst_model.model.layers[i].self_attn.o_proj.weight.data = \
            block.self_attn.o_proj.weight.data[:, local_rank * q_dim: (local_rank + 1) * q_dim].contiguous()
        
        dst_model.model.layers[i].block_sparse_moe.gate.weight.data = \
            block.block_sparse_moe.gate.weight.data
        dst_model.model.layers[i].input_layernorm.weight.data = \
            block.input_layernorm.weight.data
        dst_model.model.layers[i].post_attention_layernorm.weight.data = \
            block.post_attention_layernorm.weight.data
        
        for j, expert in enumerate(block.block_sparse_moe.experts):
            ffn_dim = dst_model.model.layers[i].block_sparse_moe.experts[j].ffn_dim
            # MOE层每个专家中的w1/w3 weight在N轴上进行切分
            dst_model.model.layers[i].block_sparse_moe.experts[j].w1.weight.data = \
                expert.w1.weight.data[local_rank * ffn_dim: (local_rank + 1) * ffn_dim, :].contiguous()
            dst_model.model.layers[i].block_sparse_moe.experts[j].w3.weight.data = \
                expert.w3.weight.data[local_rank * ffn_dim: (local_rank + 1) * ffn_dim, :].contiguous()
            # MOE层每个专家中的w2 weight在K轴上进行切分
            dst_model.model.layers[i].block_sparse_moe.experts[j].w2.weight.data = \
                expert.w2.weight.data[:, local_rank * ffn_dim: (local_rank + 1) * ffn_dim].contiguous()
```

- Attention层和MOE层结尾处插入allreduce算子

```python
import torch.distributed as dist
if self.world_size > 1:
    dist.all_reduce(attn_output)
```

# 图模式适配

[模型迁移指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha003/devguide/moddevg/torchair/torchair_01_0001.html)

**注意**：

 - 需先保证模型在npu上的eager模式功能正常和精度正确，然后再进行图模式的迁移和适配。
 - 原始Hugging Face transformers源码中，MixtralSparseMoeBlock中存在one_hot、where等算子，受torch原生dynamo约束，无法成图。我们需先对MixtralSparseMoeBlock进行适当改写，后续性能优化项里详细展示了改动点。

考虑到LLM prefill阶段，query的seq length经常是变化的；decode阶段，seq length通常是固定的。本sample以动态图的方式执行Prefill，以静态图的方式执行Decode。

CompilerConfig配置参考[torchair资料](https://www.hiascend.com/document/detail/zh/Pytorch/60RC2/modthirdparty/torchairuseguide/torchair_0021.html)

```python
# script/benchmark_mixtral.py
def graph_compile(self):
    import torchair as tng
    import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True  # tiling全下沉性能优化
    npu_backend = tng.get_npu_backend(compiler_config=config)
    self.model.model = torch.compile(self.model.model, dynamic=True, fullgraph=True, backend=npu_backend)
    
def mark_inputs(self, model_inputs):
    if self.execute_mode == "dynamo":
        input_ids = model_inputs.get("input_ids")
        kv_len = model_inputs.get("kv_len")
        attention_mask = model_inputs.get("attention_mask")
            position_ids = model_inputs.get("position_ids")
            past_key_values = model_inputs.get("past_key_values")

            # prefill with dynamic sequence length, decode with static sequence length
            torch._dynamo.mark_static(kv_len)
            for item in past_key_values:
                for sub_item in item:
                    torch._dynamo.mark_static(sub_item)
            
            if input_ids.shape[1] == 1:
                torch._dynamo.mark_static(input_ids)
                if attention_mask is not None:
                    torch._dynamo.mark_static(attention_mask)
                torch._dynamo.mark_static(position_ids)
            else:
                torch._dynamo.mark_static(kv_len)
```

- torchair提供了NPU的图构造/图编译/图执行能力。相关能力全部集成到NPU图的后端，在使用torch.compile接口时，指定NPU图后端来使能。同时提供了开关和config控制图的编译和执行流程。
- 在使用NPU图后端的时候，torchair提供了静态图和动态图两种图执行的能力。根据dynamic参数决定是否走动态图。


# 性能优化

**注**：在modeling_mixtral.py中，被修改的原函数都加了‘__’前缀，可用于对比修改后的函数变化。Mixtral结构中的非MOE部分与Llama类似，通用优化点可参考[Llama](https://gitee.com/ascend/torchair/tree/master/npu_tuned_model/llm/llama)的改动，如固定kv cache大小、替换FlashAttention&&cos/sin优化、Add+RMSNorm融合、全量优化LM Head计算量、qkv融合。本sample重点展示其余改动点。

## GMM使能&&Routing优化

Hugging face原始的MOE实现比较朴素，for循环处理每个专家，单独计算expert_num个FFN，计算效率较低。

CANN提供了[GroupedMatmul][docs/ops/gmm.md · Ascend/MindSpeed - Gitee.com](https://gitee.com/ascend/MindSpeed/blob/master/docs/ops/gmm.md)算子，可以同时计算多个专家，提高计算和搬运效率。为了使能GroupedMatmul算子，我们需修改Routing逻辑，构造对应输入。

主要流程如下：

- 离线合并多专家权重，可参考scripts/split_weight.py

- 通过MM+Softmax+TopK计算每个token对应的expert_id及routing weights

```python
# scripts/models/modeling_mixtral.py
# class MixtralSparseMoeBlock
# router_logits shape is (batch * sequence_length, n_experts)
router_logits = self.gate(hidden_states)

routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
# we cast back to the input dtype
routing_weights = routing_weights.to(hidden_states.dtype)
```

- 通过Sort+Gather，将所有的token从原始bs*seq排布，变成按专家顺序排布

```python
# scripts/models/modeling_mixtral.py
# class MixtralSparseMoeBlock
selected_experts = selected_experts.view(-1)
selected_experts_fp32 = selected_experts.to(torch.int32).to(torch.float)

_, sorted_indices = torch.sort(selected_experts_fp32, dim=-1)
hidden_states = hidden_states.unsqueeze(1).repeat((1, self.top_k, 1)).view(-1, hidden_dim)
# hidden_states_sorted_by_experts is [bs*seq*topk, hidden_size]->[experts_num*expert_capacity, hidden_size]
hidden_states_sorted_by_experts = torch.index_select(hidden_states, 0, sorted_indices)
```

- 通过OneHot+ReduceSum，计算每个专家获得的token数

```python
# scripts/models/modeling_mixtral.py
# class MixtralSparseMoeBlock
# expert_mask shape is [bs*seq*topk, expert_num]
expert_mask = self.one_hot(selected_experts_fp32, num_classes=self.num_experts)
# expert_tokens shape is [expert_num, ]
expert_tokens = torch.sum(expert_mask, dim=0)
expert_tokens = torch.cumsum(expert_tokens, dim=0).to(torch.int64)
```

- 将按专家顺序排布的tokens，以及每个专家获得的token数，一起作为GMM的输入

```python
# scripts/models/modeling_mixtral.py
# class MixtralBlockSparseTop2MLP
mm1_mm3 = gmm.npu_gmm(hidden_states, torch.transpose(self.group_w1_w3, 1, 2),
                        bias=None, group_list=expert_tokens, group_type=0)
mm1, mm3 = mm1_mm3.chunk(2, dim=-1)
intermediate_hidden_states = self.act_fn(mm1) * mm3
hidden_states = gmm.npu_gmm(intermediate_hidden_states, torch.transpose(self.group_w2, 1, 2),
                            bias=None, group_list=expert_tokens, group_type=0)
```

- 将上一步结果FinalizeRouting成原始bs*seq顺序，并与routing weights加权求和，得到整体MOE层的输出

```python
# scripts/models/modeling_mixtral.py
# class MixtralSparseMoeBlock
sorted_indices_fp32 = sorted_indices.to(torch.int32).to(torch.float)
_, unsort_indices = torch.sort(sorted_indices_fp32, dim=-1)
# hidden_states shape is [bs*seq*topk, hidden_size]
hidden_states = torch.index_select(hidden_states_sorted_by_experts, 0, unsort_indices)
# hidden_states shape is [bs*seq, topk, hidden_size]
hidden_states = hidden_states.view(-1, self.top_k, hidden_dim)
# hidden_states shape is [bs*seq, topk, hidden_size]
hidden_states = hidden_states * routing_weights.unsqueeze(-1)
# hidden_states shape is [bs*seq, hidden_size]
hidden_states = torch.sum(hidden_states, dim=1)

if self.world_size > 1:
    dist.all_reduce(hidden_states)
final_hidden_states = hidden_states.view(batch_size, -1, hidden_dim)
```

## HCCL使能AIV

利用Device的AI Vector Core计算单元来加速AllReduce，可参考[HCCL_OP_EXPANSION_MODE环境变量](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha003/apiref/envref/envref_07_0088.html)

```shell
export HCCL_OP_EXPANSION_MODE=AIV
```

# 性能数据

To Be Published


# 性能测试

本sample的目录下提供了手动Tensor并行及mixtral推理的执行样例参考

**基于搭建的conda环境，安装对应的transformers版本**

```shell
pip3 install transformers==4.40.0
```

**安装[MindSpeed](https://gitee.com/ascend/MindSpeed)**

```shell
git clone https://gitee.com/ascend/MindSpeed.git
pip install -e MindSpeed
```

**设置环境变量**

```shell
cann_path=/usr/local/Ascend 
source ${cann_path}/latest/bin/setenv.bash # 昇腾cann包安装目录   

export ASCEND_HOME_PATH=${cann_path}/latest
export HCCL_OP_EXPANSION_MODE=AIV # HCCL AIVector Core加速
```

**手动切分权重**

循环调用scripts/split_weight.py处理每个device对应的权重。其中WORLD_SIZE表示推理的卡数，path_to_mixtral_model_origin为原始完整权重路径，path_to_mixtral_model_tp为TP切分后的新权重落盘路径

```shell
export WORLD_SIZE=8

for((i=0; i<${WORLD_SIZE}; i++))
do
    export LOCAL_RANK=$i
    python scripts/split_weight.py --model-path "path_to_mixtral_model_origin" --output-path "path_to_mixtral_model_after_tp"
done
```

**mixtral多卡推理**

执行如下命令，其中nproc_per_node表示推理的卡数，path_to_mixtral_model_after_tp为TP切分后的新权重落盘路径，execute-mode用于区分单算子和图模式。

```shell
# 图模式
python -m torch.distributed.launch --nproc_per_node=8 scripts/benchmark_mixtral.py --model-path "path_to_mixtral_model_after_tp" --execute-mode dynamo
# 单算子
python -m torch.distributed.launch --nproc_per_node=8 scripts/benchmark_mixtral.py --model-path "path_to_mixtral_model_after_tp" --execute-mode eager
```
