import importlib
import json
import logging
import os
import sys
from typing import Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch_npu
from torch_npu.dynamo.torchair._utils.path_manager import PathManager

import common
import torchair

log = logging.getLogger(__name__)
_patch_table = {}


def register_patch(*model_names):
    def meta_decorator(fn):
        for model_name in model_names:
            _patch_table[model_name] = fn
        return fn

    return meta_decorator


def check_transformers_version(required_version):
    import transformers
    if transformers.__version__ != required_version:
        log.warning(f"transformers.__version__ is not equal to {required_version}, which may cause error patch.")


def use_aclnn():
    os.environ["USE_ACLOP"] = "0"


def _hf_t5_mt5_conditionalgeneration_forward_new(
    self,
    hidden_states,
    mask=None,
    key_value_states=None,
    position_bias=None,
    past_key_value=None,
    layer_head_mask=None,
    query_length=None,
    use_cache=False,
    output_attentions=False,
):
    batch_size, seq_length = hidden_states.shape[:2]

    real_seq_length = seq_length

    if past_key_value is not None:
        if len(past_key_value) != 2:
            raise ValueError(f"past_key_value should have 2 past states. Got {len(past_key_value)} past states")
        real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

    key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

    def shape(states):
        return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

    def unshape(states):
        return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

    def project(hidden_states, proj_layer, key_value_states, past_key_value):
        if key_value_states is None:
            hidden_states = shape(proj_layer(hidden_states))
        elif past_key_value is None:
            hidden_states = shape(proj_layer(key_value_states))

        if past_key_value is not None:
            if key_value_states is None:
                hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
            elif past_key_value.shape[2] != key_value_states.shape[1]:
                hidden_states = shape(proj_layer(key_value_states))
            else:
                hidden_states = past_key_value
        return hidden_states

    query_states = shape(self.q(hidden_states))

    key_states = project(
        hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
    )
    value_states = project(
        hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
    )

    scores = torch.matmul(query_states, key_states.transpose(3, 2))

    def process_position_bias():
        if not self.has_relative_attention_bias:
            position_bias = torch.zeros(
                (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
            )
            if self.gradient_checkpointing and self.training:
                position_bias.requires_grad = True
        else:
            position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)

        if past_key_value is not None:
            position_bias = position_bias[:, :, -hidden_states.size(1):, :]

        if mask is not None:
            position_bias = position_bias + mask
        return position_bias

    if position_bias is None:
        position_bias = process_position_bias()

    if self.pruned_heads:
        mask = torch.ones(position_bias.shape[1])
        mask[list(self.pruned_heads)] = 0
        position_bias_masked = position_bias[:, mask.bool()]
    else:
        position_bias_masked = position_bias

    # Only patch here, src code: [scores += position_bias_masked]
    # Prevent from two continuous _to_copy.
    scores = scores.float() + position_bias_masked

    attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)

    attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

    if layer_head_mask is not None:
        attn_weights = attn_weights * layer_head_mask

    attn_output = unshape(torch.matmul(attn_weights, value_states))
    attn_output = self.o(attn_output)

    present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
    outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

    if output_attentions:
        outputs = outputs + (attn_weights,)
    return outputs


@register_patch("LearningToPaint")
def _patch_model_1():
    # For model LearningToPaint.
    from torchbenchmark.models import LearningToPaint
    USE_DEVICE = torch.cuda.is_available() or torch_npu.npu.is_available()
    LearningToPaint.baseline.utils.util.USE_CUDA = USE_DEVICE


@register_patch("hf_GPT2", "hf_GPT2_large")
def _patch_model_2():
    # for model hf_GPT2,hf_GPT2_large (transformers.models.gpt2)
    # torch_npu.compile cannot get gpt2.past_key_value correctly because gpt2.past_key_value is discontinuous.
    # this patch should be removed after torch_npu.compile bug fixed
    # close AddLayerNormFusionPass
    close_add_layer_norm_fusion_pass()
    module_spec = importlib.util.find_spec("transformers")
    if module_spec is None:
        return

    from transformers.models.gpt2 import GPT2Model
    gpt2_src_forward = GPT2Model.forward

    def __gpt2_forward(self, input_ids, past_key_values, attention_mask, token_type_ids, position_ids, head_mask,
                       inputs_embeds, encoder_hidden_states, encoder_attention_mask, use_cache, output_attentions,
                       output_hidden_states, return_dict):

        output = gpt2_src_forward(self, input_ids, past_key_values, attention_mask, token_type_ids, position_ids,
                                  head_mask,
                                  inputs_embeds, encoder_hidden_states, encoder_attention_mask, use_cache,
                                  output_attentions,
                                  output_hidden_states, return_dict)
        if not return_dict:
            if isinstance(output[1], tuple) and isinstance(output[1], tuple) and len(output[1]) == 2:
                output = tuple([output[0], __gpt2_check_presents_make_presents_continuous(output[1]), output[2:]])
        else:
            if output.past_key_values is not None:
                output.past_key_values = __gpt2_check_presents_make_presents_continuous(output.past_key_values)
        return output

    def __gpt2_check_presents_make_presents_continuous(presents):
        output = []
        for k, v in presents:
            output.append((k.contiguous(), v.contiguous()))
        return tuple(output)

    GPT2Model.forward = __gpt2_forward


@register_patch("hf_T5", "hf_T5_base")
def _patch_model_3():
    # For model hf_T5 and hf_T5_base.
    # In these models, accuracy check will fail because in the model's block [T5Attention],
    # two continuous _to_copy are invoked: the first _to_copy converts Tensor to half
    # and the second converts it to float. In eager, there will be a loss of precision.
    # But in graph, there will be a fusion pass to prevent it happens, causing acc check fail.
    try:
        from transformers.models.t5.modeling_t5 import T5Attention
    except (ImportError, ModuleNotFoundError):
        log.warning("Import transformers failed or could not get T5Attention "
                    "from module transformers.models.t5.modeling_t5")
        return
    check_transformers_version("4.36.0")

    T5Attention.forward = _hf_t5_mt5_conditionalgeneration_forward_new


@register_patch("hf_Bart")
def _patch_model_4():
    # For model hf_Bart.
    # In the model, accuracy check will fail because in the model's block [BartAttention],
    # nn.functional.softmax will convert to float with autocast, and there will be a loss of precision in eager
    # In Dynamo, _to_copy will ge.cast float32 and float16, there will be a fussion pass to prevent it happens.
    # Add _to_copy before nn.functional.softmax to invoke TensorMove for accuracy check
    # close AddLayerNormFusionPass
    close_add_layer_norm_fusion_pass()
    try:
        import transformers
        from transformers.models.bart.modeling_bart import BartAttention
    except (ImportError, ModuleNotFoundError):
        log.warning("Import transformers failed or could not get BartAttention "
                    "from module transformers.models.bart.modeling_bart")
        return
    check_transformers_version("4.36.0")

    def new_forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling

        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # only patch here _to_copy inputs before softmax
        attn_weights = torch.ops.aten._to_copy.default(attn_weights)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value

    BartAttention.forward = new_forward


@register_patch("fastNLP_Bert")
def _patch_model_5():
    os.environ['BREAK_GRAPH_OP_LIST'] = 'NN.LINEAR'
    # None-public interface, just for test.
    # This env is added after torchair's init,
    # so need to call break_graph patch again,
    torchair._utils.npu_patch_break_graph()


@register_patch("hf_Longformer")
def _patch_model_6():
    """
    Hf_Longformer failed accurazy test because of discontiguous memory.
    Solving the problem by adding  .contiguous() after .view() and .as_strided in LongformerSelfAttention._chunk.
    This patch would be removed in the near future.
    """
    # close AddLayerNormFusionPass
    close_view_optimise()
    module_spec = importlib.util.find_spec("transformers")
    if module_spec is None:
        return
    from transformers.models.longformer import LongformerSelfAttention
    src_chunk = LongformerSelfAttention._chunk

    def _chunk(cls, hidden_states, window_overlap, onnx_export: bool = False):
        if not onnx_export:
            hidden_states = hidden_states.view(
                hidden_states.size(0),
                torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
                window_overlap * 2,
                hidden_states.size(2),
            ).contiguous()
            chunk_size = list(hidden_states.size())
            chunk_size[1] = chunk_size[1] * 2 - 1

            chunk_stride = list(hidden_states.stride())
            chunk_stride[1] = chunk_stride[1] // 2
            return hidden_states.as_strided(size=chunk_size, stride=chunk_stride).contiguous()
        return src_chunk(hidden_states, window_overlap, True)

    LongformerSelfAttention._chunk = _chunk


@register_patch("soft_actor_critic")
def _patch_model_7():
    """
    soft_actor_critic failed accurazy test because of discontiguous memory.
    Solving the problem by adding  .contiguous() in soft_actor_critic/net.py line:242 SquashedNormal.__init__
    This patch would be removed in the near future.
    """
    from torchbenchmark.models.soft_actor_critic.nets import StochasticActor, SquashedNormal, BetaDist
    import torch.nn.functional as F
    def new_forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        mu, log_std = out.chunk(2, dim=1)
        if self.dist_impl == "pyd":
            log_std = torch.tanh(log_std)
            log_std = self.log_std_low + 0.5 * (
                    self.log_std_high - self.log_std_low
            ) * (log_std + 1)
            std = log_std.exp()
            dist = SquashedNormal(mu.contiguous(), std.contiguous())
        elif self.dist_impl == "beta":
            out = 1.0 + F.softplus(out)
            alpha, beta = out.chunk(2, dim=1)
            dist = BetaDist(alpha, beta)
        return dist

    StochasticActor.forward = new_forward


@register_patch("dcgan", "mobilenet_v2", "phlippe_resnet", "shufflenet_v2_x1_0", "squeezenet1_1", "vgg16",
                "alexnet", "densenet121", "maml_omniglot")
def _patch_model_8():
    """
    close conv amp for some model only in accuracy mode.
    This patch would be removed in the near future.
    """
    if {"--only", "--amp", "--accuracy"} <= set(sys.argv):
        from torch.nn.modules.conv import Conv2d

        def conv2d_amp_disabled(self, x):
            with torch.npu.amp.autocast(enabled=False):
                return self._conv_forward(x, self.weight, self.bias)

        Conv2d.forward = conv2d_amp_disabled


@register_patch("timm_nfnet")
def _patch_model_9():
    # close conv amp for timm_nfnet only in accuracy mode.
    # Increase the batch_size to a larger size 16,
    # to mitigate the impact of BatchNorm's tolerance on convolution
    if {"--only", "--amp", "--accuracy"} <= set(sys.argv):
        try:
            import timm
            import torch.nn.functional as F
            from timm.layers.std_conv import ScaledStdConv2dSame
            from timm.layers.padding import pad_same
        except (ImportError, ModuleNotFoundError):
            log.warning("Import timm failed or could not get ScaledStdConv2dSame"
                        "from module timm.layers.std_conv.ScaledStdConv2dSame")
            return
        if timm.__version__ != '0.9.16':
            log.warning("timm.__version__ is not equal to 0.9.16, which may cause error patch.")

        def new_forward(self, x):
            if self.same_pad:
                x = pad_same(x, self.kernel_size, self.stride, self.dilation)
            weight = F.batch_norm(
                self.weight.reshape(1, self.out_channels, -1), None, None,
                weight=(self.gain * self.scale).view(-1),
                training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
            with torch.npu.amp.autocast(enabled=False):
                return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        ScaledStdConv2dSame.forward = new_forward

    try:
        from torchbenchmark.models.timm_nfnet import Model
    except (ImportError, ModuleNotFoundError):
        log.warning("Import Model failed or could not find timm_nfnet"
                    "from module torchbenchmark.models.timm_nfnet.Model")
        return

    def new__init(self, test, device, jit=False, batch_size=None, extra_args=None):
        super(Model, self).__init__(test=test, model_name='dm_nfnet_f0',
                                    device=device, batch_size=16, extra_args=extra_args)

    Model.__init__ = new__init


@register_patch("nvidia_deeprecommender")
def _patch_model_10():
    try:
        from torch_npu.contrib import transfer_to_npu
    except (ImportError, ModuleNotFoundError):
        log.warning("NPU_FlAG is False!")
        return

    try:
        from torchbenchmark.models.nvidia_deeprecommender.nvtrain import DeepRecommenderTrainBenchmark
    except (ImportError, ModuleNotFoundError):
        log.warning("Import nvidia_deeprecommender failed or could not get DeepRecommenderTrainBenchmark"
                    "from module torchbenchmark.models.nvidia_deeprecommender.nvtrain.DeepRecommenderTrainBenchmark")
        return

    def new_init(self, device="cpu", jit=False, batch_size=256, process_command_line=False):
        self.TrainInit("cuda", jit, batch_size, process_command_line)

    DeepRecommenderTrainBenchmark.__init__ = new_init


@register_patch("functorch_dp_cifar10")
def _patch_model_11():
    if {"--only", "--amp", "--accuracy"} <= set(sys.argv):
        try:
            from torchbenchmark.models.functorch_dp_cifar10 import Model
            import torchvision.models as models
        except (ImportError, ModuleNotFoundError):
            log.warning("import torchvision fail or could not get Model from module "
                        "torchbenchmark.models.functorch_dp_cifar10")
            return

        def new_init(self, test, device, batch_size=None, extra_args=[]):
            super(Model, self).__init__(test=test, device=device, batch_size=32, extra_args=extra_args)
            self.model = models.__dict__['resnet18'](
                pretrained=False, norm_layer=(lambda c: nn.GroupNorm(min(c, 32), c)))
            self.model = self.model.to(device)
            self.example_inputs = (
                torch.randn((self.batch_size, 3, 32, 32), device=self.device),
            )
            self.example_target = torch.randint(0, 10, (self.batch_size,), device=self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            self.criterion = nn.CrossEntropyLoss()

        Model.__init__ = new_init


def create_fusion_switch_file():
    fusion_config = {}
    fusion_config.setdefault("Switch", {}).setdefault("GraphFusion", {})["AddLayerNormFusionPass"] = "off"
    fusion_config_file = os.getcwd() + "/fusion_switch.cfg"
    PathManager.check_path_writeable_and_safety(fusion_config_file)
    with os.fdopen(os.open(fusion_config_file, os.O_WRONLY | os.O_CREAT, mode=600), 'w') as f:
        json.dump(fusion_config, f)
    config = torchair.CompilerConfig()
    config.fusion_config.fusion_switch_file = fusion_config_file

    def clean_fusion_config_file():
        PathManager.remove_file_safety(fusion_config_file)

    from common import register_callback
    register_callback(clean_fusion_config_file)
    return config


def close_view_optimise():
    config = create_fusion_switch_file()
    config.experimental_config.enable_view_optimize = False
    npu_backend = torchair.get_npu_backend(compiler_config=config)

    def compile_with_view_switch(args):
        return torch._dynamo.optimize(npu_backend, nopython=args.nopython)
    common.compile_with_backend = compile_with_view_switch


def close_add_layer_norm_fusion_pass():
    npu_backend = torchair.get_npu_backend(compiler_config=create_fusion_switch_file())

    def compile_with_fusion_switch(args):
        return torch._dynamo.optimize(npu_backend, nopython=args.nopython)
    common.compile_with_backend = compile_with_fusion_switch


@register_patch("hf_BigBird", "hf_DistilBert")
def _patch_model_12():
    # close AddLayerNormFusionPass
    close_add_layer_norm_fusion_pass()


@register_patch("moco")
def _patch_model_13():
    from argparse import Namespace
    import torch.distributed as dist

    try:
        from torch_npu.contrib import transfer_to_npu
    except (ImportError, ModuleNotFoundError):
        log.warning("NPU_FlAG is False!")
        return

    try:
        import torchvision.models as models
        from torchbenchmark.models.moco import Model
        from torchbenchmark.models.moco.moco.builder import MoCo
    except (ImportError, ModuleNotFoundError):
        log.warning("import torchvision fail or could not get Model,MoCo from module torchbenchmark.models.moco ")
        return

    def new_init(self, test, device, batch_size=None, extra_args=[]):
        super(Model, self).__init__(test=test, device=device, batch_size=batch_size, extra_args=extra_args)
        self.opt = Namespace(**{
            "arch": "resnet50", "epochs": 2, "start_epoch": 0, "lr": 0.03, "schedule": [120, 160], "momentum": 0.9,
            "weight_decay": 1e-4, "gpu": None, "moco_dim": 128, "moco_k": 32000, "moco_m": 0.999, "moco_t": 0.07,
            "mlp": False, "aug_plus": False, "cos": False, "fake_data": True, "distributed": True,
        })
        try:
            dist.init_process_group(backend="nccl", init_method="tcp://localhost:10001", world_size=1, rank=0)
        except RuntimeError:
            pass  # already initialized?

        if device == "cpu":
            raise NotImplementedError("DistributedDataParallel/allgather requires npu")

        self.model = MoCo(
            models.__dict__[self.opt.arch],
            self.opt.moco_dim,
            self.opt.moco_k,
            self.opt.moco_m,
            self.opt.moco_t,
            self.opt.mlp,
        )
        self.model.to(self.device)

        # Define loss function (criterion) and optimizer
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            self.opt.lr,
            momentum=self.opt.momentum,
            weight_decay=self.opt.weight_decay,
        )

        def collate_train_fn(data):
            ind = data[0]
            return [batches[2 * ind], batches[2 * ind + 1]], 0

        batches = []
        for i in range(4):
            batches.append(torch.randn(self.batch_size, 3, 224, 224).to(self.device))
        self.example_inputs = torch.utils.data.DataLoader(range(2), collate_fn=collate_train_fn)
        if torch.cuda.is_available():
            for i, (images, _) in enumerate(self.example_inputs):
                images[0] = images[0].cuda(device=0, non_blocking=True)
                images[1] = images[1].cuda(device=0, non_blocking=True)
        else:
            for i, (images, _) in enumerate(self.example_inputs):
                images[0] = images[0].npu(device=0, non_blocking=True)
                images[1] = images[1].npu(device=0, non_blocking=True)

    Model.__init__ = new_init

    if {"--only", "--amp", "--accuracy"} <= set(sys.argv):
        try:
            import torchvision
            from torchvision.models.resnet import ResNet
        except (ImportError, ModuleNotFoundError):
            log.warning("Import torchvision failed or could not get ResNet "
                        "from module torchvision.models.resnet")
            return

        def _new_forward_impl(self, x):
            # See note [TorchScript super()]
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            @torch.compiler.disable(recursive=False)
            def avgpool(x):
                x = self.avgpool(x)
                return x

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

        ResNet._forward_impl = _new_forward_impl


@register_patch("timm_vovnet")
def _patch_model_14():
    if {"--only", "--amp", "--accuracy"} <= set(sys.argv):
        import torch.nn.functional as F
        from torch.nn.modules.conv import Conv2d
        from torch.nn.modules.pooling import AdaptiveAvgPool2d
        from torch.nn.modules.linear import Linear

        def conv2d_amp_disabled(self, x):
            with torch.npu.amp.autocast(enabled=False):
                return self._conv_forward(x, self.weight, self.bias)
        Conv2d.forward = conv2d_amp_disabled

        def adaptive_avgpool_amp_disabled(self, x):
            with torch.npu.amp.autocast(enabled=False):
                return F.adaptive_avg_pool2d(x.float(), self.output_size)
        AdaptiveAvgPool2d.forward = adaptive_avgpool_amp_disabled

        def linear_amp_disabled(self, x):
            with torch.npu.amp.autocast(enabled=False):
                return F.linear(x, self.weight, self.bias)
        Linear.forward = linear_amp_disabled


@register_patch("speech_transformer")
def _patch_model_15():
    # In same scenarios, bmm will influence the format of softmax to NZ;
    # Therefore, specify _softmax input and output format to NHWC for the accuracy test
    close_add_layer_norm_fusion_pass()
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
    from torchair._ge_concrete_graph.utils import specific_op_input_layout, specific_op_output_layout
    from torchair._ge_concrete_graph.ge_converter.aten import _softmax

    def new_conveter_aten__softmax_default(
            self: Tensor, dim: int, half_to_float: bool, meta_outputs: TensorSpec = None
    ):
        """NB: aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor"""
        if half_to_float and self.dtype != DataType.DT_FLOAT16:
            raise RuntimeError(
                "torch.ops.aten._softmax.default: "
                "when half_to_tensor is True, input tensor must be half type.")
        specific_op_input_layout(self, indices=0, layout="NHWC")
        specific_op_output_layout(self, indices=0, layout="NHWC")
        return ge.SoftmaxV2(self, axes=[dim], half_to_float=half_to_float)
    _softmax.conveter_aten__softmax_default = new_conveter_aten__softmax_default


@register_patch("MT5ForConditionalGeneration")
def _patch_model_16():
    # For model huggingface model MT5ForConditionalGeneration.
    # In this model, accuracy check will fail because in the model's block [MT5Attention],
    # two continuous _to_copy are invoked: the first _to_copy converts Tensor to half
    # and the second converts it to float. In eager, there will be a loss of precision.
    # But in graph, there will be a fusion pass to prevent it happens, causing acc check fail.
    try:
        from transformers.models.mt5.modeling_mt5 import MT5Attention
    except (ImportError, ModuleNotFoundError):
        log.warning("Import transformers failed or could not get MT5Attention "
                    "from module transformers.models.mt5.modeling_mt5")
        return
    check_transformers_version("4.36.0")

    MT5Attention.forward = _hf_t5_mt5_conditionalgeneration_forward_new

    from common import register_callback
    # use aclop for MT5ForConditionalGeneration
    os.environ["USE_ACLOP"] = "1"
    # set it back to aclnn afterwards
    register_callback(use_aclnn)


@register_patch("basic_gnn_gin")
def _patch_model_17():
    from common import register_callback
    # use aclop for basic_gnn_gin
    os.environ["USE_ACLOP"] = "1"
    # set it back to aclnn afterwards
    register_callback(use_aclnn)


@register_patch("resnet50", "resnet152", "resnext50_32x4d")
def _patch_model_18():
    if {"--only", "--amp", "--accuracy"} <= set(sys.argv):
        try:
            import torchvision.models as models
        except (ImportError, ModuleNotFoundError):
            log.warning("Import torchvision failed or could not get models "
                    "from module torchvision.models")
            return

        if 'resnet50' in sys.argv:
            from torchbenchmark.models.resnet50 import Model
            model = 'resnet50'
            weight = models.ResNet50_Weights.IMAGENET1K_V1
        elif 'resnet152' in sys.argv:
            from torchbenchmark.models.resnet152 import Model
            model = 'resnet152'
            weight = models.ResNet152_Weights.IMAGENET1K_V1
        elif 'resnext50_32x4d' in sys.argv:
            from torchbenchmark.models.resnext50_32x4d import Model
            model = 'resnext50_32x4d'
            weight = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
        else:
            raise RuntimeError("args.only expect model resnet50, resnet152 or resnext50_32x4d")

        def new_init(self, test, device, batch_size=None, extra_args=[]):
            super(Model, self).__init__(model_name=model, test=test, device=device,
                                        batch_size=32, weights=weight,
                                        extra_args=extra_args)
        Model.__init__ = new_init


def patch_model(model_name):
    if model_name not in _patch_table.keys():
        return
    # do patch
    _patch_table[model_name]()
