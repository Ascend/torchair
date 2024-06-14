import importlib
import logging
import os
from typing import Optional, Tuple, Type, Union
import sys
import torch
import torch.nn as nn
import torch_npu
import torchair


log = logging.getLogger(__name__)
_patch_table = {}


def register_patch(*model_names):
    def meta_decorator(fn):
        for model_name in model_names:
            _patch_table[model_name] = fn
        return fn
    return meta_decorator


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
        import transformers
        from transformers.models.t5.modeling_t5 import T5Attention
    except (ImportError, ModuleNotFoundError):
        log.warning("Import transformers failed or could not get T5Attention "
                    "from module transformers.models.t5.modeling_t5")
        return
    if transformers.__version__ != '4.36.0':
        log.warning("transformers.__version__ is not equal to 4.36.0, which may cause error patch.")

    def new_forward(
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

    T5Attention.forward = new_forward


@register_patch("hf_Bart")
def _patch_model_4():
    # For model hf_Bart.
    # In the model, accuracy check will fail because in the model's block [BartAttention],
    # nn.functional.softmax will convert to float with autocast, and there will be a loss of precision in eager
    # In Dynamo, _to_copy will ge.cast float32 and float16, there will be a fussion pass to prevent it happens.
    # Add _to_copy before nn.functional.softmax to invoke TensorMove for accuracy check
    try:
        import transformers
        from transformers.models.bart.modeling_bart import BartAttention
    except (ImportError, ModuleNotFoundError):
        log.warning("Import transformers failed or could not get BartAttention "
                    "from module transformers.models.bart.modeling_bart")
        return
    if transformers.__version__ != '4.36.0':
        log.warning("transformers.__version__ is not equal to 4.36.0, which may cause error patch.")

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

        #only patch here _to_copy inputs before softmax
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
def _path_model_6():
    """
    Hf_Longformer failed accurazy test because of discontiguous memory.
    Solving the problem by adding  .contiguous() after .view() and .as_strided in LongformerSelfAttention._chunk.
    This patch would be removed in the near future.
    """
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
def _path_model_7():
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


@register_patch("dcgan", "mobilenet_v2", "phlippe_resnet", "timm_vision_transformer", "shufflenet_v2_x1_0",
                "squeezenet1_1")
def _path_model_8():
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
def _path_model_9():
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
def _path_model_10():
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
            import torchvision
            from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet, conv1x1
        except (ImportError, ModuleNotFoundError):
            log.warning("Import torchvision failed or could not get BasicBlock,Bottleneck,ResNet,conv1x1 "
                        "from module torchvision.models.resnet")
            return

        class GraphBreak(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                torch._dynamo.graph_break()
                return x

        def new_forward(self, x):
            identity = x

            out = self.conv1(x)
            torch._dynamo.graph_break()
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            torch._dynamo.graph_break()
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out

        def _new_forward_impl(self, x):
            # See note [TorchScript super()]
            x = self.conv1(x)
            torch._dynamo.graph_break()
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

            return x

        def _new_make_layer(
                self,
                block: Type[Union[BasicBlock, Bottleneck]],
                planes: int,
                blocks: int,
                stride: int = 1,
                dilate: bool = False,
        ) -> nn.Sequential:
            norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            if dilate:
                self.dilation *= stride
                stride = 1
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    GraphBreak(),
                    norm_layer(planes * block.expansion),
                )

            layers = []
            layers.append(
                block(
                    self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation,
                    norm_layer
                )
            )
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(
                    block(
                        self.inplanes,
                        planes,
                        groups=self.groups,
                        base_width=self.base_width,
                        dilation=self.dilation,
                        norm_layer=norm_layer,
                    )
                )

            return nn.Sequential(*layers)

        ResNet._make_layer = _new_make_layer
        ResNet._forward_impl = _new_forward_impl
        BasicBlock.forward = new_forward



def patch_model(model_name):
    if model_name not in _patch_table.keys():
        return
    # do patch
    _patch_table[model_name]()
