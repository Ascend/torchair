import importlib
import logging

import torch
import torch.nn as nn
import torch_npu


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
                raise ValueError(f"past_key_value should have 2 past states. Got { len(past_key_value)} past states")
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
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

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


def patch_model(model_name):
    if model_name not in _patch_table.keys():
        return
    # do patch
    _patch_table[model_name]()
