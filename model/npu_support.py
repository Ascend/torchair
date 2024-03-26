import importlib

import torch
import torch_npu


def _patch_model_1():
    # For model LearningToPaint.
    from torchbenchmark.models import LearningToPaint
    USE_DEVICE = torch.cuda.is_available() or torch_npu.npu.is_available()
    LearningToPaint.baseline.utils.util.USE_CUDA = USE_DEVICE


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



def patch_torchbench():
    _patch_model_1()
    _patch_model_2()

