import sys
import torch
from torchair.core.utils import logger

__all__ = ["use_internal_format_weight"]

_TORCH_NPU_MODULE = None
_DEEPSPEED_MODULE = None


def _weight_format_cast(model: torch.nn.Module):
    def _cast_to_internal_format(module: torch.nn.Module, class_name):

        def _cast_to_internal_format_for_quant_conv2d(module: torch.nn.Module, class_name):
            if not isinstance(module, _TORCH_NPU_MODULE.contrib.module.quant_conv2d.QuantConv2d):
                return
            if module.weight is None or module.weight.data is None:
                return
            if module.weight.data.is_cpu:
                raise RuntimeError(f'Cpu weight is not supported.'
                                   f'The format cast to FRACTAL_Z only supports npu tensor.'
                                   f'You should call model to npu, before calling this API.')
            if module.groups > 1 or module.weight.dtype != torch.int8:
                return
            # ACL_FORMAT_FRACTAL_Z
            module.weight.data = _TORCH_NPU_MODULE.npu_format_cast(module.weight.data, 4)

            if module.scale.dtype != torch.float32:
                return
            module.scale.data = _TORCH_NPU_MODULE.npu_trans_quant_param(module.scale, module.offset)

        # Add weight format cast for other modules here
        if issubclass(class_name, torch.nn.Conv2d):
            if module.weight is None or module.weight.data is None:
                return
            if module.weight.data.is_cpu:
                raise RuntimeError(f'Cpu weight is not supported.'
                                   f'The format cast to FRACTAL_Z only supports npu tensor.'
                                   f'You should call model to npu, before calling this API.')
            if module.groups > 1 or module.weight.dtype != torch.float16:
                return

            module.weight.data = _TORCH_NPU_MODULE.npu_format_cast(module.weight.data, 4)  # ACL_FORMAT_FRACTAL_Z

        if issubclass(class_name, torch.nn.Linear):
            if module.weight.data.is_cpu:
                raise RuntimeError(f'Cpu weight is not supported.'
                                   f'The format cast to FRACTAL_NZ only supports npu tensor.'
                                   f'You should call model to npu, before calling this API.')

            module.weight.data = _TORCH_NPU_MODULE.npu_format_cast(module.weight.data, 29)  # ACL_FORMAT_FRACTAL_NZ

        if _DEEPSPEED_MODULE is not None:
            if issubclass(class_name, (_DEEPSPEED_MODULE.module_inject.layers.LinearLayer,
                                       _DEEPSPEED_MODULE.module_inject.layers.LinearAllreduce)):
                if module.weight.data.is_cpu:
                    raise RuntimeError(f'Cpu weight is not supported.'
                                       f'The format cast to FRACTAL_NZ only supports npu tensor.'
                                       f'You should call model to npu, before calling this API.')
                module.weight.data = _TORCH_NPU_MODULE.npu_format_cast(module.weight.data, 29)  # ACL_FORMAT_FRACTAL_NZ

        if isinstance(module, _TORCH_NPU_MODULE.contrib.module.linear_a8w8_quant.LinearA8W8Quant):
            module.scale.data = _TORCH_NPU_MODULE.npu_trans_quant_param(module.scale, module.offset)
            if "Ascend310P" in _TORCH_NPU_MODULE.npu.get_device_name():
                module.weight.data = _TORCH_NPU_MODULE.npu_format_cast(module.weight.data, 29)
        if isinstance(module, _TORCH_NPU_MODULE.contrib.module.linear_quant.LinearQuant):
            module.weight.data = _TORCH_NPU_MODULE.npu_format_cast(module.weight.data, 29)
        if isinstance(module, _TORCH_NPU_MODULE.contrib.module.linear_weight_quant.LinearWeightQuant):
            if module.quant_scale is not None:
                quant_scale = module.quant_scale.data
                quant_offset = None
                if module.quant_offset is not None:
                    quant_offset = module.quant_offset.data

                module.quant_scale.data = _TORCH_NPU_MODULE.npu_trans_quant_param(quant_scale, quant_offset)
            if "Ascend310P" in _TORCH_NPU_MODULE.npu.get_device_name() or \
               "Ascend910B" in _TORCH_NPU_MODULE.npu.get_device_name():
                module.weight.data = _TORCH_NPU_MODULE.npu_format_cast(module.weight.data, 29) # ACL_FORMAT_FRACTAL_NZ

        _cast_to_internal_format_for_quant_conv2d(module, class_name)

    current_class = model.__class__
    _cast_to_internal_format(model, current_class)

    if not model.children:
        return

    for sub_module in model.children():
        if isinstance(sub_module, torch.nn.Module):
            _weight_format_cast(sub_module)


def use_internal_format_weight(model: torch.nn.Module):
    if 'torch_npu' not in sys.modules:
        logger.error(f'This interface is only enabled in a torch npu env.')
        return
    global _TORCH_NPU_MODULE
    _TORCH_NPU_MODULE = sys.modules['torch_npu']

    if 'deepspeed' in sys.modules:
        global _DEEPSPEED_MODULE
        _DEEPSPEED_MODULE = sys.modules['deepspeed']

    _weight_format_cast(model)
