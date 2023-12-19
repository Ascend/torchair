import sys
import torch
from torchair.core.utils import logger

_TORCH_NPU_MODULE = None


def _weight_format_cast(model: torch.nn.Module):
    def _cast_to_internal_format(module: torch.nn.Module, class_name):
        # Add weight format cast for other modules here

        if issubclass(class_name, torch.nn.Linear):
            if module.weight.data.is_cpu:
                raise RuntimeError(f'Cpu weight is not supported.'
                                   f'The format cast to FRACTAL_NZ only supports npu tensor.'
                                   f'You should call model to npu, before calling this API.')

            module.weight.data = _TORCH_NPU_MODULE.npu_format_cast(module.weight.data, 29)  # ACL_FORMAT_FRACTAL_NZ

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

    _weight_format_cast(model)
