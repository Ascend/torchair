import os
import torch
import torch.utils._pytree as pytree

import torchair
from torchair.core.utils import logger
from torchair._ge_concrete_graph.fx2ge_converter import ExportSuccess
from torchair.configs.compiler_config import CompilerConfig
from torchair import get_npu_backend

__all__ = ["dynamo_export"]


def _get_model_weight_names(model: torch.nn.Module):
    weight_name = {}
    for name, p in model.named_buffers():
        logger.debug(
            f' named_buffers {name} , tensor id is {id(p)}, size is {p.shape}')
        weight_name[id(p)] = name
    for name, p in model.named_parameters():
        logger.debug(
            f' named_parameters {name} , tensor id is {id(p)}, size is {p.shape}'
        )
        weight_name[id(p)] = name

    return weight_name


def _get_export_config(model, export_path: str, export_name: str, config=CompilerConfig()):
    config.export.export_mode = True
    config.export.export_path_dir = export_path
    config.export.export_name = export_name
    config.experimental_config.enable_ref_data = True
    config.export.weight_name = _get_model_weight_names(model)
    return config


def _is_symlink(path):
    path_abspath = os.path.abspath(path)
    if os.path.islink(path_abspath):
        logger.error(f"Target file path {path_abspath} should not be an symbolic link.")
        return True
    return False


def dynamo_export(*args, model: torch.nn.Module, export_path: str = "export_file",
                  export_name: str = "export", dynamic: bool = False, config=CompilerConfig(), **kwargs):
    # get last name
    export_name = os.path.split(export_name)[-1]
    # check symbolic link
    if _is_symlink(export_path):
        return

    target_path = os.path.join(export_path, export_name) + ".air"
    if _is_symlink(target_path):
        return

    if torch.__version__ < '2.3.0':
        torchair.patch_for_hcom()
    logger.info(f'dynamo_export: export_path: {export_path}, export_name: {export_name}, dynamic: {dynamic}')
    config = _get_export_config(model, export_path, export_name, config)

    torch._dynamo.reset()
    model = torch.compile(model,
                          backend=get_npu_backend(compiler_config=config),
                          fullgraph=True,
                          dynamic=dynamic)

    with torch.no_grad():
        try:
            model(*args, **kwargs)
        except ExportSuccess as e:
            logger.info(e)
        else:
            logger.info(f'export error!')
