import torch
import torch.utils._pytree as pytree

from torchair.core.utils import logger
from torchair.ge_concrete_graph.fx2ge_converter import ExportSuccess
from torchair.configs.compiler_config import CompilerConfig
from torchair import get_npu_backend


def _save_name(config, model: torch.nn.Module, **kwargs):
    weight_name = {}
    inputs_name = {}
    for name, p in model.named_buffers():
        logger.debug(
            f' named_buffers {name} , tensor id is {id(p)}, size is {p.shape}')
        weight_name[id(p)] = name
    for name, p in model.named_parameters():
        logger.debug(
            f' named_parameters {name} , tensor id is {id(p)}, size is {p.shape}'
        )
        weight_name[id(p)] = name

    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            inputs_name[id(value)] = key
        else:
            flat_args, _ = pytree.tree_flatten((value,))
            for inp in flat_args:
                if isinstance(inp, torch.Tensor):
                    inputs_name[id(inp)] = key

    config.export_config.weight_name = weight_name
    config.export_config.inputs_name = inputs_name
    return


def _get_export_config(export_path, export_name, model, **kwargs):
    config = CompilerConfig()
    config.debug.graph_dump.type = "pbtxt"
    config.export_config.export_mode = True

    _save_name(config, model, **kwargs)

    rank = None
    try:
        rank = torch.distributed.get_rank()
    except:
        logger.info(f'not frontend segmentation')

    if rank is not None:
        config.export_config.export_path_dir = "./export_file_rank_" + str(
            rank) if export_path is None else export_path + "_rank_" + str(
                rank)
        config.export_config.export_name = "export_rank" + str(
            rank) + ".air" if export_name is None else export_name + str(
                rank) + ".air"
    else:
        config.export_config.export_path_dir = "./export_file" if export_path is None else export_path
        config.export_config.export_name = "export.air" if export_name is None else export_name

    return config


def dynamo_export(*args, model: torch.nn.Module, export_path: str = None,
                  export_name: str = None, dynamic: bool = False, **kwargs):
    logger.info(
        f'dynamo_export: export_path: {export_path}, export_name: {export_name}, '
        + f'dynamic: {dynamic}')
    config = _get_export_config(export_path, export_name, model, **kwargs)

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
