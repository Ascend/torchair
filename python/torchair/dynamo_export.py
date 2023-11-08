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


def _get_export_config(model, export_path: str, export_name: str,
                       auto_atc_config_generated: bool = False, **kwargs):
    config = CompilerConfig()
    config.debug.graph_dump.type = "pbtxt"
    config.export_config.export_mode = True
    config.export_config.auto_atc_config_generated = auto_atc_config_generated
    config.export_config.export_path_dir = export_path
    config.export_config.export_name = export_name

    _save_name(config, model, **kwargs)
    return config


def dynamo_export(*args, model: torch.nn.Module, export_path: str = "export_file",
                  export_name: str = "export", dynamic: bool = False,
                  auto_atc_config_generated: bool = False, **kwargs):
    from torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce import functional_collectives_context
    logger.info(
        f'dynamo_export: export_path: {export_path}, export_name: {export_name}, '
        + f'dynamic: {dynamic}, auto_atc_config_generated: {auto_atc_config_generated}')
    config = _get_export_config(model, export_path, export_name, auto_atc_config_generated, **kwargs)

    torch._dynamo.reset()
    model = torch.compile(model,
                          backend=get_npu_backend(compiler_config=config),
                          fullgraph=True,
                          dynamic=dynamic)

    with torch.no_grad(), functional_collectives_context():
        try:
            model(*args, **kwargs)
        except ExportSuccess as e:
            logger.info(e)
        else:
            logger.info(f'export error!')
