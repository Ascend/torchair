import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torchair.utils.path_manager import PathManager
from torchair.configs.compiler_config import CompilerConfig
from torchair import get_npu_backend
from torchair.ge_concrete_graph.fx2ge_converter import ExportSuccess
from torchair.core.utils import logger
from torchair.ge_concrete_graph.compiled_model import CompiledModel, unserialize_graph
from torchair.dynamo_export import dynamo_export
from torchair.utils.export_utils import get_export_file_name


def save_graph(model: torch.nn.Module,
               args: Tuple[Any, ...],
               dynamic: bool = False,
               config=CompilerConfig(),
               save_path: Union[str, pathlib.Path] = "./save_path",
               kwargs: Optional[Dict[str, Any]] = None):
    if isinstance(save_path, pathlib.Path):
        save_path = str(save_path)

    config.export.enable_save_load_mode = True
    if kwargs is None:
        kwargs = {}
    dynamo_export(*args, model=model, export_path=save_path, export_name="compiled_graph", dynamic=dynamic,
                  config=config, **kwargs)


def load_graph(load_path: Union[str, pathlib.Path]):
    if isinstance(load_path, (str, pathlib.Path)):
        load_path = str(load_path)
    file_name = get_export_file_name("compiled_graph")
    with open(load_path + "/" + file_name, "rb") as f:
        serialized_compiled_graph = f.read()

    compiled_model = unserialize_graph(serialized_compiled_graph)

    return compiled_model
