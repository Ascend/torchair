__all__ = []

import functools
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Tuple
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist
from torch.fx import Interpreter
from torch.fx.graph_module import GraphModule
from torch.fx.node import Argument, Target

from npugraph_ex.core.utils import logger


def _timestamp():
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S%f")


_dump_info = threading.local()


@contextmanager
def _dump_ctx(*, node=None):
    try:
        _dump_info.node = node
        yield
    finally:
        _dump_info.node = None


def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bfloat16:
            return x.detach().cpu().to(torch.float32).numpy()
        return x.detach().cpu().numpy()
    try:
        return np.array(x)
    except Exception:
        return np.array(f'{x}')


def _get_node_stack(node):
    return node.stack_trace.split(' File ')[-1].replace('\n', '') if node.stack_trace else '<no stack>'


def _is_dumping():
    return hasattr(_dump_info, 'node') and _dump_info.node is not None


def _trace_dump(f):
    @functools.wraps(f)
    def inner(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]):
        if not _is_dumping():
            return f(self, target, args, kwargs)

        logger.debug(
            f'Dumping {target} called from {_get_node_stack(_dump_info.node)} to {self.step_path}')
        os.makedirs(self.step_path, exist_ok=True)

        timestamp = _timestamp()
        for i, arg in enumerate(args):
            if isinstance(arg, (list, tuple)) and len(arg) and isinstance(arg[0], torch.Tensor):
                for j, tensor in enumerate(arg):
                    fn = self.full_path(f'{_dump_info.node.name}-{target}.INPUT.{i}.{j}.{timestamp}')
                    np.save(fn, _as_numpy(tensor))
                continue
            fn = self.full_path(f'{_dump_info.node.name}-{target}.INPUT.{i}.{timestamp}')
            np.save(fn, _as_numpy(arg))

        result = f(self, target, args, kwargs)

        for i, tensor in enumerate(result if isinstance(result, (list, tuple)) else [result]):
            fn = self.full_path(f'{_dump_info.node.name}-{target}.OUTPUT.{i}.{timestamp}')
            np.save(fn, _as_numpy(tensor))
        return result

    return inner


class _NpuFxDumper(Interpreter):
    """
    Interpreter for collect npu graph meta from fx graph, such as sym of output, input shape ranges, etc.
    """

    def __init__(self, gm: GraphModule, config=None, name=None):
        super().__init__(gm)
        self._config = config
        self._step = 0
        self._name = name if name else f'gm_{_timestamp()}_dump'

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def run(self, *args, **kwargs):
        result = super().run(*args, **kwargs)
        self._step += 1
        return result

    def full_path(self, name):
        return os.path.join(self.step_path, name)

    @property
    def step_path(self):
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            global_rank = dist.get_rank()
            worldsize = dist.get_world_size()
            path = f'worldsize{worldsize}_global_rank{global_rank}/'
            return os.path.join(os.path.dirname(self._config.eager_data_dump_full_path(path + f'{self._name}/')), f'{self._step}')
        return os.path.join(os.path.dirname(self._config.eager_data_dump_full_path(f'worldsize1_global_rank0/{self._name}/')),
                            f'{self._step}')

    def run_node(self, n):
        with _dump_ctx(node=n):
            return super().run_node(n)

    @_trace_dump
    def call_function(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        return super().call_function(target, args=args, kwargs=kwargs)
