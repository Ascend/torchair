import logging
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union, Callable

import torch
from torch.fx.node import Argument, Target
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensor
from torchair.configs.compiler_config import CompilerConfig



class ValuePack:
    def __init__(self, meta, npu_meta=None) -> None:
        self._meta = meta
        self._npu_meta = meta if npu_meta is None else npu_meta

    @property
    def meta(self):
        return self._meta

    @property
    def npu(self):
        return self._npu_meta

    def __repr__(self) -> str:
        if isinstance(self._meta, FakeTensor):
            meta_str = f"FakeTensor(dtype={self._meta.dtype}, size={list(self._meta.size())}"
        elif isinstance(self._meta, torch.Tensor):
            meta_str = f"torch.Tensor(dtype={self._meta.dtype}, size={list(self._meta.size())}"
        elif isinstance(self._meta, torch.SymInt):
            meta_str = f"torch.SymInt({self._meta})"
        else:
            try:
                meta_str = f"{type(self._meta)}({self._meta})"
            except Exception:
                meta_str = f"{type(self._meta)}"
        return f'Pack(meta:{meta_str} npu:{self._npu_meta})'


class ConcreteGraphBase(ABC):
    @abstractmethod
    def context(self):
        '''
        返回一个上下文管理器，用于在with语句中使用，表示正在构图
        '''
        pass

    @abstractmethod
    def parse_input(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        '''
        处理fx图的输入节点，其中meta_outputs为MetaTensor时的输出
        '''
        pass

    @abstractmethod
    def parse_output(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        '''
        处理fx图的输出节点，其中meta_outputs为MetaTensor时的输出，args为同时包含MetaTensor和NpuTensor输出的ValuePack
        '''
        pass

    @abstractmethod
    def parse_symlist(self, syms):
        '''
        处理fx图中的SymIntList输入，输入syms可能为int或者SymInt，或者一个ValuePack
        '''
        pass

    @abstractmethod
    def parse_node(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        '''
        处理fx图中的普通节点，其中meta_outputs为MetaTensor时的输出，对于args中的每个值：
        - 如果该入参中的值中包含symInt, 则arg为NpuTensor
        - 如果该入参为Tensor, 则arg为NpuTensor
        - 如果为基本类型，则arg为基本类型
        - 当节点的输入本身为List[Tensor]时，则为List[NpuTensor]
        '''
        pass

    @abstractmethod
    def dump(self, path: str):
        '''
        dump图到文件
        '''
        pass

    @abstractmethod
    def compile(self) -> Any:
        '''
        编译图，返回一个可调用的对象，用于执行fx图
        '''
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        '''
        ConcreteGraphBase实例可以直接调用，用于执行fx图
        '''
        pass


def _is_symlist(arg):
    if not isinstance(arg, (list, tuple)) or len(arg) == 0:
        return False

    for v in arg:
        if isinstance(v, ValuePack) and isinstance(v.meta, (int, torch.SymInt)):
            return True

    return False
