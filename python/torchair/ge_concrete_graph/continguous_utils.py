import copy

from typing import Any, Dict, List, Tuple, Union, Callable

import torch
from torch.fx.node import Argument, Target
from torch._dynamo.utils import detect_fake_mode
from torch._subclasses.fake_tensor import FakeTensor

from torchair.core.utils import logger
from torchair.ge_concrete_graph.ge_graph import is_sym, Tensor
from torchair.ge_concrete_graph.continguous import gen_contiguous_storagesize, gen_contiguous_stride, optimize_view

view_white_list = {'aten.permute.default': 0, 'aten.view.default': 0, 'aten.transpose.int': 0, 'aten.t.default': 0}


class ViewFakeTensor:
    def __init__(self, meta, srcshape):
        self._meta = meta
        self._srcshape = srcshape
        self._mapsym = {}
    
    @property
    def meta(self):
        return self._meta

    @property
    def srcshape(self):
        return self._srcshape

    @property
    def mapsym(self):
        return self._mapsym
    
    def set_mapsym(self, mapsym):
        if not isinstance(mapsym, dict):
            raise AssertionError
        self._mapsym.update(mapsym)

    def set_meta(self, meta):
        self._meta = meta


def set_ge_outputs(ge_outputs, meta_outputs):
    if isinstance(ge_outputs, Tensor):
        ge_outputs.set_meta(meta_outputs)
    elif isinstance(ge_outputs, int):
        if not isinstance(meta_outputs, (torch.SymInt, int)):
            raise AssertionError("meta_outputs must be a torch.SymInt or an integer.")
    else:
        if not isinstance(ge_outputs, (list, tuple)):
            raise AssertionError
        if not isinstance(meta_outputs, (list, tuple)):
            raise AssertionError
        if len(ge_outputs) != len(meta_outputs):
            raise AssertionError
        for meta_output, ge_output in zip(meta_outputs, ge_outputs):
            if meta_output is None:
                continue
            set_ge_outputs(ge_output, meta_output)


def set_meta_tensor_info(arg):
    view_shape = arg.symsize
    view_stride = gen_contiguous_stride(view_shape)
    view_storagesize = gen_contiguous_storagesize(view_shape)
    meta_empty = torch.empty([view_storagesize], device=arg.meta.device, dtype=arg.meta.dtype)
    args_meta = meta_empty.as_strided(view_shape, view_stride, 0)
        
    return args_meta


def set_fake_mapsym(view_list, fake):
    mapsym = {}
    if isinstance(view_list, Tensor):
        for idx, sym in enumerate(view_list.meta):
            if is_sym(sym):
                mapsym[str(sym)] = view_list.node.input[idx]
        fake.set_mapsym(mapsym)


def is_view_case(target, args, meta_outputs):
    if str(target) in view_white_list:
        input_tensor_index = view_white_list.get(str(target))
        if input_tensor_index is None:
            raise AssertionError("Unsupported case for contiguous.")

        if not hasattr(args[input_tensor_index], "view_faketensor"):
            meta_con = set_meta_tensor_info(args[input_tensor_index])
            fake = ViewFakeTensor(meta_con, args[input_tensor_index].symsize)
        else:
            fake = getattr(args[input_tensor_index], "view_faketensor")

        if str(target) == "aten.view.default":
            set_fake_mapsym(args[1], fake)
            if not all([(not is_sym(meta_dim) or str(meta_dim) in fake.mapsym.keys()) \
                for meta_dim in meta_outputs.size()]):
                return False
        setattr(args[input_tensor_index], "view_faketensor", fake)
        return True
    else:
        return False


def _get_npu_outputs_of_view_ops(target, args, kwargs, meta_outputs):
    input_tensor_index = view_white_list.get(str(target))
    args_meta = [None] * len(args)
    for idx, arg in enumerate(args):
        if hasattr(arg, "view_faketensor"):
            args_meta[idx] = getattr(arg, "view_faketensor").meta
        elif isinstance(arg, Tensor):
            args_meta[idx] = arg.meta
        else:
            args_meta[idx] = arg

    fake_mode = detect_fake_mode(None)
    with fake_mode:
        meta_outputs_contiguous = target(*args_meta, **kwargs)

    fake = copy.copy(getattr(args[input_tensor_index], "view_faketensor"))
    fake.set_meta(meta_outputs_contiguous)
    if isinstance(meta_outputs_contiguous, (tuple, list)):
        view_npu_outputs = [copy.copy(args[input_tensor_index]) for _ in range(len(meta_outputs))]
        for idx, npu_output in enumerate(view_npu_outputs):
            setattr(npu_output, "view_faketensor", fake)
    elif isinstance(meta_outputs_contiguous, FakeTensor):
        view_npu_outputs = copy.copy(args[input_tensor_index])
        setattr(view_npu_outputs, "view_faketensor", fake)
    else:
        raise AssertionError("Other output types are not supported for view ops, except for tensor and tensor list.")
    
    set_ge_outputs(view_npu_outputs, meta_outputs)
    return view_npu_outputs


def guard_view_input(func):
    def wrapper(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        if self.config.experimental_config.enable_view_optimize:
            #判断经过的target是可跳过的view类操作，则刷新反推的meta信息
            #若为不可跳过的view类操作或计算类节点，则对参数做反推的推导
            if is_view_case(target, args, meta_outputs):
                logger.info(f"Do view optimize, skip the operator {str(target)}")
                ge_outputs = _get_npu_outputs_of_view_ops(target, args, kwargs, meta_outputs)
                return ge_outputs                     
            else:
                args_new = [optimize_view(arg, self.graph) for arg in args]
            return func(self, target, args_new, kwargs, meta_outputs)
        else:
            return func(self, target, args, kwargs, meta_outputs)

    return wrapper