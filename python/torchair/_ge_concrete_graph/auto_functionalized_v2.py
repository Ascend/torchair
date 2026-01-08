from torch._dynamo.utils import detect_fake_mode
from torchair.ge._ge_graph import get_default_ge_graph
from torchair._ge_concrete_graph.ge_converter.converter_utils import *
from torchair._ge_concrete_graph.infer_symbol_calculate import infer_ge_output_by_symbol_calculate
from torchair._ge_concrete_graph.utils import is_host_data_tensor
from torchair._ge_concrete_graph.infer_symbol_shape import infer_and_gen_sym_shape_silent


def _view_copy(self, bases_list, *, dependencies = [], out_op = None):
    dst = bases_list[self.base_index]
    src = self._as_strided
    size = self.size
    stride = self.stride
    storage_offset = self.storage_offset

    if isinstance(size, List):
        src_dim = len(size)
        src_stride = [1] * src_dim
        for i in range(src_dim - 1):
            src_stride[src_dim - i - 2] = src_stride[src_dim - i - 1] * size[src_dim - i - 1]
    else:
        src_dim = size.symsize[0]
        src_stride = [1] * src_dim
        src_stride[-1] = ge.Const(1, dtype=DataType.DT_INT64)
        index = len(src_stride) - 2
        while(index >= 0):
            src_stride[index] = ge.Mul(src_stride[index + 1], ge.Gather(size, index + 1))
            index = index - 1
        src_stride = ge.Pack(src_stride, N=src_dim, axis=0)    
    return ge.ViewCopy(dst, size, stride, storage_offset, src, size, src_stride, 0, dependencies=dependencies)


def _not_view_copy(self, bases_list, *, dependencies = [], out_op = None):
    """
    auto_functionalize_v2会将TensorMove返回给后续节点作为输入,
    本函数通过in-place op上TensorMove输入的input_desc name找到同名的输出(即ref输出),将该ref输出返回给后续节点,
    从而使后续pass能够消除TensorMove

    * ******************************************
    *        Data                     Data     
    *          |                        |       
    *      TensorMove ---|          TensorMove  
    *          |         |              |       
    *     in-place op    |   --->   in-place op 
    *                   /               |          
    *                  /                |       
    *              output             output    
    * ******************************************   
    """    
    for i, input_tensor in enumerate(out_op.input):
        if (input_tensor == bases_list[self.base_index].tensor):
            input_name = out_op.input_desc[i].name
            if input_name is None:
                raise RuntimeError(f'Can not find input: {bases_list[self.base_index].tensor} in in-place op of auto_functionalize_v2')    
            for out_index, output_name in enumerate(out_op.output_desc):
                if (output_name.name == input_name):
                    return Tensor(out_op, out_index)
    raise RuntimeError(f'Can not find output: {bases_list[self.base_index].tensor} in in-place op of auto_functionalize_v2')   


def _get_meta_attr(input_value):
    if hasattr(input_value, 'meta'):
        return input_value.meta
    return input_value


def _regenerate_slice_view(self, bases_list, symbol_input_map):
    fake_mode = detect_fake_mode(None)
    with fake_mode:
        meta_out = torch.ops.aten.slice.Tensor(
            _get_meta_attr(bases_list[self.base_index]), _get_meta_attr(self.dim), _get_meta_attr(self.start), _get_meta_attr(self.end)
        )
    self.size = _sym_list_to_ge_tensor(list(meta_out.size()), symbol_input_map)
    self.stride = _sym_list_to_ge_tensor(list(meta_out.stride()), symbol_input_map)
    self.storage_offset = infer_ge_output_by_symbol_calculate(symbol_input_map, meta_out.storage_offset())
    as_strided = ge.AsStrided(bases_list[self.base_index],
                              self.size,
                              self.stride,
                              self.storage_offset)
    as_strided.set_meta(meta_out)
    self._as_strided = as_strided
    return as_strided


def _regenerate_as_strided_view(self, bases_list, symbol_input_map=None):
    fake_mode = detect_fake_mode(None)
    with fake_mode:
        meta_out = torch.as_strided(
            _get_meta_attr(bases_list[self.base_index]),
            _get_meta_attr(self.size),
            _get_meta_attr(self.stride),
            _get_meta_attr(self.storage_offset),
        )
    as_strided = ge.AsStrided(bases_list[self.base_index],
                              self.size,
                              self.stride,
                              self.storage_offset)
    as_strided.set_meta(meta_out)
    self._as_strided = as_strided
    return as_strided


def _regenerate_alias_view(self, bases_list, symbol_input_map=None):
    fake_mode = detect_fake_mode(None)
    with fake_mode:
        meta_out = torch.ops.aten.alias.default(_get_meta_attr(bases_list[self.base_index]))
    self.size = _sym_list_to_ge_tensor(list(meta_out.size()), symbol_input_map)
    self.stride = _sym_list_to_ge_tensor(list(meta_out.stride()), symbol_input_map)
    self.storage_offset = infer_ge_output_by_symbol_calculate(symbol_input_map, meta_out.storage_offset())
    as_strided = ge.AsStrided(bases_list[self.base_index],
                              self.size,
                              self.stride,
                              self.storage_offset)
    self._as_strided = as_strided
    return as_strided


def _sym_list_to_ge_tensor(sym_list, symbol_input_map):
    if all(isinstance(sym, int) for sym in sym_list):
        return sym_list
    npu_syms = []
    for sym in sym_list:
        npu_syms.append(infer_ge_output_by_symbol_calculate(symbol_input_map, sym))
    pack_tensor = ge.Pack(npu_syms, N=len(npu_syms), axis=0)
    pack_tensor.set_meta(sym_list)
    if all([is_host_data_tensor(sym_i) for sym_i in npu_syms]):
        pack_tensor.node.attr['_inputs_all_sym'].b = True

    # force unknown shape with ge.Pack when parse symlist
    return force_op_unknown_shape(pack_tensor)


def _regenerate_not_view(self, bases_list, symbol_input_map=None):
    return bases_list[self.base_index]


def conveter_auto_functionalize_v2(*args, **kwargs):
    from torch._higher_order_ops.auto_functionalize import get_mutable_args, read_view_information_from_args
    all_bases = kwargs.pop("_all_bases", [])
    symbol_input_map = kwargs.pop("symbol_input_map", {})
    _mutable_op = args[0]
    mutable_args_names, mutable_args_types = get_mutable_args(_mutable_op)
    args_view_info = read_view_information_from_args(
        mutable_args_names, mutable_args_types, kwargs, all_bases
    )

    all_bases_new = []
    for base_tensor in all_bases:
        base_tensor_copy = ge.TensorMove(base_tensor)
        base_tensor_copy.set_meta(base_tensor.meta)
        all_bases_new.append(base_tensor_copy)

    new_kwargs = dict(**kwargs)

    from torch._higher_order_ops.auto_functionalize import AsStridedViewInfo, SliceViewInfo, AliasViewInfo, NotView

    AsStridedViewInfo.regenerate_ge_view = _regenerate_as_strided_view
    AsStridedViewInfo.view_copy = _view_copy
    SliceViewInfo.regenerate_ge_view = _regenerate_slice_view
    SliceViewInfo.view_copy = _view_copy
    AliasViewInfo.regenerate_ge_view = _regenerate_alias_view
    AliasViewInfo.view_copy = _view_copy
    NotView.regenerate_ge_view = _regenerate_not_view
    NotView.view_copy = _not_view_copy

    for arg_name in mutable_args_names:
        if args_view_info[arg_name] is None:
            new_kwargs[arg_name] = None
        elif isinstance(args_view_info[arg_name], list):
            new_kwargs[arg_name] = []
            for i, elem in enumerate(args_view_info[arg_name]):
                if elem is None:
                    new_kwargs[arg_name].append(None)
                else:
                    view_info = args_view_info[arg_name][i]
                    new_kwargs[arg_name].append(
                        view_info.regenerate_ge_view(all_bases_new)
                    )
        else:
            new_kwargs[arg_name] = args_view_info[arg_name].regenerate_ge_view(
                all_bases_new, symbol_input_map
            )

    from .fx2ge_converter import get_or_auto_gen_converter 
    _mutable_op_converter = get_or_auto_gen_converter(_mutable_op)

    graph = get_default_ge_graph()
    num_ops = len(graph.op)

    out = _mutable_op_converter(**new_kwargs)
    
    infer_and_gen_sym_shape_silent(_mutable_op, [], new_kwargs, out, graph.op[num_ops:])  

    all_bases_new_update = []
    for view_info in args_view_info.values():
        if isinstance(out, (list, tuple)):
            all_bases_new_update.append(view_info.view_copy(all_bases_new, dependencies=out, out_op=out[0].node))
        else:
            all_bases_new_update.append(view_info.view_copy(all_bases_new, dependencies=[out], out_op=out.node))        

    if isinstance(out, tuple):
        return (*out, *all_bases_new_update)
    else:
        return (out, *all_bases_new_update)

