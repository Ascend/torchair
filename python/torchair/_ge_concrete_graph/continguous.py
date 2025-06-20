from torch.fx.experimental.symbolic_shapes import hint_int
from torchair.core._concrete_graph import ValuePack
from torchair.ge._ge_graph import Tensor, is_sym, DataType
from torchair._ge_concrete_graph.utils import is_host_data_tensor, force_op_unknown_shape
from . import ge_apis as ge


def is_contiguous(stride, shape):
    if len(stride) <= 1:
        return True
    if stride[-1] != 1:
        return False
    for i in range(len(shape) - 1):
        if hint_int(stride[i]) != hint_int(stride[i + 1]) * hint_int(shape[i + 1]):
            return False
    return True


def gen_contiguous_storagesize(shapes):
    if not isinstance(shapes, (tuple, list)):
        raise AssertionError(f"Invalid shape type:{type(shapes)} to generate contiguous stride.")
    
    if not shapes:
        return 1
    storage_size = 1
    for i in shapes:
        storage_size *= i
    return storage_size


def gen_contiguous_stride(shapes):
    if not isinstance(shapes, (tuple, list)):
        raise AssertionError(f"Invalid shape type:{type(shapes)} to generate contiguous stride.")

    if not shapes:
        return []
    strides = [1]
    for i in reversed(shapes[1:]):
        strides.insert(0, strides[0] * i)
    return strides


#获取动态shape，若遇到动态shape从view节点获取的mapsym中寻找，除此之外的动态shape目前不支持构造节点
def get_sym_node_from_graph(npu_input_src, symbol_dim, graph):
    mapsym = getattr(npu_input_src, "view_faketensor").mapsym
    srcshape = getattr(npu_input_src, "view_faketensor").srcshape
    if str(symbol_dim) in mapsym.keys():
        for op in reversed(graph.op):
            if op.name == mapsym[str(symbol_dim)].split(":")[0]:
                return Tensor(op, int(mapsym[str(symbol_dim)].split(":")[1]))
    elif symbol_dim in srcshape:
        for dim_idx, dim_val in enumerate(srcshape):
            if is_sym(dim_val) and dim_val == symbol_dim:
                return ge.Gather(ge.Shape(npu_input_src, dtype=DataType.DT_INT32), dim_idx)
    raise AssertionError("unsupported case for reshape.")


def _build_valid_stride(stride, shape):
    need_transpose = False
    simple_stride, simple_shape = [], []
    stride_index_dicts = {}
    for i, j in zip(stride, shape):
        if i != 0 and j != 1:
            simple_stride.append(i)
            simple_shape.append(j)
    last_stride = hint_int(simple_stride[0])
    for index, stride in enumerate(simple_stride):
        stride_index_dicts[index] = [simple_stride[index], simple_shape[index]]
        if last_stride < hint_int(stride):
            need_transpose = True
        last_stride = hint_int(stride)
    return stride_index_dicts, need_transpose


def _build_reshape_node(npu_input, npu_input_src, meta_size, graph):
    if all([not is_sym(meta_dim) for meta_dim in meta_size]):
        return ge.Reshape(npu_input, ge.Const(meta_size, dtype=DataType.DT_INT64))

    target_shape = []
    view_operator_map = getattr(graph, "view_operator_map", {})
    for meta_dim in meta_size:
        if not is_sym(meta_dim):
            target_shape.append(ge.Const(meta_dim, dtype=DataType.DT_INT64))
        else:
            if str(meta_dim) in view_operator_map.keys():
                target_shape.append(view_operator_map.get(str(meta_dim)))
            else:
                view_operator_map[str(meta_dim)] = get_sym_node_from_graph(npu_input_src, meta_dim, graph)
                target_shape.append(view_operator_map.get(str(meta_dim)))
    setattr(graph, "view_operator_map", view_operator_map)
    pack_tensor = ge.Pack(target_shape, N=len(target_shape), axis=0)
    if all([is_host_data_tensor(sym_i) for sym_i in target_shape]):
        pack_tensor.node.attr['_inputs_all_sym'].b = True
    return ge.Reshape(npu_input, force_op_unknown_shape(pack_tensor))


def _build_transpose_perm(stride_index_dicts):
    permidx = []
    src_trans_shape = []
    dst_trans_shape = []
    for indexes in stride_index_dicts.keys():
        permidx.append(int(indexes))
    for values in stride_index_dicts.values():
        src_trans_shape.append(values[1])
    permlist = [None] * len(permidx)
    for i, value in enumerate(permidx):
        permlist[value] = i
    for idx, _ in enumerate(permlist):
        dst_trans_shape.append(stride_index_dicts.get(idx)[1])
    return permlist, src_trans_shape, dst_trans_shape


def _optimize_non_contiguous(npu_input, meta_input, graph):
    npu_input_src = npu_input
    npu_shape = getattr(npu_input, "view_faketensor").srcshape
    meta_shape = list(meta_input.size())
    meta_stride = list(meta_input.stride())
    need_transpose = False

    #剔除shape为1的stride，并记录shape、stride与index的映射关系，同时判断是否存在transpose
    stride_index_dicts, need_transpose = _build_valid_stride(meta_stride, meta_shape)

    #存在transpose，则计算transpose的相关参数
    stride_index_dicts = dict(sorted(stride_index_dicts.items(), key=lambda \
        s: (hint_int(s[1][0]), hint_int(s[1][1])), reverse=True))
    permlist, src_trans_shape, dst_trans_shape = _build_transpose_perm(stride_index_dicts)

    if need_transpose:
        if src_trans_shape != npu_shape:
            npu_input = _build_reshape_node(npu_input, npu_input_src, src_trans_shape, graph)
        npu_input = ge.Transpose(npu_input, permlist)
        if dst_trans_shape != meta_shape:
            npu_input = _build_reshape_node(npu_input, npu_input_src, meta_shape, graph)
    else:
        npu_input = _build_reshape_node(npu_input, npu_input_src, meta_shape, graph)

    return npu_input


def optimize_view(npu_input, graph):
    if isinstance(npu_input, ValuePack):
        npu_input = npu_input.npu
    npu_input_src = npu_input

    view_operator_map = getattr(graph, "view_operator_map", {})
    #非Tensor的arg直接返回
    if not hasattr(npu_input, "view_faketensor"):
        if isinstance(npu_input, (list, tuple)):
            return [optimize_view(arg, graph) for arg in npu_input]
        return npu_input

    fake = getattr(npu_input, "view_faketensor")
    meta_real = npu_input.meta
    npu_shape = fake.srcshape
    npu_stride = gen_contiguous_stride(npu_shape)
    meta_input = fake.meta
    meta_shape = list(meta_input.size())
    meta_stride = list(meta_input.stride())

    #未经过view类操作的arg直接返回
    if meta_shape == npu_shape and meta_stride == npu_stride:
        return npu_input

    if (npu_input_src, meta_input) in view_operator_map.keys():
        return view_operator_map[(npu_input_src, meta_input)]

    #经过view类操作的arg判断是否为长度为1，若为连续则只经过Reshape；若为非连续则进入反推判断
    if is_contiguous(meta_shape, meta_stride):
        npu_input = _build_reshape_node(npu_input, npu_input, meta_shape, graph)
    else:
        npu_input = _optimize_non_contiguous(npu_input, meta_input, graph)

    view_operator_map[(npu_input_src, meta_input)] = npu_input
    setattr(graph, "view_operator_map", view_operator_map)
    npu_input.set_meta(meta_real)
    return npu_input
