import sympy
from torch.fx.experimental.symbolic_shapes import hint_int
from torchair.core._concrete_graph import ValuePack
from torchair._ge_concrete_graph.ge_graph import Tensor, is_sym, DataType
from torchair._ge_concrete_graph.utils import is_host_data_tensor
from . import ge_apis as ge


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
def get_sym_node_from_graph(npu_input, symbol_dim, graph):
    mapsym = getattr(npu_input, "view_faketensor").mapsym
    for op in graph.op:
        if op.name == mapsym[str(symbol_dim)].split(":")[0]:
            return Tensor(op, int(mapsym[str(symbol_dim)].split(":")[1]))
    raise AssertionError("unsupported case for reshape.")


def _build_valid_stride(stride, shape):
    simplestride, simpleshape = [], []
    strideindexdicts = {}
    for i, j in zip(stride, shape):
        simplestride.append(i)
        simpleshape.append(j)
    for index, stride in enumerate(simplestride):
        strideindexdicts[index] = [simplestride[index], simpleshape[index]]
    
    return strideindexdicts


def _build_reshape_node(npu_input, meta_size, graph):
    if all([not sympy.simplify(str(meta_dim)).has(sympy.Symbol) for meta_dim in meta_size]):
        return ge.Reshape(npu_input, ge.Const(meta_size, dtype=DataType.DT_INT64))

    target_shape = []
    for meta_dim in meta_size:
        if not sympy.simplify(str(meta_dim)).has(sympy.Symbol):
            target_shape.append(ge.Const(meta_dim, dtype=DataType.DT_INT64))
        else:
            target_shape.append(get_sym_node_from_graph(npu_input, meta_dim, graph))
    pack_tensor = ge.Pack(target_shape, N=len(target_shape), axis=0)
    if all([is_host_data_tensor(sym_i) for sym_i in target_shape]):
        pack_tensor.node.attr['_inputs_all_sym'].b = True
    return ge.Reshape(npu_input, pack_tensor)


def _build_transpose_perm(strideindexdicts):
    permidx = []
    srctransshape = []
    for indexes in strideindexdicts.keys():
        permidx.append(int(indexes))
    for values in strideindexdicts.values():
        srctransshape.append(values[1])
    permlist = [None] * len(permidx)
    for i, value in enumerate(permidx):
        permlist[value] = i
    return permlist, srctransshape


def _optimize_non_contiguous(npu_input, meta_input, graph):
    npu_shape = getattr(npu_input, "view_faketensor").srcshape
    meta_shape = list(meta_input.size())
    meta_stride = list(meta_input.stride())

    #剔除shape为1的stride，并记录shape、stride与index的映射关系，同时判断是否存在transpose
    strideindexdicts = _build_valid_stride(meta_stride, meta_shape)

    #存在transpose，则计算transpose的相关参数
    strideindexdicts = dict(sorted(strideindexdicts.items(), key=lambda \
        s:(hint_int(s[1][0]), hint_int(s[1][1])), reverse=True))
    permlist, srctransshape = _build_transpose_perm(strideindexdicts)

    if srctransshape != npu_shape:
        npu_input = _build_reshape_node(npu_input, srctransshape, graph)
    npu_input = ge.Transpose(npu_input, permlist)

    return npu_input


def optimize_view(npu_input, graph):
    if isinstance(npu_input, ValuePack):
        npu_input = npu_input.npu
    
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

    #经过view类操作的arg判断是否为长度为1，若为连续则只经过Reshape；若为非连续则进入反推判断
    if len(meta_shape) == 1:
        npu_input = _build_reshape_node(npu_input, meta_shape, graph)
    else:
        npu_input = _optimize_non_contiguous(npu_input, meta_input, graph)
    npu_input.set_meta(meta_real)
    return npu_input