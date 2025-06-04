from typing import Any, Dict, List, Tuple, Union, Callable, Optional
from torchair._ge_concrete_graph.ge_ir_pb2 import GraphDef, OpDef, TensorDescriptor, TensorDef
from torchair.ge._ge_graph import get_default_ge_graph, next_unique_name
from torchair.ge._ge_graph import auto_convert_to_tensor
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair.ge._ge_graph import compat_as_bytes, compat_as_bytes_list
from torchair.ge._ge_graph import trans_to_list_list_int, trans_to_list_list_float
from torchair._ge_concrete_graph import auto_generated_ge_raw_ops as raw_ops


# Auto infer num outputs for IR IdentityN
@auto_convert_to_tensor([True], [False])
def IdentityN(x: List[Tensor], *, dependencies=[], node_name=None):
    """REG_OP(IdentityN)\n
  .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))\n
  .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))\n
  """

    size_of_y = len(x)
    return raw_ops._IdentityN(x, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


# Auto infer num outputs for IR GroupedMatmul
@auto_convert_to_tensor([True, True, True, True, True, True, True, False, False],
                        [False, False, False, False, False, False, False, True, True])
def GroupedMatmul(x: List[Tensor], weight: List[Tensor], bias: List[Tensor], scale: List[Tensor], offset: List[Tensor],
                  antiquant_scale: List[Tensor], antiquant_offset: List[Tensor], group_list: Optional[Tensor],
                  per_token_scale: Optional[Tensor], *, split_item: int = 0, dtype: int = 0,
                  transpose_weight: bool = False, transpose_x: bool = False,
                  group_type: int = -1, group_list_type: int = 0, act_type: int = 0,
                  tuning_config: Optional[List[int]] = None):
    """REG_OP(GroupedMatmul)\n
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT}))\n
    .DYNAMIC_INPUT(weight, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT, DT_INT4}))\n
    .DYNAMIC_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))\n
    .DYNAMIC_INPUT(scale, TensorType({DT_UINT64, DT_BF16, DT_FLOAT32}))\n
    .DYNAMIC_INPUT(offset, TensorType({DT_FLOAT32}))\n
    .DYNAMIC_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))\n
    .DYNAMIC_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))\n
    .OPTIONAL_INPUT(group_list, TensorType({DT_INT64}))\n
    .OPTIONAL_INPUT(per_token_scale, TensorType({DT_FLOAT}))\n
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT}))\n
    .ATTR(split_item, Int, 0)\n
    .ATTR(dtype, Int, 0)\n
    .ATTR(transpose_weight, Bool, false)\n
    .ATTR(transpose_x, Bool, false)\n
    .ATTR(group_type, Int, -1)\n
    .ATTR(group_list_type, Int, 0)\n
    .ATTR(act_type, Int, 0)\n
    .ATTR(tuning_config, ListInt, [0])\n
    """
    if tuning_config is None:
        tuning_config = [0]

    size_of_y = 0
    if split_item == 0 or split_item == 1:
        if group_list is not None:
            size_of_y = group_list.symsize[0]
        else:
            size_of_y = len(x)
    elif split_item == 2 or split_item == 3:
        size_of_y = 1

    return raw_ops._GroupedMatmul(x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, group_list,
                                  per_token_scale, size_of_y=size_of_y, split_item=split_item, dtype=dtype,
                                  transpose_weight=transpose_weight, transpose_x=transpose_x,
                                  group_type=group_type, group_list_type=group_list_type,
                                  act_type=act_type, tuning_config=tuning_config)


# Auto infer num outputs for IR ShapeN
@auto_convert_to_tensor([True], [False])
def ShapeN(x: List[Tensor], *, dtype: int = 3, dependencies=[], node_name=None):
    """REG_OP(ShapeN)\n
  .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))\n
  .DYNAMIC_OUTPUT(y, TensorType({DT_INT32, DT_INT64}))\n
  .ATTR(dtype, Int, DT_INT32)\n
  """

    size_of_y = len(x)
    return raw_ops._ShapeN(x, size_of_y=size_of_y, dtype=dtype, dependencies=dependencies, node_name=node_name)


# Auto infer num outputs for IR IdentityN
@auto_convert_to_tensor([True, False, False, False], [False, False, False, True])
def ScatterList(var: List[Tensor], indice: Tensor, updates: Tensor, mask: Optional[Tensor], *,
    reduce: str = "update", axis: int = -2, dependencies=[], node_name=None):
    """REG_OP(ScatterList)\n
    .DYNAMIC_INPUT(var, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                                    DT_UINT16, DT_UINT32, DT_UINT64}))\n
    .INPUT(indice, TensorType::IndexNumberType())\n
    .INPUT(updates, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                                DT_UINT16, DT_UINT32, DT_UINT64}))\n
    .OPTIONAL_INPUT(mask, TensorType({DT_UINT8}))\n
    .DYNAMIC_OUTPUT(var, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                                      DT_UINT16, DT_UINT32, DT_UINT64}))\n
    .ATTR(reduce, String, "update")\n
      .ATTR(axis, Int, -2)\n
    """
    size_of_var = len(var)
    return raw_ops._ScatterList(var, indice, updates, mask, size_of_var=size_of_var,
                                 reduce=reduce, axis=axis, dependencies=dependencies, node_name=node_name)


# Auto infer num outputs for IR Copy
@auto_convert_to_tensor([False], [False])
def Copy(x: Tensor, *, N: int, dependencies=[], node_name=None):
    """REG_OP(Copy)\n
  .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64}))\n
  .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64}))\n
  .REQUIRED_ATTR(N, Int)\n
  """

    size_of_y = N
    return raw_ops._Copy(x, size_of_y=size_of_y, N=N, dependencies=dependencies, node_name=node_name)


# Auto infer num outputs for IR Batch
@auto_convert_to_tensor([True], [False])
def Batch(x_tensors: List[Tensor], *, num_batch_threads: int, max_batch_size: int, batch_timeout_micros: int, grad_timeout_micros: int, max_enqueued_batches: int = 10, allowed_batch_sizes: List[int] = [], container: str = "", shared_name: str = "", batching_queue: str = "", dependencies=[], node_name=None):
    """REG_OP(Batch)\n
  .DYNAMIC_INPUT(x_tensors, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE}))\n
  .OUTPUT(y_index, TensorType({ DT_INT64 }))\n
  .OUTPUT(y_id, TensorType({ DT_INT64 }))\n
  .DYNAMIC_OUTPUT(y_tensors, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_INT64, DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_BOOL}))\n
  .REQUIRED_ATTR(num_batch_threads, Int)\n
  .REQUIRED_ATTR(max_batch_size, Int)\n
  .ATTR(max_enqueued_batches, Int, 10)\n
  .REQUIRED_ATTR(batch_timeout_micros, Int)\n
  .ATTR(allowed_batch_sizes, ListInt, {})\n
  .REQUIRED_ATTR(grad_timeout_micros, Int)\n
  .ATTR(container, String, "")\n
  .ATTR(shared_name, String, "")\n
  .ATTR(batching_queue, String, "")\n
  """

    size_of_y_tensors = 0
    raise RuntimeError("Fix this by determine num outputs of Batch")


# Auto infer num outputs for IR BoostedTreesBucketize
@auto_convert_to_tensor([True, True], [False, False])
def BoostedTreesBucketize(float_values: List[Tensor], bucket_boundaries: List[Tensor], *, num_features: int, dependencies=[], node_name=None):
    """REG_OP(BoostedTreesBucketize)\n
  .DYNAMIC_INPUT(float_values, TensorType({DT_FLOAT}))\n
  .DYNAMIC_INPUT(bucket_boundaries, TensorType({DT_FLOAT}))\n
  .DYNAMIC_OUTPUT(y, TensorType({DT_INT32}))\n
  .REQUIRED_ATTR(num_features, Int)\n
  """

    size_of_y = 0
    raise RuntimeError("Fix this by determine num outputs of BoostedTreesBucketize")


# Auto infer num outputs for IR SwitchN
@auto_convert_to_tensor([False, False], [False, False])
def SwitchN(data: Tensor, pred_value: Tensor, *, dependencies=[], node_name=None):
    """REG_OP(SwitchN)\n
  .INPUT(data, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_BOOL}))\n
  .INPUT(pred_value, TensorType({DT_INT64}))\n
  .DYNAMIC_OUTPUT(output, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_BOOL}))\n
  """

    size_of_output = 0
    raise RuntimeError("Fix this by determine num outputs of BoostedTreesBucketize")


# Auto infer num outputs for IR CTCBeamSearchDecoder
@auto_convert_to_tensor([False, False], [False, False])
def CTCBeamSearchDecoder(inputs: Tensor, sequence_length: Tensor, *, beam_width: int, top_paths: int, merge_repeated: bool = True, dependencies=[], node_name=None):
    """REG_OP(CTCBeamSearchDecoder)\n
  .INPUT(inputs, TensorType({DT_FLOAT, DT_DOUBLE}))\n
  .INPUT(sequence_length, TensorType({DT_INT32}))\n
  .REQUIRED_ATTR(beam_width, Int)\n
  .REQUIRED_ATTR(top_paths, Int)\n
  .ATTR(merge_repeated, Bool, true)\n
  .DYNAMIC_OUTPUT(decoded_indices, TensorType({DT_INT64}))\n
  .DYNAMIC_OUTPUT(decoded_values, TensorType({DT_INT64}))\n
  .DYNAMIC_OUTPUT(decoded_shape, TensorType({DT_INT64}))\n
  .OUTPUT(log_probability, TensorType({DT_FLOAT, DT_DOUBLE}))\n
  """

    size_of_decoded_indices = 0
    size_of_decoded_values = 0
    size_of_decoded_shape = 0
    raise RuntimeError("Fix this by determine num outputs of CTCBeamSearchDecoder")


# Auto infer num outputs for IR QueueDequeue
@auto_convert_to_tensor([False], [False])
def QueueDequeue(handle: Tensor, *, component_types: List[int], timeout_ms: int = -1, dependencies=[], node_name=None):
    """REG_OP(QueueDequeue)\n
  .INPUT(handle, TensorType({DT_RESOURCE}))\n
  .DYNAMIC_OUTPUT(components, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE, DT_RESOURCE, DT_STRING, DT_COMPLEX64, DT_COMPLEX128, DT_QINT16, DT_QUINT16, DT_QINT8, DT_QUINT8, DT_QINT32}))\n
  .ATTR(timeout_ms, Int, -1)\n
  .REQUIRED_ATTR(component_types, ListType)\n
  """

    size_of_components = len(component_types)
    return raw_ops._QueueDequeue(handle, size_of_components=size_of_components, component_types=component_types, timeout_ms=timeout_ms, dependencies=dependencies, node_name=node_name)


# Auto infer num outputs for IR QueueDequeueMany
@auto_convert_to_tensor([False, False], [False, False])
def QueueDequeueMany(handle: Tensor, n: Tensor, *, component_types: List[int], timeout_ms: int = -1, dependencies=[], node_name=None):
    """REG_OP(QueueDequeueMany)\n
  .INPUT(handle, TensorType({DT_RESOURCE}))\n
  .INPUT(n, TensorType({DT_INT32}))\n
  .DYNAMIC_OUTPUT(components, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE, DT_RESOURCE, DT_STRING, DT_COMPLEX64, DT_COMPLEX128, DT_QINT16, DT_QUINT16, DT_QINT8, DT_QUINT8, DT_QINT32}))\n
  .ATTR(timeout_ms, Int, -1)\n
  .REQUIRED_ATTR(component_types, ListType)\n
  """

    size_of_components = len(component_types)
    return raw_ops._QueueDequeueMany(handle, n, size_of_components=size_of_components, component_types=component_types, timeout_ms=timeout_ms, dependencies=dependencies, node_name=node_name)


# Auto infer num outputs for IR QueueDequeueUpTo
@auto_convert_to_tensor([False, False], [False, False])
def QueueDequeueUpTo(handle: Tensor, n: Tensor, *, component_types: List[int], timeout_ms: int = -1, dependencies=[], node_name=None):
    """REG_OP(QueueDequeueUpTo)\n
  .INPUT(handle, TensorType({DT_RESOURCE}))\n
  .INPUT(n, TensorType({DT_INT32}))\n
  .DYNAMIC_OUTPUT(components, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE, DT_RESOURCE, DT_STRING, DT_COMPLEX64, DT_COMPLEX128, DT_QINT16, DT_QUINT16, DT_QINT8, DT_QUINT8, DT_QINT32}))\n
  .ATTR(timeout_ms, Int, -1)\n
  .REQUIRED_ATTR(component_types, ListType)\n
  """

    size_of_components = len(component_types)
    return raw_ops._QueueDequeueUpTo(handle, n, size_of_components=size_of_components, component_types=component_types, timeout_ms=timeout_ms, dependencies=dependencies, node_name=node_name)


# Auto infer num outputs for IR StagePeek
@auto_convert_to_tensor([False], [False])
def StagePeek(index: Tensor, *, dtypes: List[int], capacity: int = 0, memory_limit: int = 0, container: str = "", shared_name: str = "", dependencies=[], node_name=None):
    """REG_OP(StagePeek)\n
  .INPUT(index, TensorType({DT_INT32}))\n
  .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, DT_UINT32, DT_UINT64}))\n
  .ATTR(capacity, Int, 0)\n
  .ATTR(memory_limit, Int, 0)\n
  .ATTR(container, String, "")\n
  .ATTR(shared_name, String, "")\n
  .ATTR(dtypes, ListType, {})\n
  """

    size_of_y = len(dtypes)
    return raw_ops._StagePeek(index, size_of_y=size_of_y, capacity=capacity, memory_limit=memory_limit, container=container, shared_name=shared_name, dtypes=dtypes, dependencies=dependencies, node_name=node_name)


# Auto infer num outputs for IR DynamicPartition
@auto_convert_to_tensor([False, False], [False, False])
def DynamicPartition(x: Tensor, partitions: Tensor, *, num_partitions: int = 1, dependencies=[], node_name=None):
    """REG_OP(DynamicPartition)\n
  .INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_INT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128, DT_RESOURCE, DT_STRING}))\n
  .INPUT(partitions, TensorType({DT_INT32}))\n
  .DYNAMIC_OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_INT64, DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128, DT_RESOURCE, DT_STRING}))\n
  .ATTR(num_partitions, Int, 1)\n
  """

    size_of_y = num_partitions
    return raw_ops._DynamicPartition(x, partitions, size_of_y=size_of_y, num_partitions=num_partitions, dependencies=dependencies, node_name=node_name)


# Auto infer num outputs for IR Unstage
@auto_convert_to_tensor([], [])
def Unstage(*, dtypes: List[int], capacity: int = 0, memory_limit: int = 0, container: str = "", shared_name: str = "", dependencies=[], node_name=None):
    """REG_OP(Unstage)\n
  .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, DT_UINT32, DT_UINT64}))\n
  .ATTR(capacity, Int, 0)\n
  .ATTR(memory_limit, Int, 0)\n
  .ATTR(container, String, "")\n
  .ATTR(shared_name, String, "")\n
  .REQUIRED_ATTR(dtypes, ListType)\n
  """

    size_of_y = len(dtypes)
    return raw_ops._Unstage(size_of_y=size_of_y, dtypes=dtypes, capacity=capacity, memory_limit=memory_limit, container=container, shared_name=shared_name, dependencies=dependencies, node_name=node_name)


# Auto infer num outputs for IR MapUnstage
@auto_convert_to_tensor([False, False], [False, False])
def MapUnstage(key: Tensor, indices: Tensor, *, dtypes: List[int], capacity: int = 0, memory_limit: int = 0, container: str = "", shared_name: str = "", dependencies=[], node_name=None):
    """REG_OP(MapUnstage)\n
  .INPUT(key, TensorType({DT_INT64}))\n
  .INPUT(indices, TensorType({DT_INT32}))\n
  .DYNAMIC_OUTPUT(values, TensorType({ DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, DT_UINT32, DT_UINT64, DT_RESOURCE, DT_STRING, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32 }))\n
  .ATTR(capacity, Int, 0)\n
  .ATTR(memory_limit, Int, 0)\n
  .ATTR(dtypes, ListType, {})\n
  .ATTR(container, String, "")\n
  .ATTR(shared_name, String, "")\n
  """

    size_of_values = len(dtypes)
    return raw_ops._MapUnstage(key, indices, size_of_values=size_of_values, capacity=capacity, memory_limit=memory_limit, dtypes=dtypes, container=container, shared_name=shared_name, dependencies=dependencies, node_name=node_name)


# Auto infer num outputs for IR MapUnstageNoKey
@auto_convert_to_tensor([False], [False])
def MapUnstageNoKey(indices: Tensor, *, capacity: int = 0, memory_limit: int = 0, dtypes: List[int] = [], container: str = "", shared_name: str = "", dependencies=[], node_name=None):
    """REG_OP(MapUnstageNoKey)\n
  .INPUT(indices, TensorType({DT_INT32}))\n
  .OUTPUT(key, TensorType({DT_INT64}))\n
  .DYNAMIC_OUTPUT(values, TensorType({ DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, DT_UINT32, DT_UINT64, DT_RESOURCE, DT_STRING, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32 }))\n
  .ATTR(capacity, Int, 0)\n
  .ATTR(memory_limit, Int, 0)\n
  .ATTR(dtypes, ListType, {})\n
  .ATTR(container, String, "")\n
  .ATTR(shared_name, String, "")\n
  """

    size_of_values = len(dtypes)
    return raw_ops._MapUnstageNoKey(indices, size_of_values=size_of_values, capacity=capacity, memory_limit=memory_limit, dtypes=dtypes, container=container, shared_name=shared_name, dependencies=dependencies, node_name=node_name)


# Auto infer num outputs for IR MapPeek
@auto_convert_to_tensor([False, False], [False, False])
def MapPeek(key: Tensor, indices: Tensor, *, capacity: int = 0, memory_limit: int = 0, dtypes: List[int] = [], container: str = "", shared_name: str = "", dependencies=[], node_name=None):
    """REG_OP(MapPeek)\n
  .INPUT(key, TensorType({DT_INT64}))\n
  .INPUT(indices, TensorType({DT_INT32}))\n
  .DYNAMIC_OUTPUT(values, TensorType({ DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, DT_UINT32, DT_UINT64, DT_RESOURCE, DT_STRING, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32 }))\n
  .ATTR(capacity, Int, 0)\n
  .ATTR(memory_limit, Int, 0)\n
  .ATTR(dtypes, ListType, {})\n
  .ATTR(container, String, "")\n
  .ATTR(shared_name, String, "")\n
  """

    size_of_values = len(dtypes)
    return raw_ops._MapPeek(key, indices, size_of_values=size_of_values, capacity=capacity, memory_limit=memory_limit, dtypes=dtypes, container=container, shared_name=shared_name, dependencies=dependencies, node_name=node_name)


# Auto infer num outputs for IR OrderedMapPeek
@auto_convert_to_tensor([False, False], [False, False])
def OrderedMapPeek(key: Tensor, indices: Tensor, *, capacity: int = 0, memory_limit: int = 0, dtypes: List[int] = [], container: str = "", shared_name: str = "", dependencies=[], node_name=None):
    """REG_OP(OrderedMapPeek)\n
  .INPUT(key, TensorType({DT_INT64}))\n
  .INPUT(indices, TensorType({DT_INT32}))\n
  .DYNAMIC_OUTPUT(values, TensorType({ DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, DT_UINT32, DT_UINT64, DT_RESOURCE, DT_STRING, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32 }))\n
  .ATTR(capacity, Int, 0)\n
  .ATTR(memory_limit, Int, 0)\n
  .ATTR(dtypes, ListType, {})\n
  .ATTR(container, String, "")\n
  .ATTR(shared_name, String, "")\n
  """

    size_of_values = len(dtypes)
    return raw_ops._OrderedMapPeek(key, indices, size_of_values=size_of_values, capacity=capacity, memory_limit=memory_limit, dtypes=dtypes, container=container, shared_name=shared_name, dependencies=dependencies, node_name=node_name)


# Auto infer num outputs for IR OrderedMapUnstageNoKey
@auto_convert_to_tensor([False], [False])
def OrderedMapUnstageNoKey(indices: Tensor, *, capacity: int = 0, memory_limit: int = 0, dtypes: List[int] = [], container: str = "", shared_name: str = "", dependencies=[], node_name=None):
    """REG_OP(OrderedMapUnstageNoKey)\n
  .INPUT(indices, TensorType({DT_INT32}))\n
  .OUTPUT(key, TensorType({DT_INT64}))\n
  .DYNAMIC_OUTPUT(values, TensorType({ DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, DT_UINT32, DT_UINT64, DT_RESOURCE, DT_STRING, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32 }))\n
  .ATTR(capacity, Int, 0)\n
  .ATTR(memory_limit, Int, 0)\n
  .ATTR(dtypes, ListType, {})\n
  .ATTR(container, String, "")\n
  .ATTR(shared_name, String, "")\n
  """

    size_of_values = len(dtypes)
    return raw_ops._OrderedMapUnstageNoKey(indices, size_of_values=size_of_values, capacity=capacity, memory_limit=memory_limit, dtypes=dtypes, container=container, shared_name=shared_name, dependencies=dependencies, node_name=node_name)


# Auto infer num outputs for IR OrderedMapUnstage
@auto_convert_to_tensor([False, False], [False, False])
def OrderedMapUnstage(key: Tensor, indices: Tensor, *, capacity: int = 0, memory_limit: int = 0, dtypes: List[int] = [], container: str = "", shared_name: str = "", dependencies=[], node_name=None):
    """REG_OP(OrderedMapUnstage)\n
  .INPUT(key, TensorType({DT_INT64}))\n
  .INPUT(indices, TensorType({DT_INT32}))\n
  .DYNAMIC_OUTPUT(values, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_INT64, DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_BOOL, DT_UINT32, DT_UINT64}))\n
  .ATTR(capacity, Int, 0)\n
  .ATTR(memory_limit, Int, 0)\n
  .ATTR(dtypes, ListType, {})\n
  .ATTR(container, String, "")\n
  .ATTR(shared_name, String, "")\n
  """

    size_of_values = len(dtypes)
    return raw_ops._OrderedMapUnstage(key, indices, size_of_values=size_of_values, capacity=capacity, memory_limit=memory_limit, dtypes=dtypes, container=container, shared_name=shared_name, dependencies=dependencies, node_name=node_name)


# Auto infer num outputs for IR BarrierTakeMany
@auto_convert_to_tensor([False, False], [False, False])
def BarrierTakeMany(handle: Tensor, num_elements: Tensor, *, component_types: List[int], allow_small_batch: bool = False, wait_for_incomplete: bool = False, timeout_ms: int = -1, dependencies=[], node_name=None):
    """REG_OP(BarrierTakeMany)\n
  .INPUT(handle, TensorType({DT_STRING_REF}))\n
  .INPUT(num_elements, TensorType(DT_INT32))\n
  .OUTPUT(indices, TensorType({DT_INT64}))\n
  .OUTPUT(keys, TensorType({DT_STRING}))\n
  .DYNAMIC_OUTPUT(values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128, DT_RESOURCE, DT_STRING}))\n
  .REQUIRED_ATTR(component_types, ListType)\n
  .ATTR(allow_small_batch, Bool, false)\n
  .ATTR(wait_for_incomplete, Bool, false)\n
  .ATTR(timeout_ms, Int, -1)\n
  """

    size_of_values = len(component_types)
    return raw_ops._BarrierTakeMany(handle, num_elements, size_of_values=size_of_values, component_types=component_types, allow_small_batch=allow_small_batch, wait_for_incomplete=wait_for_incomplete, timeout_ms=timeout_ms, dependencies=dependencies, node_name=node_name)


# Auto infer num outputs for IR IteratorGetNext
@auto_convert_to_tensor([False], [False])
def IteratorGetNext(x: Tensor, *, output_types: List[int], output_shapes: List[List[int]] = [[], []], _kernel: str = "dp", dependencies=[], node_name=None):
    """REG_OP(IteratorGetNext)\n
  .INPUT(x, TensorType::ALL())\n
  .DYNAMIC_OUTPUT(y, TensorType::ALL())\n
  .ATTR(output_types, ListInt, {})\n
  .ATTR(output_shapes, ListListInt, {{},{}})\n
  .ATTR(output_num, Int, 1)\n
  .ATTR(_kernel, String, "dp")\n
  """

    size_of_y = len(output_types)
    output_num = size_of_y
    return raw_ops._IteratorGetNext(x, size_of_y=size_of_y, output_types=output_types, output_shapes=output_shapes, output_num=output_num, _kernel=_kernel, dependencies=dependencies, node_name=node_name)


# Auto infer num outputs for IR HcomBroadcast
@auto_convert_to_tensor([True], [False])
def HcomBroadcast(x: List[Tensor], *, root_rank: int, group: str, fusion: int = 0, fusion_id: int = -1, dependencies=[], node_name=None):
    """REG_OP(HcomBroadcast)\n
  .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT16, DT_DOUBLE, DT_BF16}))\n
  .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT16, DT_DOUBLE, DT_BF16}))\n
  .REQUIRED_ATTR(root_rank, Int)\n
  .REQUIRED_ATTR(group, String)\n
  .ATTR(fusion, Int, 0)\n
  .ATTR(fusion_id, Int, -1)\n
  """

    size_of_y = len(x)
    return raw_ops._HcomBroadcast(x, size_of_y=size_of_y, root_rank=root_rank, group=group, fusion=fusion, fusion_id=fusion_id, dependencies=dependencies, node_name=node_name)


# Auto infer num outputs for IR AssistHelp
@auto_convert_to_tensor([True], [False])
def AssistHelp(x: List[Tensor], *, func_name: str, dependencies=[], node_name=None):
    """REG_OP(AssistHelp)\n
  .DYNAMIC_INPUT(x, TensorType({ DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE }))\n
  .DYNAMIC_OUTPUT(y, TensorType({ DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))\n
  .REQUIRED_ATTR(func_name, String)\n
  """

    size_of_y = len(x)
    return raw_ops._AssistHelp(x, size_of_y=size_of_y, func_name=func_name, dependencies=dependencies, node_name=node_name)


# Auto infer num outputs for IR GetNext
@auto_convert_to_tensor([], [])
def GetNext(*, output_types: List[int], output_shapes: List[List[int]] = [], channel_name: str = "", dependencies=[], node_name=None):
    """REG_OP(GetNext)\n
  .DYNAMIC_OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL}))\n
  .ATTR(output_types, ListInt, {})\n
  .ATTR(output_shapes, ListListInt, {})\n
  .ATTR(output_num, Int, 1)\n
  .ATTR(channel_name, String, "")\n
  """

    size_of_y = len(output_types)
    output_num = size_of_y
    return raw_ops._GetNext(size_of_y=size_of_y, output_types=output_types, output_shapes=output_shapes, output_num=output_num, channel_name=channel_name, dependencies=dependencies, node_name=node_name)


# Auto infer num outputs for IR DynamicRNNGrad
@auto_convert_to_tensor([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True])
def DynamicRNNGrad(x: Tensor, w: Tensor, b: Tensor, y: Tensor, init_h: Tensor, init_c: Tensor, h: Tensor, c: Tensor, dy: Tensor, dh: Tensor, dc: Tensor, i: Tensor, j: Tensor, f: Tensor, o: Tensor, tanhct: Optional[Tensor], seq_length: Optional[Tensor], mask: Optional[Tensor], wci: Optional[Tensor], wcf: Optional[Tensor], wco: Optional[Tensor], *, cell_type: str = "LSTM", direction: str = "UNIDIRECTIONAL", cell_depth: int = 0, use_peephole: bool = False, keep_prob: float = -1.000000, cell_clip: float = -1.000000, num_proj: int = 0, time_major: bool = True, forget_bias: float = 0.000000, dependencies=[], node_name=None):
    """REG_OP(DynamicRNNGrad)\n
  .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))\n
  .INPUT(w, TensorType({DT_FLOAT16, DT_FLOAT}))\n
  .INPUT(b, TensorType({DT_FLOAT16, DT_FLOAT}))\n
  .INPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))\n
  .INPUT(init_h, TensorType({DT_FLOAT16, DT_FLOAT}))\n
  .INPUT(init_c, TensorType({DT_FLOAT16, DT_FLOAT}))\n
  .INPUT(h, TensorType({DT_FLOAT16, DT_FLOAT}))\n
  .INPUT(c, TensorType({DT_FLOAT16, DT_FLOAT}))\n
  .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT}))\n
  .INPUT(dh, TensorType({DT_FLOAT16, DT_FLOAT}))\n
  .INPUT(dc, TensorType({DT_FLOAT16, DT_FLOAT}))\n
  .INPUT(i, TensorType({DT_FLOAT16, DT_FLOAT}))\n
  .INPUT(j, TensorType({DT_FLOAT16, DT_FLOAT}))\n
  .INPUT(f, TensorType({DT_FLOAT16, DT_FLOAT}))\n
  .INPUT(o, TensorType({DT_FLOAT16, DT_FLOAT}))\n
  .OPTIONAL_INPUT(tanhct, TensorType({DT_FLOAT16, DT_FLOAT}))\n
  .OPTIONAL_INPUT(seq_length, TensorType({DT_INT32}))\n
  .OPTIONAL_INPUT(mask, TensorType({DT_UINT8}))\n
  .OPTIONAL_INPUT(wci, TensorType({DT_FLOAT16, DT_FLOAT}))\n
  .OPTIONAL_INPUT(wcf, TensorType({DT_FLOAT16, DT_FLOAT}))\n
  .OPTIONAL_INPUT(wco, TensorType({DT_FLOAT16, DT_FLOAT}))\n
  .OUTPUT(dw, TensorType({DT_FLOAT16, DT_FLOAT}))\n
  .OUTPUT(db, TensorType({DT_FLOAT16, DT_FLOAT}))\n
  .OUTPUT(dx, TensorType({DT_FLOAT16, DT_FLOAT}))\n
  .OUTPUT(dh_prev, TensorType({DT_FLOAT16, DT_FLOAT}))\n
  .OUTPUT(dc_prev, TensorType({DT_FLOAT16, DT_FLOAT}))\n
  .DYNAMIC_OUTPUT(dwci, TensorType({DT_FLOAT16, DT_FLOAT}))\n
  .DYNAMIC_OUTPUT(dwcf, TensorType({DT_FLOAT16, DT_FLOAT}))\n
  .DYNAMIC_OUTPUT(dwco, TensorType({DT_FLOAT16, DT_FLOAT}))\n
  .ATTR(cell_type, String, "LSTM")\n
  .ATTR(direction, String, "UNIDIRECTIONAL")\n
  .ATTR(cell_depth, Int, 0)\n
  .ATTR(use_peephole, Bool, false)\n
  .ATTR(keep_prob, Float, -1.0)\n
  .ATTR(cell_clip, Float, -1.0)\n
  .ATTR(num_proj, Int, 0)\n
  .ATTR(time_major, Bool, true)\n
  .ATTR(forget_bias, Float, 0.0)\n
  """

    size_of_dwci = len(wci)
    size_of_dwcf = len(wcf)
    size_of_dwco = len(wco)
    return raw_ops._DynamicRNNGrad(x, w, b, y, init_h, init_c, h, c, dy, dh, dc, i, j, f, o, tanhct, seq_length, mask, wci, wcf, wco, size_of_dwci=size_of_dwci, size_of_dwcf=size_of_dwcf, size_of_dwco=size_of_dwco, cell_type=cell_type, direction=direction, cell_depth=cell_depth, use_peephole=use_peephole, keep_prob=keep_prob, cell_clip=cell_clip, num_proj=num_proj, time_major=time_major, forget_bias=forget_bias, dependencies=dependencies, node_name=node_name)


# Auto infer num outputs for IR SparseSplit
@auto_convert_to_tensor([False, False, False, False], [False, False, False, False])
def SparseSplit(split_dim: Tensor, indices: Tensor, values: Tensor, shape: Tensor, *, num_split: int = 1, dependencies=[], node_name=None):
    """REG_OP(SparseSplit)\n
  .INPUT(split_dim, TensorType({DT_INT64}))\n
  .INPUT(indices, TensorType({DT_INT64}))\n
  .INPUT(values, TensorType({DT_INT64, DT_INT32, DT_UINT16, DT_INT16, DT_UINT8, DT_INT8, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128, DT_BOOL, DT_STRING, DT_RESOURCE}))\n
  .INPUT(shape, TensorType({DT_INT64}))\n
  .DYNAMIC_OUTPUT(y_indices, TensorType({DT_INT64}))\n
  .DYNAMIC_OUTPUT(y_values, TensorType({DT_INT64, DT_INT32, DT_UINT16, DT_INT16, DT_UINT8, DT_INT8, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128, DT_BOOL, DT_STRING, DT_RESOURCE}))\n
  .DYNAMIC_OUTPUT(y_shape, TensorType({DT_INT64}))\n
  .ATTR(num_split, Int, 1)\n
  """

    size_of_y_indices = num_split
    size_of_y_values = num_split
    size_of_y_shape = num_split
    return raw_ops._SparseSplit(split_dim, indices, values, shape, size_of_y_indices=size_of_y_indices, size_of_y_values=size_of_y_values, size_of_y_shape=size_of_y_shape, num_split=num_split, dependencies=dependencies, node_name=node_name)


# Auto infer num outputs for IR Split
@auto_convert_to_tensor([False, False], [False, False])
def Split(split_dim: Tensor, x: Tensor, *, num_split: int, dependencies=[], node_name=None):
    """REG_OP(Split)\n
  .INPUT(split_dim, TensorType({DT_INT32}))\n
  .INPUT(x, TensorType::BasicType())\n
  .DYNAMIC_OUTPUT(y, TensorType::BasicType())\n
  .REQUIRED_ATTR(num_split, Int)\n
  """

    size_of_y = num_split
    return raw_ops._Split(split_dim, x, size_of_y=size_of_y, num_split=num_split, dependencies=dependencies, node_name=node_name)


# Auto infer num outputs for IR SplitD
@auto_convert_to_tensor([False], [False])
def SplitD(x: Tensor, *, split_dim: int, num_split: int, dependencies=[], node_name=None):
    """REG_OP(SplitD)\n
  .INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT, DT_FLOAT16}))\n
  .DYNAMIC_OUTPUT(y, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT, DT_FLOAT16}))\n
  .REQUIRED_ATTR(split_dim, Int)\n
  .REQUIRED_ATTR(num_split, Int)\n
  """

    size_of_y = num_split
    return raw_ops._SplitD(x, size_of_y=size_of_y, split_dim=split_dim, num_split=num_split, dependencies=dependencies, node_name=node_name)


# Auto infer num outputs for IR SplitV
@auto_convert_to_tensor([False, False, False], [False, False, False])
def SplitV(x: Tensor, size_splits: Tensor, split_dim: Tensor, *, num_split: int, dependencies=[], node_name=None):
    """REG_OP(SplitV)\n
  .INPUT(x, TensorType::BasicType())\n
  .INPUT(size_splits, TensorType::IndexNumberType())\n
  .INPUT(split_dim, TensorType({DT_INT32}))\n
  .DYNAMIC_OUTPUT(y, TensorType::BasicType())\n
  .REQUIRED_ATTR(num_split, Int)\n
  """

    size_of_y = num_split
    return raw_ops._SplitV(x, size_splits, split_dim, size_of_y=size_of_y, num_split=num_split, dependencies=dependencies, node_name=node_name)


# Auto infer num outputs for IR SplitVD
@auto_convert_to_tensor([False], [False])
def SplitVD(x: Tensor, *, size_splits: List[int], split_dim: int, num_split: int, dependencies=[], node_name=None):
    """REG_OP(SplitVD)\n
  .INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT, DT_FLOAT16}))\n
  .DYNAMIC_OUTPUT(y, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT, DT_FLOAT16}))\n
  .REQUIRED_ATTR(size_splits, ListInt)\n
  .REQUIRED_ATTR(split_dim, Int)\n
  .REQUIRED_ATTR(num_split, Int)\n
  """

    if len(size_splits) != num_split:
        raise AssertionError("The length of size_splits must be equal to num_split.")
    size_of_y = num_split
    return raw_ops._SplitVD(x, size_of_y=size_of_y, size_splits=size_splits, split_dim=split_dim, num_split=num_split, dependencies=dependencies, node_name=node_name)


# Auto infer num outputs for IR ConcatOffset
@auto_convert_to_tensor([False, True], [False, False])
def ConcatOffset(concat_dim: Tensor, x: List[Tensor], *, dependencies=[], node_name=None):
    """REG_OP(ConcatOffset)\n
  .INPUT(concat_dim, TensorType({DT_INT32}))\n
  .DYNAMIC_INPUT(x, TensorType({DT_INT32}))\n
  .DYNAMIC_OUTPUT(y, TensorType({DT_INT32}))\n
  .REQUIRED_ATTR(N, Int)\n
  """

    size_of_y = len(x)
    N = size_of_y
    return raw_ops._ConcatOffset(concat_dim, x, size_of_y=size_of_y, N=N, dependencies=dependencies, node_name=node_name)


# Auto infer num outputs for IR ConcatOffsetD
@auto_convert_to_tensor([True], [False])
def ConcatOffsetD(x: List[Tensor], *, concat_dim: int, dependencies=[], node_name=None):
    """REG_OP(ConcatOffsetD)\n
  .DYNAMIC_INPUT(x, TensorType({DT_INT32}))\n
  .DYNAMIC_OUTPUT(y, TensorType({DT_INT32}))\n
  .REQUIRED_ATTR(concat_dim, Int)\n
  .REQUIRED_ATTR(N, Int)\n
  """

    size_of_y = len(x)
    N = size_of_y
    return raw_ops._ConcatOffsetD(x, size_of_y=size_of_y, concat_dim=concat_dim, N=N, dependencies=dependencies, node_name=node_name)


# Auto infer num outputs for IR Unpack
@auto_convert_to_tensor([False], [False])
def Unpack(x: Tensor, *, num: int, axis: int = 0, dependencies=[], node_name=None):
    """REG_OP(Unpack)\n
  .INPUT(x, TensorType::BasicType())\n
  .DYNAMIC_OUTPUT(y, TensorType::BasicType())\n
  .REQUIRED_ATTR(num, Int)\n
  .ATTR(axis, Int, 0)\n
  """

    size_of_y = num
    return raw_ops._Unpack(x, size_of_y=size_of_y, num=num, axis=axis, dependencies=dependencies, node_name=node_name)


@auto_convert_to_tensor([True], [False])
def ForeachCos(x: List[Tensor], *, dependencies=[], node_name=None):
    """REG_OP(ForeachCos)\n
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))\n
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))\n
    """
    size_of_y = len(x)
    return raw_ops._ForeachCos(x, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


@auto_convert_to_tensor([True], [False])
def ForeachACos(x: List[Tensor], *, dependencies=[], node_name=None):
    """REG_OP(ForeachACos)\n
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))\n
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))\n
    """
    
    size_of_y = len(x)
    return raw_ops._ForeachACos(x, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


@auto_convert_to_tensor([True], [False])
def ForeachAbs(x: List[Tensor], *, dependencies=[], node_name=None):
    """REG_OP(ForeachAbs)\n
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
    """
    size_of_y = len(x)
    return raw_ops._ForeachAbs(x, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


@auto_convert_to_tensor([True], [False])
def ForeachASin(x: List[Tensor], *, dependencies=[], node_name=None):
    """REG_OP(ForeachASin)\n
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))\n
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))\n
    """
    size_of_y = len(x)
    return raw_ops._ForeachASin(x, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


@auto_convert_to_tensor([True], [False])
def ForeachATan(x: List[Tensor], *, dependencies=[], node_name=None):
    """REG_OP(ForeachATan)\n
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))\n
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))\n
    """
    size_of_y = len(x)
    return raw_ops._ForeachATan(x, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


@auto_convert_to_tensor([True], [False])
def ForeachCosh(x: List[Tensor], *, dependencies=[], node_name=None):
    """REG_OP(ForeachCosh)\n
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))\n
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))\n
    """
    size_of_y = len(x)
    return raw_ops._ForeachCosh(x, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


@auto_convert_to_tensor([True], [False])
def ForeachErf(x: List[Tensor], *, dependencies=[], node_name=None):
    """REG_OP(ForeachErf)\n
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))\n
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))\n
    """
    size_of_y = len(x)
    return raw_ops._ForeachErf(x, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


@auto_convert_to_tensor([True], [False])
def ForeachErfc(x: List[Tensor], *, dependencies=[], node_name=None):
    """REG_OP(ForeachErfc)\n
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))\n
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))\n
    """
    size_of_y = len(x)
    return raw_ops._ForeachErfc(x, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


@auto_convert_to_tensor([True], [False])
def ForeachExp(x: List[Tensor], *, dependencies=[], node_name=None):
    """REG_OP(ForeachExp)\n
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))\n
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))\n
    """
    size_of_y = len(x)
    return raw_ops._ForeachExp(x, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


@auto_convert_to_tensor([True], [False])
def ForeachExpm1(x: List[Tensor], *, dependencies=[], node_name=None):
    """REG_OP(ForeachExpm1)\n
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))\n
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))\n
    """
    size_of_y = len(x)
    return raw_ops._ForeachExpm1(x, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


@auto_convert_to_tensor([True], [False])
def ForeachLog(x: List[Tensor], *, dependencies=[], node_name=None):
    """REG_OP(ForeachLog)\n
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))\n
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))\n
    """
    size_of_y = len(x)
    return raw_ops._ForeachLog(x, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


@auto_convert_to_tensor([True], [False])
def ForeachLog10(x: List[Tensor], *, dependencies=[], node_name=None):
    """REG_OP(ForeachLog10)\n
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))\n
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))\n
    """
    size_of_y = len(x)
    return raw_ops._ForeachLog10(x, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


@auto_convert_to_tensor([True], [False])
def ForeachLog1p(x: List[Tensor], *, dependencies=[], node_name=None):
    """REG_OP(ForeachLog1p)\n
  .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))\n
  .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))\n
  """
    size_of_y = len(x)
    return raw_ops._ForeachLog1p(x, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


@auto_convert_to_tensor([True], [False])
def ForeachLog2(x: List[Tensor], *, dependencies=[], node_name=None):
    """REG_OP(ForeachLog2)\n
  .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))\n
  .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))\n
  """
    size_of_y = len(x)
    return raw_ops._ForeachLog2(x, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


@auto_convert_to_tensor([True], [False])
def ForeachSigmoid(x: List[Tensor], *, dependencies=[], node_name=None):
    """REG_OP(ForeachSigmoid)\n
  .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))\n
  .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))\n
  """
    size_of_y = len(x)
    return raw_ops._ForeachSigmoid(x, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


@auto_convert_to_tensor([True], [False])
def ForeachSin(x: List[Tensor], *, dependencies=[], node_name=None):
    """REG_OP(ForeachSin)\n
  .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))\n
  .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))\n
  """
    size_of_y = len(x)
    return raw_ops._ForeachSin(x, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


@auto_convert_to_tensor([True], [False])
def ForeachSinh(x: List[Tensor], *, dependencies=[], node_name=None):
    """REG_OP(ForeachSinh)\n
  .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))\n
  .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))\n
  """
    size_of_y = len(x)
    return raw_ops._ForeachSinh(x, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


@auto_convert_to_tensor([True], [False])
def ForeachTan(x: List[Tensor], *, dependencies=[], node_name=None):
    """REG_OP(ForeachTan)\n
  .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))\n
  .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))\n
  """
    size_of_y = len(x)
    return raw_ops._ForeachTan(x, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


@auto_convert_to_tensor([True], [False])
def ForeachTanh(x: List[Tensor], *, dependencies=[], node_name=None):
    """REG_OP(ForeachTanh)\n
  .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))\n
  .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))\n
  """
    size_of_y = len(x)
    return raw_ops._ForeachTanh(x, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


@auto_convert_to_tensor([True, False], [False, False])
def ForeachRoundOffNumber(x: List[Tensor], roundMode: int, *, dependencies=[], node_name=None):
    """REG_OP(ForeachRoundOffNumber)\n
  .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))\n
  .REQUIRED_ATTR(roundMode, Int)\n
  .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))\n
  """
    size_of_y = len(x)
    return raw_ops._ForeachRoundOffNumber(x, roundMode=roundMode, size_of_y=size_of_y,
                                          dependencies=dependencies, node_name=node_name)


@auto_convert_to_tensor([True, True, True, False], [False, False, False, False])
def ForeachAddcdivScalar(x: List[Tensor], x1: List[Tensor], x2: List[Tensor], scalar: Tensor, *, dependencies=[], node_name=None):
    """REG_OP(ForeachAddcdivScalar)\n
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
    """
    size_of_y = len(x)
    return raw_ops._ForeachAddcdivScalar(x, x1, x2, scalar, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


@auto_convert_to_tensor([True, True, True, False], [False, False, False, False])
def ForeachAddcdivScalarList(x: List[Tensor], x1: List[Tensor], x2: List[Tensor], scalar: Tensor, *, dependencies=[], node_name=None):
    """REG_OP(ForeachAddcdivScalarList)\n
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
    """
    size_of_y = len(x)
    return raw_ops._ForeachAddcdivScalarList(x, x1, x2, scalar, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


@auto_convert_to_tensor([True, True, True, False], [False, False, False, False])
def ForeachAddcdivList(x: List[Tensor], x1: List[Tensor], x2: List[Tensor], scalar: Tensor, *, dependencies=[], node_name=None):
    """REG_OP(ForeachAddcdivList)\n
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
    """
    size_of_y = len(x)
    return raw_ops._ForeachAddcdivList(x, x1, x2, scalar, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


# This api is auto-generated from IR ForeachAddcmulScalar
@auto_convert_to_tensor([True, True, True, False], [False, False, False, False])
def ForeachAddcmulScalar(x1: List[Tensor], x2: List[Tensor], x3: List[Tensor], scalar: Tensor, *, dependencies=[], node_name=None):
    """REG_OP(ForeachAddcmulScalar)\n
.DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))\n
.DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))\n
.DYNAMIC_INPUT(x3, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))\n
.INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))\n
.DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))\n
"""

    size_of_y = len(x1)
    return raw_ops._ForeachAddcmulScalar(x1, x2, x3, scalar, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


# This api is auto-generated from IR ForeachAddcmulScalarList
@auto_convert_to_tensor([True, True, True, False], [False, False, False, False])
def ForeachAddcmulScalarList(x1: List[Tensor], x2: List[Tensor], x3: List[Tensor], scalar: Tensor, *, dependencies=[], node_name=None):
    """REG_OP(ForeachAddcmulScalarList)\n
.DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))\n
.DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))\n
.DYNAMIC_INPUT(x3, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))\n
.INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))\n
.DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))\n
"""

    size_of_y = len(x1)
    return raw_ops._ForeachAddcmulScalarList(x1, x2, x3, scalar, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


@auto_convert_to_tensor([True, True, True, False], [False, False, False, False])
def ForeachAddcmulList(x: List[Tensor], x1: List[Tensor], x2: List[Tensor], scalar: Tensor, *, dependencies=[], node_name=None):
    """REG_OP(ForeachAddcmulList)\n
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
    """
    size_of_y = len(x)
    return raw_ops._ForeachAddcmulList(x, x1, x2, scalar, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


# This api is auto-generated from IR ForeachLerpScalar
@auto_convert_to_tensor([True, True, False], [False, False, False])
def ForeachLerpScalar(x1: List[Tensor], x2: List[Tensor], weight: Tensor, *, dependencies=[], node_name=None):
    """REG_OP(ForeachLerpScalar)\n
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
    .INPUT(weight, TensorType({DT_FLOAT, DT_FLOAT16}))\n
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
    .OP_END_FACTORY_REG(ForeachLerpScalar)\n
"""
    size_of_y = len(x1)
    return raw_ops._ForeachLerpScalar(x1, x2, weight, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


# This api is auto-generated from IR ForeachLerpList
@auto_convert_to_tensor([True, True, True], [False, False, False])
def ForeachLerpList(x1: List[Tensor], x2: List[Tensor], weights: List[Tensor], *, dependencies=[], node_name=None):
    """REG_OP(ForeachLerpList)\n
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
    .INPUT(weight, TensorType({DT_FLOAT, DT_FLOAT16}))\n
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
    .OP_END_FACTORY_REG(ForeachLerpList)\n
"""
    size_of_y = len(x1)
    return raw_ops._ForeachLerpList(x1, x2, weights, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


# This api is auto-generated from IR ForeachMaximumScalar
@auto_convert_to_tensor([True, False], [False, False])
def ForeachMaximumScalar(x: List[Tensor], scalar: Tensor, *, dependencies=[], node_name=None):
    """REG_OP(ForeachMaximumScalar)\n
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16}))\n
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
    .OP_END_FACTORY_REG(ForeachMaximumScalar)\n
"""
    size_of_y = len(x)
    return raw_ops._ForeachMaximumScalar(x, scalar, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


# This api is auto-generated from IR ForeachMaximumScalarList
@auto_convert_to_tensor([True, False], [False, False])
def ForeachMaximumScalarList(x: List[Tensor], scalars: Tensor, *, dependencies=[], node_name=None):
    """REG_OP(ForeachMaximumScalarList)\n
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
"""

    size_of_y = len(x)
    return raw_ops._ForeachMaximumScalarList(x, scalars, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


# This api is auto-generated from IR ForeachMaximumList
@auto_convert_to_tensor([True, True], [False, False])
def ForeachMaximumList(x1: List[Tensor], x2: List[Tensor], *, dependencies=[], node_name=None):
    """REG_OP(ForeachMaximumList)\n
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
"""

    size_of_y = len(x1)
    return raw_ops._ForeachMaximumList(x1, x2, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


# This api is auto-generated from IR ForeachMinimumScalar
@auto_convert_to_tensor([True, False], [False, False])
def ForeachMinimumScalar(x: List[Tensor], scalar: Tensor, *, dependencies=[], node_name=None):
    """REG_OP(ForeachMinimumScalar)\n
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16}))\n
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
    .OP_END_FACTORY_REG(ForeachMinimumScalar)\n
"""
    size_of_y = len(x)
    return raw_ops._ForeachMinimumScalar(x, scalar, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


# This api is auto-generated from IR ForeachMinimumScalarList
@auto_convert_to_tensor([True, False], [False, False])
def ForeachMinimumScalarList(x: List[Tensor], scalars: Tensor, *, dependencies=[], node_name=None):
    """REG_OP(ForeachMinimumScalarList)\n
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
"""

    size_of_y = len(x)
    return raw_ops._ForeachMinimumScalarList(x, scalars, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


# This api is auto-generated from IR ForeachMinimumList
@auto_convert_to_tensor([True, True], [False, False])
def ForeachMinimumList(x1: List[Tensor], x2: List[Tensor], *, dependencies=[], node_name=None):
    """REG_OP(ForeachMinimumList)\n
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
"""

    size_of_y = len(x1)
    return raw_ops._ForeachMinimumList(x1, x2, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


# This api is auto-generated from IR ForeachNorm
@auto_convert_to_tensor([True, False], [False, False])
def ForeachNorm(x: List[Tensor], scalar: Tensor, *, dependencies=[], node_name=None):
    """REG_OP(ForeachNorm)\n
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
"""

    size_of_y = len(x)
    return raw_ops._ForeachNorm(x, scalar, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


# This api is auto-generated from IR ForeachAddScalar
@auto_convert_to_tensor([True, False], [False, False])
def ForeachAddScalar(x: List[Tensor], scalar: Tensor, *, dependencies=[], node_name=None):
    """REG_OP(ForeachAddScalar)\n
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
"""

    size_of_y = len(x)
    return raw_ops._ForeachAddScalar(x, scalar, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


# This api is auto-generated from IR ForeachAddList
@auto_convert_to_tensor([True, True, False], [False, False, False])
def ForeachAddList(x1: List[Tensor], x2: List[Tensor], alpha: Tensor, *, dependencies=[], node_name=None):
    """REG_OP(ForeachAddList)\n
.DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.INPUT(alpha, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
"""

    size_of_y = len(x1)
    return raw_ops._ForeachAddList(x1, x2, alpha, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


# This api is auto-generated from IR ForeachAddScalarList
@auto_convert_to_tensor([True, False], [False, False])
def ForeachAddScalarList(x1: List[Tensor], x2: Tensor, *, dependencies=[], node_name=None):
    """REG_OP(ForeachAddScalarList)\n
.DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.INPUT(alpha, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
"""

    size_of_y = len(x1)
    return raw_ops._ForeachAddScalarList(x1, x2, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


# This api is auto-generated from IR ForeachDivScalarList
@auto_convert_to_tensor([True, False], [False, False])
def ForeachDivScalarList(x1: List[Tensor], x2: Tensor, *, dependencies=[], node_name=None):
    """REG_OP(ForeachDivScalarList)\n
.DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.INPUT(alpha, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
"""

    size_of_y = len(x1)
    return raw_ops._ForeachDivScalarList(x1, x2, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


# This api is auto-generated from IR ForeachDivScalar
@auto_convert_to_tensor([True, False], [False, False])
def ForeachDivScalar(x: List[Tensor], scalar: Tensor, *, dependencies=[], node_name=None):
    """REG_OP(ForeachDivScalar)\n
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
"""

    size_of_y = len(x)
    return raw_ops._ForeachDivScalar(x, scalar, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


# This api is auto-generated from IR ForeachDivList
@auto_convert_to_tensor([True, True], [False, False])
def ForeachDivList(x1: List[Tensor], x2: List[Tensor], *, dependencies=[], node_name=None):
    """REG_OP(ForeachDivList)\n
.DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16}))\n
.DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16}))\n
.DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))\n
"""
    size_of_y = len(x1)
    return raw_ops._ForeachDivList(x1, x2, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


# This api is auto-generated from IR ForeachMulList
# @auto_convert_to_tensor([True, True, False], [False, False, False])
@auto_convert_to_tensor([True, True], [False, False])
def ForeachMulList(x1: List[Tensor], x2: List[Tensor], *, dependencies=[], node_name=None):
    """REG_OP(ForeachMulList)\n
.DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
"""

    size_of_y = len(x1)
    return raw_ops._ForeachMulList(x1, x2, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


# This api is auto-generated from IR ForeachMulScalar
@auto_convert_to_tensor([True, False], [False, False])
def ForeachMulScalar(x: List[Tensor], scalar: Tensor, *, dependencies=[], node_name=None):
    """REG_OP(ForeachMulScalar)\n
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT
"""

    size_of_y = len(x)
    return raw_ops._ForeachMulScalar(x, scalar, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


# This api is auto-generated from IR ForeachMulScalarList
@auto_convert_to_tensor([True, False], [False, False])
def ForeachMulScalarList(x1: List[Tensor], x2: Tensor, *, dependencies=[], node_name=None):
    """REG_OP(ForeachMulScalarList)\n
.DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.INPUT(alpha, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
"""

    size_of_y = len(x1)
    return raw_ops._ForeachMulScalarList(x1, x2, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


# This api is auto-generated from IR ForeachNeg
@auto_convert_to_tensor([True], [False])
def ForeachNeg(x: List[Tensor], *, dependencies=[], node_name=None):
    """REG_OP(ForeachNeg)\n
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
"""

    size_of_y = len(x)
    return raw_ops._ForeachNeg(x, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


# This api is auto-generated from IR ForeachPowList
@auto_convert_to_tensor([True, True], [False, False])
def ForeachPowList(x: List[Tensor], x1: List[Tensor], *, dependencies=[], node_name=None):
    """REG_OP(ForeachPowList)\n
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))\n
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))\n
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))\n
"""

    size_of_y = len(x)
    return raw_ops._ForeachPowList(x, x1, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


# This api is auto-generated from IR ForeachPowScalar
@auto_convert_to_tensor([True, False], [False, False])
def ForeachPowScalar(x: List[Tensor], scalar: Tensor, *, dependencies=[], node_name=None):
    """REG_OP(ForeachPowScalar)\n
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))\n
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))\n
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))\n
"""

    size_of_y = len(x)
    return raw_ops._ForeachPowScalar(x, scalar, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


# This api is auto-generated from IR ForeachPowScalarList
@auto_convert_to_tensor([True, False], [False, False])
def ForeachPowScalarList(x: List[Tensor], scalar: Tensor, *, dependencies=[], node_name=None):
    """REG_OP(ForeachPowScalarList)\n
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))\n
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))\n
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))\n
"""

    size_of_y = len(x)
    return raw_ops._ForeachPowScalarList(x, scalar, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


# This api is auto-generated from IR ForeachPowScalarAndTensor
@auto_convert_to_tensor([False, True], [False, False])
def ForeachPowScalarAndTensor(scalar: Tensor, x: List[Tensor], *, dependencies=[], node_name=None):
    """REG_OP(ForeachPowScalarAndTensor)\n
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))\n
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))\n
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))\n
    """

    size_of_y = len(x)
    return raw_ops._ForeachPowScalarAndTensor(scalar, x, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


# This api is auto-generated from IR ForeachSubScalar
@auto_convert_to_tensor([True, False], [False, False])
def ForeachSubScalar(x: List[Tensor], scalar: Tensor, *, dependencies=[], node_name=None):
    """REG_OP(ForeachSubScalar)\n
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
"""

    size_of_y = len(x)
    return raw_ops._ForeachSubScalar(x, scalar, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


# This api is auto-generated from IR ForeachSubList
@auto_convert_to_tensor([True, True, False], [False, False, False])
def ForeachSubList(x: List[Tensor], x1: List[Tensor], alpha: Tensor, *, dependencies=[], node_name=None):
    """REG_OP(ForeachSubList)\n
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
.INPUT(alpha, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
"""

    size_of_y = len(x)
    return raw_ops._ForeachSubList(x, x1, alpha, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)


# This api is auto-generated from IR ForeachSubScalarList
@auto_convert_to_tensor([True, False], [False, False])
def ForeachSubScalarList(x: List[Tensor], scalar: Tensor, *, dependencies=[], node_name=None):
    """REG_OP(ForeachSubScalarList)\n
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
.DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))\n
"""

    size_of_y = len(x)
    return raw_ops._ForeachSubScalarList(x, scalar, size_of_y=size_of_y, dependencies=dependencies, node_name=node_name)

