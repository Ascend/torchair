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
@auto_convert_to_tensor([True, True, True, True, True, True, True, False],
                        [False, False, False, False, False, False, False, True])
def GroupedMatmul(x: List[Tensor], weight: List[Tensor], bias: List[Tensor], scale: List[Tensor], offset: List[Tensor],
                  antiquant_scale: List[Tensor], antiquant_offset: List[Tensor], group_list: Optional[Tensor], *,
                  split_item: int = 0, dtype: int = 0, transpose_weight: bool = False, transpose_x: bool = False,
                  group_type: int = -1):
    """REG_OP(GroupedMatmul)\n
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))\n
    .DYNAMIC_INPUT(weight, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))\n
    .DYNAMIC_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))\n
    .DYNAMIC_INPUT(scale, TensorType({DT_UINT64}))\n
    .DYNAMIC_INPUT(offset, TensorType({DT_FLOAT32}))\n
    .DYNAMIC_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))\n
    .DYNAMIC_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))\n
    .OPTIONAL_INPUT(group_list, TensorType({DT_INT64}))\n
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))\n
    .ATTR(split_item, Int, 0)\n
    .ATTR(dtype, Int, 0)\n
    .ATTR(transpose_weight, Bool, false)\n
    .ATTR(transpose_x, Bool, false)\n
    .ATTR(group_type, Int, -1)\n
    """

    size_of_y = 0
    if split_item == 0 or split_item == 1:
        size_of_y = len(x)
    elif split_item == 2 or split_item == 3:
        size_of_y = 1

    return raw_ops._GroupedMatmul(x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, group_list,
                                  size_of_y=size_of_y, split_item=split_item, dtype=dtype,
                                  transpose_weight=transpose_weight, transpose_x=transpose_x, group_type=group_type)


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
    return raw_ops._Batch(x_tensors, size_of_y_tensors=size_of_y_tensors, num_batch_threads=num_batch_threads, max_batch_size=max_batch_size, batch_timeout_micros=batch_timeout_micros, grad_timeout_micros=grad_timeout_micros, max_enqueued_batches=max_enqueued_batches, allowed_batch_sizes=allowed_batch_sizes, container=container, shared_name=shared_name, batching_queue=batching_queue, dependencies=dependencies, node_name=node_name)


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
    raise RuntimeError(
        "Fix this by determine num outputs of BoostedTreesBucketize")
    return raw_ops._BoostedTreesBucketize(float_values, bucket_boundaries, size_of_y=size_of_y, num_features=num_features, dependencies=dependencies, node_name=node_name)


# Auto infer num outputs for IR SwitchN
@auto_convert_to_tensor([False, False], [False, False])
def SwitchN(data: Tensor, pred_value: Tensor, *, dependencies=[], node_name=None):
    """REG_OP(SwitchN)\n
  .INPUT(data, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_BOOL}))\n
  .INPUT(pred_value, TensorType({DT_INT64}))\n
  .DYNAMIC_OUTPUT(output, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_BOOL}))\n
  """

    size_of_output = 0
    raise RuntimeError(
        "Fix this by determine num outputs of BoostedTreesBucketize")
    return raw_ops._SwitchN(data, pred_value, size_of_output=size_of_output, dependencies=dependencies, node_name=node_name)


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
    raise RuntimeError(
        "Fix this by determine num outputs of CTCBeamSearchDecoder")
    return raw_ops._CTCBeamSearchDecoder(inputs, sequence_length, size_of_decoded_indices=size_of_decoded_indices, size_of_decoded_values=size_of_decoded_values, size_of_decoded_shape=size_of_decoded_shape, beam_width=beam_width, top_paths=top_paths, merge_repeated=merge_repeated, dependencies=dependencies, node_name=node_name)


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
        raise AssertionError
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
