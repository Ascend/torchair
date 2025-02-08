import os
from typing import TypedDict
import torch

from torchair.core.utils import logger
from torchair._ge_concrete_graph import ge_apis as ge
from torchair.ge._ge_graph import torch_type_to_ge_type
from torchair.ge._ge_graph import compat_as_bytes, GeGraph
from torchair._ge_concrete_graph.utils import dump_graph
from torchair._utils.path_manager import PathManager
from torchair._ge_concrete_graph.ge_ir_pb2 import ModelDef, GraphDef


def _sort_graph_data_index(graph: GraphDef):
    data_index = 0
    for op in graph.op:
        if op.type == "Data":
            op.attr["index"].i = data_index
            data_index += 1
    for op in graph.op:
        if op.type == "RefData":
            op.attr["index"].i = data_index
            data_index += 1


def get_export_rank_file_name(export_name, rank):
    return export_name + str(rank) + ".air"


def _get_subpath(export_path_dir):
    rank = None
    try:
        rank = torch.distributed.get_rank()
    except:
        logger.info(f'not frontend segmentation')

    if rank is not None:
        export_path_subdir = export_path_dir + "/" + "rank_" + str(rank)
    else:
        export_path_subdir = export_path_dir

    logger.info(f'export_path_subdir is {export_path_subdir}')
    return export_path_subdir


def get_export_file_name(export_name):
    rank = None
    try:
        rank = torch.distributed.get_rank()
    except:
        logger.info(f'not frontend segmentation')

    if rank is not None:
        export_file_name = get_export_rank_file_name(export_name, rank)
    else:
        export_file_name = export_name + ".air"

    logger.info(f'get_export_file_name is {export_file_name}')
    return export_file_name


def _make_const_node(input_tensor, name):
    y = ge.Const(input_tensor.cpu(),
                 dtype=torch_type_to_ge_type(input_tensor.dtype),
                 node_name=name,
                 readable=False)
    return y


def _is_weight_externalized(inputs, weight_name, export_graph):
    protobuf_size = export_graph.ByteSize()
    weight_externalized = False
    used_weight_num = 0
    # protobuf max size 2G, reserved 200M buffer
    max_protobuf_size = (2048 - 200) * 1024 * 1024
    for i, inp in enumerate(inputs):
        if id(inp) in weight_name:
            protobuf_size += inp.element_size() * inp.nelement()
            used_weight_num += 1

    if protobuf_size > max_protobuf_size:
        weight_externalized = True

    logger.info(f'export: protobuf_size and weight to const size = {protobuf_size} , ' + \
                f'max_protobuf_size {max_protobuf_size}, used_weight_num = {used_weight_num}' + \
                f' , and weight_externalized is {weight_externalized}')
    return weight_externalized, used_weight_num


def _convert_data_to_const(inputs, export_graph, file_path, weight_name):
    weight_externalized, used_weight_num = _is_weight_externalized(inputs, weight_name, export_graph)
    if used_weight_num == 0:
        return weight_externalized, used_weight_num

    for i, inp in enumerate(inputs):
        file_id = weight_name.get(id(inp))
        if file_id is not None:
            if not inp.is_contiguous():
                raise AssertionError
            logger.debug(f'  Weight {i} dtype: {inp.dtype} shape: {inp.shape}')
            if weight_externalized:
                y = ge.FileConstant(shape=list(inp.shape),
                                    dtype=torch_type_to_ge_type(inp.dtype),
                                    file_path=file_path + "/" + file_id.replace(".", "_"),
                                    node_name=export_graph.op[i].name)
            else:
                y = _make_const_node(inp, export_graph.op[i].name)
            export_graph.op[i].Clear()
            export_graph.op[i].MergeFrom(y.node)

    _sort_graph_data_index(export_graph)
    return weight_externalized, used_weight_num


def _save_weight2file(inputs, file_path, weight_name, used_weight_num):
    logger.info(f'save Weight tensor to file...')
    saved_num = 0
    for i, inp in enumerate(inputs):
        file_id = weight_name.get(id(inp))
        if file_id is None:
            continue

        file_path_and_name = file_path + "/" + file_id.replace(".", "_")
        if inp.dtype is torch.bfloat16:
            PathManager.check_path_writeable_and_safety(file_path_and_name)
            with open(file_path_and_name, "w") as f:
                # args0: file handle
                # args1: True mean save as readable not tar.gz or other
                # args2: False mean not save data len
                inp.cpu().untyped_storage()._write_file(f, True, False, torch._utils._element_size(torch.bfloat16))
        else:
            inp.numpy(force=True).tofile(file_path_and_name)

        saved_num += 1
        print('\r torchair dynamo export save weight {0}% {1}/{2}'.format(
            min(100, int(saved_num / used_weight_num * 100)), saved_num, used_weight_num), end='')
    print(" ")
    logger.info(f'save Weight tensor to file over...')


_next_export_graph_id = 0


def make_export_graph(ori_graph, inputs, root_file_path, weight_name):
    export_graph = GeGraph()
    export_graph.MergeFrom(ori_graph._proto)
    export_graph.set_used_process_group(ori_graph.used_process_group)
    logger.debug(f'exported graph name: {export_graph.name}')

    sub_file_path = _get_subpath(root_file_path)
    os.makedirs(sub_file_path, exist_ok=True)

    weight_externalized, used_weight_num = _convert_data_to_const(inputs, export_graph, sub_file_path, weight_name)

    if used_weight_num != 0 and weight_externalized:
        _save_weight2file(inputs, sub_file_path, weight_name, used_weight_num)

    dump_graph(sub_file_path + "/dynamo.pbtxt", export_graph)

    return export_graph
