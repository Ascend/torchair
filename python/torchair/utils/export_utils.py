import os
from typing import TypedDict
import torch

from torchair.core.utils import logger
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.ge_graph import torch_type_to_ge_type
from torchair.ge_concrete_graph.ge_graph import compat_as_bytes, GeGraph
from torchair.ge_concrete_graph.utils import dump_graph
from torchair.utils.path_manager import PathManager
from torchair.ge_concrete_graph.ge_ir_pb2 import ModelDef


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
        export_file_name = export_name + str(rank) + ".air"
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
    logger.debug(f'exported graph name: {export_graph.name}')

    sub_file_path = _get_subpath(root_file_path)
    os.makedirs(sub_file_path, exist_ok=True)

    weight_externalized, used_weight_num = _convert_data_to_const(inputs, export_graph, sub_file_path, weight_name)

    if used_weight_num != 0 and weight_externalized:
        _save_weight2file(inputs, sub_file_path, weight_name, used_weight_num)

    dump_graph(sub_file_path + "/dynamo.pbtxt", export_graph)

    return export_graph


def _get_dict_attr_key_value_name(attr_name):
    return attr_name + "_keys", attr_name + "_values"


def serialize_str_dict_attr(model_def: ModelDef, attr_name, attr_dict):
    key_name, value_name = _get_dict_attr_key_value_name(attr_name)
    model_def.attr[key_name].list.s[:] = [compat_as_bytes(_) for _ in attr_dict.keys()]
    model_def.attr[key_name].list.val_type = 1
    model_def.attr[value_name].list.s[:] = [compat_as_bytes(_) for _ in attr_dict.values()]
    model_def.attr[value_name].list.val_type = 1


def serialize_int_dict_attr(model_def: ModelDef, attr_name, attr_dict):
    key_name, value_name = _get_dict_attr_key_value_name(attr_name)
    model_def.attr[key_name].list.i[:] = [_ for _ in attr_dict.keys()]
    model_def.attr[key_name].list.val_type = 2
    model_def.attr[value_name].list.i[:] = [_ for _ in attr_dict.values()]
    model_def.attr[value_name].list.val_type = 2


def _get_attr_list(model_def: ModelDef, attr_name):
    if model_def.attr[attr_name].list.val_type == 2:
        return model_def.attr[attr_name].list.i[:]
    elif model_def.attr[attr_name].list.val_type == 1:
        return [_.decode("utf-8") for _ in model_def.attr[attr_name].list.s]
    else:
        raise ValueError(f"Unsupported value type {model_def.attr[attr_name].list.val_type}")


def unserialize_dict_attr(model_def: ModelDef, attr_name):
    key_name, value_name = _get_dict_attr_key_value_name(attr_name)

    list_key = _get_attr_list(model_def, key_name)
    list_value = _get_attr_list(model_def, value_name)

    logger.debug(f"get {key_name} keys from graph: {list_key}")
    logger.debug(f"get {value_name} values from graph:: {list_value}")
    if len(list_key) != len(list_value):
        raise AssertionError(f"{attr_name} keys len {len(list_key)}"
                             f" is not equal to {attr_name} values len {len(list_value)}")
    return {list_key[i]: list_value[i] for i in range(len(list_key))}
