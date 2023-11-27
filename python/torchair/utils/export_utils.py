import os
import torch

from torchair.core.utils import logger
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.ge_graph import torch_type_to_ge_type
from torchair.ge_concrete_graph.ge_graph import compat_as_bytes, GeGraph
from torchair.ge_concrete_graph.utils import dump_graph


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


def _convert_data_to_const(inputs, export_graph, file_path, config):
    weight_name = config.export_config.weight_name
    used_weight_num = 0

    for i, inp in enumerate(inputs):
        file_id = weight_name.get(id(inp))
        if file_id is not None:
            assert inp.is_contiguous()
            logger.debug(f'  Weight {i} dtype: {inp.dtype} shape: {inp.shape}')
            y = ge.FileConstant(shape=list(inp.shape),
                                dtype=torch_type_to_ge_type(inp.dtype),
                                file_path=file_path + "/" + file_id.replace(".", "_"),
                                node_name=export_graph.op[i].name)

            export_graph.op[i].Clear()
            export_graph.op[i].MergeFrom(y.node)
            used_weight_num += 1

    return used_weight_num


def _save_weight2file(inputs, file_path, weight_name, used_weight_num):
    if used_weight_num == 0:
        logger.info(f'no Weight tensor need save to file.')
        return

    logger.info(f'save Weight tensor to file...')
    saved_num = 0
    for i, inp in enumerate(inputs):
        file_id = weight_name.get(id(inp))
        if file_id is not None:
            file_path_and_name = file_path + "/" + file_id.replace(".", "_")
            inp.numpy(force=True).tofile(file_path_and_name)
            saved_num += 1
            print('\r torchair dynamo export save weight {0}% {1}/{2}'.format(
                  min(100, int(saved_num / used_weight_num * 100)), saved_num, used_weight_num), end='')
    print(" ")
    logger.info(f'save Weight tensor to file over...')


def make_export_graph(inputs, config, ori_graph):
    export_graph = GeGraph()
    export_graph.MergeFrom(ori_graph._proto)
    file_path = config.export_config.export_path_dir
    sub_file_path = _get_subpath(file_path)

    os.makedirs(sub_file_path, exist_ok=True)

    used_weight_num = _convert_data_to_const(inputs, export_graph, file_path, config)

    _save_weight2file(inputs, sub_file_path, config.export_config.weight_name, used_weight_num)

    dump_graph(sub_file_path + "/dynamo.pbtxt", export_graph)

    return export_graph