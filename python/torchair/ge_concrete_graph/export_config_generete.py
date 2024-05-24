import json
import logging
from typing import List, Set
import torch

from torchair.core.utils import logger
from torchair.ge_concrete_graph.ge_ir_pb2 import GraphDef
from torchair._utils.path_manager import PathManager
from torchair._utils.export_utils import get_export_rank_file_name


hcom_ops_set = {"HcomAllReduce", "HcomReduceScatter", "HcomAllGather", "HcomBroadcast", "HcomReduce",
                "HcomSend", "HcomReceive", "HcomRemoteRead", "HcomRemoteRefRead", "HcomRemoteWrite",
                "HcomRemoteScatterWrite", "HcomAllToAllV", "HcomAllToAll", "HcomAllToAllVC",
                "HcomGatherAllToAllV", "HorovodAllgather", "HorovodAllreduce", "HorovodBroadcast"}


def _generate_model_relation_config(file_path, export_name, world_ranklist: List, group_set: Set):
    if torch.distributed.get_rank() != 0:
        return
    model_relation_config = {"deploy_config": [], "model_name_to_instance_id": [],
                             "comm_group": [], "rank_table": []}

    for rankid in world_ranklist:
        submodel_name = get_export_rank_file_name(export_name, rankid)
        deploy_config_dict = {}
        deploy_config_dict["submodel_name"] = submodel_name
        # torch中不存在cluster概念，不能获得nodeid信息,因此全部写为0，需要用户手动调整，资料中解释
        deploy_config_dict["deploy_device_id_list"] = "0:0:" + str(rankid)
        model_relation_config["deploy_config"].append(deploy_config_dict)

        model_name_to_instance_id_dict = {}
        model_name_to_instance_id_dict["submodel_name"] = submodel_name
        model_name_to_instance_id_dict["model_instance_id"] = rankid
        model_relation_config["model_name_to_instance_id"].append(
            model_name_to_instance_id_dict)

        rank_table_dict = {}
        rank_table_dict["rank_id"] = rankid
        rank_table_dict["model_instance_id"] = rankid
        model_relation_config["rank_table"].append(rank_table_dict)

    for group_tuple in group_set:
        comm_group_dict = {}
        comm_group_dict["group_name"] = group_tuple[0]
        comm_group_dict["group_rank_list"] = str(group_tuple[1])
        model_relation_config["comm_group"].append(comm_group_dict)

    PathManager.check_path_writeable_and_safety(file_path + "/model_relation_config.json")
    with open(file_path + "/model_relation_config.json", 'w') as write_f:
        json.dump(model_relation_config, write_f, indent=4, ensure_ascii=False)

    return


def _generate_numa_config(file_path, world_ranklist: List):
    if torch.distributed.get_rank() != 0:
        return
    numa_config = {"cluster": [], "item_def": [{"item_type": "Ascend910"}],
                   "node_def": [{"item": [{"item_type": "Ascend910"}]}]}
    cluster_nodes = {"cluster_nodes": [], "nodes_toplogy": {}}
    # torch中不能获得nodeid信息,因此node只有node_id=0，将全部rank都放在node0中，后续需要用户手动调整
    node = {"node_id": 0, "node_type": "ATLAS800",
            "ipaddr": "127.0.0.1", "port": 29500, "item_list": []}

    for rankid in world_ranklist:
        item = {}
        item["item_id"] = rankid
        node["item_list"].append(item)
    cluster_nodes["cluster_nodes"].append(node)
    numa_config["cluster"].append(cluster_nodes)
    PathManager.check_path_writeable_and_safety(file_path + "/numa_config.json")
    with open(file_path + "/numa_config.json", 'w') as write_f:
        json.dump(numa_config, write_f, indent=4, ensure_ascii=False)

    return


def _get_groups_from_graph(graph: GraphDef, group_set: Set):
    for op in graph.op:
        if op.type in hcom_ops_set:
            str_groupname = op.attr["group"].s.decode()
            str_ranklist = str(op.attr["ranklist"].list.i)
            group_set.add((str_groupname, str_ranklist))
            logger.info(f"{op.type} in group = {str_groupname} ranklist = {str_ranklist}")
            if len(group_set) > 1:
                # 目前后端部署，只支持单集合通信域
                logger.error(f"Cann currently does not support multiple groups," + \
                             f"found at least two groups exist, " + \
                             f"failed to generate configuration file.")
                return False
    return True


def generate_config(config, file_path, export_graph):
    if torch.distributed.is_initialized():
        default_pg = torch.distributed.distributed_c10d._get_default_group()
        default_pg_rank_list = torch.distributed.get_process_group_ranks(default_pg)
        group_set = set()
        if _get_groups_from_graph(export_graph, group_set):
            logger.info(f"generate_atc_config file_path: {file_path}, " +
                        "file_name: {config.export.export_name}")
            _generate_model_relation_config(file_path, config.export.export_name,
                                            default_pg_rank_list, group_set)
            _generate_numa_config(file_path, default_pg_rank_list)
    return
