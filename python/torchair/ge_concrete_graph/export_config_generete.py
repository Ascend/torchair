import json
import logging
from typing import List
import torch

from torchair.core.utils import logger
from torchair.ge_concrete_graph.ge_ir_pb2 import GraphDef


def generate_atc_config(file_path, export_name, world_ranklist: List, group_list: List):
    model_relation_config = {"deploy_config": [], "model_name_to_instance_id": [],
                             "comm_group": [], "rank_table": []}
    rank = torch.distributed.get_rank()
    if str(rank) != "0":
        return

    for rankid in world_ranklist:
        submodel_name = export_name + "_rank" + str(rankid) + ".air"
        deploy_config_dict = {}
        deploy_config_dict["submodel_name"] = submodel_name
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

    for group_tuple in group_list:
        comm_group_dict = {}
        comm_group_dict["group_name"] = group_tuple[0]
        comm_group_dict["group_rank_list"] = str(group_tuple[1])
        model_relation_config["comm_group"].append(comm_group_dict)

    with open(file_path + "/model_relation_config.json", 'w') as write_f:
        json.dump(model_relation_config, write_f, indent=4, ensure_ascii=False)

    numa_config = {"cluster": [], "item_def": [{"item_type": ""}],
                   "node_def": [{"item": [{"item_type": ""}]}]}
    cluster_nodes = {"cluster_nodes": [], "nodes_toplogy": {}}
    node = {"node_id": 0, "node_type": "",
            "ipaddr": "0.0.0.0", "port": 0, "item_list": []}

    for rankid in world_ranklist:
        item = {}
        item["item_id"] = rankid
        node["item_list"].append(item)
    cluster_nodes["cluster_nodes"].append(node)
    numa_config["cluster"].append(cluster_nodes)
    with open(file_path + "/numa_config.json", 'w') as write_f:
        json.dump(numa_config, write_f, indent=4, ensure_ascii=False)

    return


def get_grouplist_from_graph(graph: GraphDef):
    group_list = []
    for op in graph.op:
        if op.type == "HcomAllReduce" or op.type == "HcomReduceScatter":
            str_groupname = str(op.attr["group"].s)
            str_ranklist = str(op.attr["ranklist"].list.i)
            logger.info(
                f"{op.type} in group = {str_groupname} ranklist = {str_ranklist}")
            group_list.append((str_groupname, str_ranklist))
    return group_list
