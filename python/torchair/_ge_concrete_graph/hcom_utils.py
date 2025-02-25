import inspect

import torch
from torchair.ge._ge_graph import compat_as_bytes, get_default_ge_graph


def get_group_name_and_record(tag, rank_list, group_size):
    pg = torch.distributed.distributed_c10d._find_or_create_pg_by_ranks_and_tag(tag, rank_list, group_size)
    device = torch.distributed.distributed_c10d._get_pg_default_device(pg)
    if device.type == "cpu":
        # create unqiue group name to same pg by tag and ranklist
        group_name = encode_pg_tag_ranklist(tag, rank_list)
    elif device.type == "npu":
        rank = torch.distributed.get_rank()
        group_name = pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank, init_comm=True)
    else:
        raise ValueError("The initialized aggregate communication backend is not a CPU or NPU.")
    # record rank list for export
    get_default_ge_graph().record_process_group(group_name, rank_list, tag)
    return group_name


def record_pg_to_graph(graph):
    # This func will update pg to graph, special for op like MC2, who is not record process group in convert
    if not torch.distributed.is_initialized():
        return
    all_created_pg = {}
    rank = torch.distributed.get_rank()
    for pg in torch.distributed.distributed_c10d._world.pg_map.keys():
        if torch.distributed.distributed_c10d.get_backend(pg) != "hccl":
            continue
        hcom_pg_name = pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank, init_comm=False)
        if hcom_pg_name != "":
            pg_rank_list = torch.distributed.distributed_c10d.get_process_group_ranks(pg)
            tag = torch.distributed.distributed_c10d._get_group_tag(pg)
            all_created_pg[hcom_pg_name] = pg_rank_list, tag

    for op in graph.op:
        if "group" not in op.attr:
            continue
        group_name = op.attr["group"].s.decode()
        if group_name in all_created_pg:
            graph.record_process_group(group_name, all_created_pg[group_name][0], all_created_pg[group_name][1])


def encode_pg_tag_ranklist(tag, rank_list):
    return f"tag={tag};rank_list={','.join(str(i) for i in rank_list)}"


def rename_cached_pgname(graph_proto, pg_name_to_tag_ranklist):
    if len(pg_name_to_tag_ranklist) == 0:
        return
    for op in graph_proto.op:
        if "group" not in op.attr:
            continue
        group_name = op.attr["group"].s.decode()
        if group_name not in pg_name_to_tag_ranklist:
            continue
        rank_list, tag = pg_name_to_tag_ranklist[group_name]
        op.attr["group"].s = compat_as_bytes(encode_pg_tag_ranklist(tag, rank_list))


def codegen_refresh_cache_pgname(used_process_group):
    from torch._inductor.utils import IndentedBuffer
    code = IndentedBuffer()
    code.writelines([f'def init_process_group(ge_graph):'])
    with code.indent():
        fun_encode_pg_tag_ranklist_source = inspect.getsource(encode_pg_tag_ranklist)
        code.writelines(fun_encode_pg_tag_ranklist_source.splitlines())

        code.splice(f'''
        cache_inited_group = {{}}
        used_process_group = {used_process_group}
        rank = torch.distributed.get_rank();
        for group_name, (rank_list, tag) in used_process_group.items():
            pg = torch.distributed.distributed_c10d._find_pg_by_ranks_and_tag(tag, rank_list)
            if pg is None:
                raise AssertionError(f"During cache loading, ",
                                     f"the pg with the same name created during save cache could not be found. ",
                                     f"Possible reasons for this situation include an inconsistency in the number ",
                                     f"of pgs created during save cache and the input parameters when loading cache,",
                                     f" which makes it impossible to find the created pg through the tag. ",
                                     f"Please check the script, or delete the cache files of the cache_module ",
                                     f"and try again.")
            new_group_name = pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank, init_comm=True)
            cache_inited_group[encode_pg_tag_ranklist(tag, rank_list)] = new_group_name

        from torchair.ge.attr import Str
        for op in ge_graph._proto.op:
            if "group" not in op.attr:
                continue
            tag_ranklist = Str.get(op.attr["group"]).value
            if tag_ranklist not in cache_inited_group:
                continue
            Str(cache_inited_group[tag_ranklist]).merge_to(op.attr["group"])
        ''')
    code.writelines([f'init_process_group(ge_graph)'])
    return code
