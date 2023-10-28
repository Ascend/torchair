import torch

from torchair.core.utils import logger


def get_subpath(export_path_dir):
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
