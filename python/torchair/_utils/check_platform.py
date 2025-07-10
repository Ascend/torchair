def is_not_support():
    try:
        import torch_npu
        is_not_support_bool = False
    except (ImportError) as e:
        is_not_support_bool = False
    return is_not_support_bool