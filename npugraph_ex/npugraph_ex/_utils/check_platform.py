# 用于判断是否为Ascend910_95后，在执行对应特性
def is_arch35():
    try:
        import torch_npu
        device_name = torch_npu.npu.get_device_name()
        is_arch35_bool = "Ascend910_95" in device_name or "Ascend950" in device_name
    except (ImportError) as e:
        is_arch35_bool = False
    return is_arch35_bool