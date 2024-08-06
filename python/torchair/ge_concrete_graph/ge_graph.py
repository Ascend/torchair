from torchair.core.utils import logger
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, get_default_ge_graph, next_unique_name, \
compat_as_bytes, compat_as_bytes_list, get_invalid_desc, auto_convert_to_tensor


logger.warning_once(f'The usage of torchair.ge_concrete_graph .* will not be supported in the future,'
                    f' please complete the API switch as soon as possible.')