from torchair.core.utils import logger
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter


logger.warning_once(f'The usage of torchair.ge_concrete_graph .* will not be supported in the future,'
                    f' please complete the API switch as soon as possible.')