__all__ = []

import torch
from torchair.core.utils import logger
from torchair.patterns.pattern_pass_manager import _pattern_manager
from torchair.patterns.add_rms_norm_cast import _register_addrmsnormcast_pattern
from torchair.patterns.add_rms_norm_dynamic_quant import _register_addrmsnormdynamicquant_pattern, \
    _register_addrmsnormdynamicquant_pattern2
from torchair.patterns.batch_matmul_transpose import _register_batchmatmultranspose_patterns
from torchair._utils.graph_utils import debug_compare_fx_graphs
from torchair.patterns.add_rms_norm_quant import _register_addrmsnormquant_patterns

pattern_pass_manager = _pattern_manager()


@debug_compare_fx_graphs(pass_name="apply_pattern_passes")
def _apply_pattern_passes(graph_module: torch.fx.GraphModule, example_inputs=None, config=None):
    if torch.__version__ < "2.6":
        logger.warning(f'The pattern_fusion_pass is unsupported when torch < 2.6 .')
        return

    # Register pattern replacement rules
    _register_addrmsnormdynamicquant_pattern(pattern_pass_manager)
    _register_addrmsnormdynamicquant_pattern2(pattern_pass_manager)
    _register_addrmsnormcast_pattern(pattern_pass_manager)
    if config.mode.value != "max-autotune":
        _register_batchmatmultranspose_patterns(pattern_pass_manager)
        _register_addrmsnormquant_patterns(pattern_pass_manager)

    # Set stream labels for all nodes before pattern pass
    from torchair._utils.graph_utils import add_stream_label_to_node_meta
    add_stream_label_to_node_meta(graph_module)

    # Apply all registered pattern replacements
    pattern_pass_manager.apply_pass(graph_module)
