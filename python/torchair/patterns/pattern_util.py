__all__ = []

import torch
from torchair.core.utils import logger
from torchair.patterns.pattern_pass_manager import _PatternPassManager
from torchair.patterns.add_rms_norm_cast import _register_addrmsnormcast_pattern
from torchair.patterns.add_rms_norm_dynamic_quant import _register_addrmsnormdynamicquant_pattern, _register_addrmsnormdynamicquant_pattern2

pattern_pass_manager = _PatternPassManager()


def _apply_pattern_passes(graph_module: torch.fx.GraphModule, example_inputs=None, config=None):
    if torch.__version__ < "2.6":
        logger.warning(f'The pattern_fusion_pass is unsupported when torch < 2.6 .')
        return

    # Register pattern replacement rules
    _register_addrmsnormdynamicquant_pattern(pattern_pass_manager)
    _register_addrmsnormdynamicquant_pattern2(pattern_pass_manager)
    _register_addrmsnormcast_pattern(pattern_pass_manager)

    # Apply all registered pattern replacements
    pattern_pass_manager.apply_pass(graph_module)