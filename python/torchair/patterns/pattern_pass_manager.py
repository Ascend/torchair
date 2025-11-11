__all__ = []

import functools
import torch

try:
    import torch_npu
except ImportError:
    pass

from torch._inductor.pattern_matcher import register_replacement, PatternMatcherPass
try:
    from torch._inductor.pattern_matcher import fwd_only
except ImportError:
    from torch._inductor.pattern_matcher import inference_graph as fwd_only
from torch._subclasses.fake_tensor import FakeTensorMode


class PatternPassManager:
    def __init__(self):
        if 'pass_name' in PatternMatcherPass.__init__.__code__.co_varnames:
            self.pass_dict = PatternMatcherPass(pass_name="torchair_generic_pattern_pass")
        else:
            self.pass_dict = PatternMatcherPass()

    def _return_true(self):
        return True
    
    def register_pattern(self, search_fn, replace_fn, example_inputs, trace_fn=fwd_only, extra_check=_return_true, search_fn_pattern=None):
        """
        Register a new pattern for matching and replacement.

        Args:
            search_fn: The function to search for in the graph.
            replace_fn: The function to replace the matched pattern with.
            example_inputs: Example inputs for tracing the pattern.
            trace_fn: The function used for tracing the pattern (fwd_only or joint_fwd_bwd, default is fwd_only)
            pass_dict: Dict of passes to register to.
            extra_check: Additional check condition (default is _return_true).
        """
        if hasattr(register_replacement, '__code__') and 'pass_dicts' in register_replacement.__code__.co_varnames:
            register_replacement(
                search_fn=search_fn,
                replace_fn=replace_fn,
                example_inputs=example_inputs,
                trace_fn=trace_fn,
                pass_dicts=self.pass_dict,
                extra_check=extra_check,
                search_fn_pattern=search_fn_pattern
            )
        else:
            register_replacement(
                search_fn=search_fn,
                replace_fn=replace_fn,
                example_inputs=example_inputs,
                trace_fn=trace_fn,
                pass_dict=self.pass_dict,
                extra_check=extra_check,
            )

    def apply_pass(self, fx_graph):
        """
        Apply the registered pattern pass to the given FX graph.

        Args:
            fx_graph: The FX graph to apply the pattern pass to.
        """
        self.pass_dict.apply(fx_graph)

pattern_pass_manager = PatternPassManager()


def _apply_pattern_passes(graph_module: torch.fx.GraphModule):
    # Register pattern replacement rules
    _register_addrmsnormdynamicquant_pattern()
    _register_addrmsnormdynamicquant_pattern2()
    _register_addrmsnormcast_pattern()

    # Apply all registered pattern replacements
    pattern_pass_manager.apply_pass(graph_module)


@functools.lru_cache(None)
def _register_addrmsnormdynamicquant_pattern():
    def search_fn(x1, x2, gamma, smooth_scales):
        y, _, xOut = torch_npu.npu_add_rms_norm(x1, x2, gamma)
        yOut, scale1Out = torch_npu.npu_dynamic_quant(y, smooth_scales=smooth_scales)
        return yOut, xOut, scale1Out
    
    def replace_fn(x1, x2, gamma, smooth_scales):
        yOut, _, xOut, scale1Out, _ = torch_npu.npu_add_rms_norm_dynamic_quant(
            x1, x2, gamma,
            smooth_scale1=smooth_scales,
            output_mask=[True, True],
        )
        return yOut, xOut, scale1Out
    
    fake_mode = FakeTensorMode()
    with fake_mode:
        # sizes/values don't actually matter for initial trace
        # once we get a possible match we re-trace with the actual values and verify the match still holds
        input_tensor = functools.partial(torch.empty, (1, 1, 2), device="npu:0", dtype=torch.float16)
        kwargs_tensor = functools.partial(torch.empty, 2, device="npu:0", dtype=torch.float16)
        pattern_pass_manager.register_pattern(
            search_fn=search_fn,
            replace_fn=replace_fn,
            example_inputs=(input_tensor(), input_tensor(), kwargs_tensor(), kwargs_tensor())
        )


@functools.lru_cache(None)
def _register_addrmsnormdynamicquant_pattern2():
    def search_fn(x1, x2, gamma):
        y, _, xOut = torch_npu.npu_add_rms_norm(x1, x2, gamma)
        yOut, scale1Out = torch_npu.npu_dynamic_quant(y.flatten(0, 1))
        scale1Out_view = scale1Out.view(-1, 1)
        return yOut, xOut, scale1Out_view
    
    def replace_fn(x1, x2, gamma):
        yOut, _, xOut, scale1Out, _ = torch_npu.npu_add_rms_norm_dynamic_quant(
            x1, x2, gamma,
            output_mask=[True, True],
        )
        yOut_flatten = yOut.flatten(0, 1)
        scale1Out_view = scale1Out.view(-1, 1)
        return yOut_flatten, xOut, scale1Out_view
    
    fake_mode = FakeTensorMode()
    with fake_mode:
        # sizes/values don't actually matter for initial trace
        # once we get a possible match we re-trace with the actual values and verify the match still holds
        input_tensor = functools.partial(torch.empty, (1, 1, 2), device="npu:0", dtype=torch.float16)
        kwargs_tensor = functools.partial(torch.empty, 2, device="npu:0", dtype=torch.float16)
        pattern_pass_manager.register_pattern(
            search_fn=search_fn,
            replace_fn=replace_fn,
            example_inputs=(input_tensor(), input_tensor(), kwargs_tensor())
        )


@functools.lru_cache(None)
def _register_addrmsnormcast_pattern():
    def search_fn(x1, x2, gamma):
        y, _, xOut = torch_npu.npu_add_rms_norm(x1, x2, gamma)
        h = y.size(-1)
        y_cast = y.view(-1, h).to(torch.float32)
        return y, xOut, y_cast
    
    def replace_fn(x1, x2, gamma):
        y_cast, y, _, xOut = torch_npu.npu_add_rms_norm_cast(
            x1, x2, gamma
        )
        h = y.size(-1)
        y_cast1 = y_cast.view(-1, h)
        return y, xOut, y_cast1
    
    fake_mode = FakeTensorMode()
    with fake_mode:
        # sizes/values don't actually matter for initial trace
        # once we get a possible match we re-trace with the actual values and verify the match still holds
        input_tensor = functools.partial(torch.empty, (1, 1, 2), device="npu:0", dtype=torch.float16)
        kwargs_tensor = functools.partial(torch.empty, 2, device="npu:0", dtype=torch.float16)
        pattern_pass_manager.register_pattern(
            search_fn=search_fn,
            replace_fn=replace_fn,
            example_inputs=(input_tensor(), input_tensor(), kwargs_tensor())
        )