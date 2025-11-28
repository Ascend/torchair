__all__ = ["register_replacement"]

from torch._inductor.pattern_matcher import Match, PatternMatcherPass
from torch._inductor.pattern_matcher import register_replacement as register
try:
    from torch._inductor.pattern_matcher import fwd_only
except ImportError:
    from torch._inductor.pattern_matcher import inference_graph as fwd_only

_global_pattern_pass_manager = None


def _return_true(match: Match):
    return True


class _PatternPassManager:
    def __init__(self):
        if 'pass_name' in PatternMatcherPass.__init__.__code__.co_varnames:
            self.pass_dict = PatternMatcherPass(pass_name="torchair_generic_pattern_pass")
        else:
            self.pass_dict = PatternMatcherPass()

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
        if hasattr(register, '__code__') and 'pass_dicts' in register.__code__.co_varnames:
            register(
                search_fn=search_fn,
                replace_fn=replace_fn,
                example_inputs=example_inputs,
                trace_fn=trace_fn,
                pass_dicts=self.pass_dict,
                extra_check=extra_check,
                search_fn_pattern=search_fn_pattern
            )
        else:
            register(
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


def _pattern_manager():
    global _global_pattern_pass_manager
    if _global_pattern_pass_manager is None:
        _global_pattern_pass_manager = _PatternPassManager()
    return _global_pattern_pass_manager


def register_replacement(search_fn, replace_fn, example_inputs, trace_fn=fwd_only, extra_check=_return_true, search_fn_pattern=None):
    _global_pattern_pass_manager = _pattern_manager()
    _global_pattern_pass_manager.register_pattern(search_fn=search_fn,
                replace_fn=replace_fn,
                example_inputs=example_inputs,
                trace_fn=trace_fn,
                extra_check=extra_check,
                search_fn_pattern=search_fn_pattern)