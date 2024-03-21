from typing import List, Callable, Any, Dict, Tuple, Union, Optional
import itertools

import torch
from torch import nn
import torch.utils._pytree as pytree
from torch._functorch.aot_autograd import aot_export_module, make_boxed_func

try:
    from torch._functorch.aot_autograd import call_func_with_args
except ImportError:
    from torch._functorch._aot_autograd.utils import call_func_at_runtime_with_args as call_func_with_args

try:
    from torch._functorch.aot_autograd import _graph_output_names
except ImportError:
    from torch._functorch._aot_autograd.input_output_analysis import _graph_output_names


def aot_module_simplified_joint(
        mod: nn.Module,
        args,
        compiler: Callable,
        decompositions: Optional[Dict] = None,
        output_loss_index: Optional[int] = None,
) -> nn.Module:
    """
    This is the simplified or low overhead version of aot_module with joint graph.
    For frontends like TorchDynamo, the input functions/modules to AOT are static
    and have unpacked inputs/outputs. This gives us an opportunity to remove the
        (1) pytree overhead to parse inputs/outputs,
        (2) AOT Autograd cache for joint graph,
        (3) Reading of params/buffers in every forward call

    :func:`aot_module_simplified_joint` removes these overheads.
    """

    named_parameters = dict(mod.named_parameters(remove_duplicate=False))
    named_buffers = dict(mod.named_buffers(remove_duplicate=False))

    params_and_buffers = {
        **dict(named_parameters),
        **dict(named_buffers),
    }
    params_and_buffers_flat, _ = pytree.tree_flatten(params_and_buffers)
    params_and_buffers_flat = tuple(params_and_buffers_flat)

    fx_g, signature = aot_export_module(mod, args, trace_joint=True, output_loss_index=output_loss_index)

    user_args_flat, _ = pytree.tree_flatten(args)
    # Retrieve graph output names
    graph_output_names = _graph_output_names(fx_g)

    num_user_fw_outs = len(signature.user_outputs)
    num_mutated_inputs = len(signature.buffers_to_mutate)
    num_fw_outs = num_user_fw_outs + num_mutated_inputs
    backward_output_names = graph_output_names[num_fw_outs:]
    grad_index = itertools.count(0)
    gradients_to_parameters = {
        (next(grad_index) + num_fw_outs): i
        for i, param in enumerate(params_and_buffers_flat)
        if param.requires_grad
    }
    gradients_to_user_inputs = {
        (next(grad_index) + num_fw_outs): (i + len(params_and_buffers_flat))
        for i, user_input in enumerate(user_args_flat)
        if user_input.requires_grad
    }
    if (len(gradients_to_parameters) + len(gradients_to_user_inputs) != len(backward_output_names)):
        raise AssertionError
    parameters = list(named_parameters)
    buffers = list(named_buffers)
    inputs_buffers = list(range(len(parameters), (len(parameters) + len(buffers))))

    start, stop = 0, num_mutated_inputs
    mutated_buffers_to_inputs_buffers = dict(zip(range(start, stop), inputs_buffers))
    start, stop = stop, stop + num_user_fw_outs
    user_outputs = list(range(start, stop))

    full_args = []
    full_args.extend(params_and_buffers_flat)
    full_args.extend(args)

    compiled_fw = compiler(fx_g, full_args)

    if not hasattr(compiled_fw, "_boxed_call"):
        compiled_fw = make_boxed_func(compiled_fw)

    disable_amp = torch._C._is_any_autocast_enabled()

    def runtime_wrapper(args):
        all_outs = call_func_with_args(
            compiled_fw,
            args,
            disable_amp=disable_amp,
        )
        for k, v in mutated_buffers_to_inputs_buffers.items():
            args[v].data = all_outs[k]
        for k, v in gradients_to_parameters.items():
            args[v].grad = all_outs[k]
        for k, v in gradients_to_user_inputs.items():
            args[v].grad = all_outs[k]
        act_ret = [all_outs[i] for i in user_outputs]
        return act_ret

    compiled_fn = runtime_wrapper

    # TODO: There is something deeply wrong here; compiled_fn running with
    # the boxed calling convention, but aot_module_simplified somehow
    # historically returned a function that was not the boxed calling
    # convention.  This should get fixed...
    def forward(*runtime_args):
        full_args = []
        full_args.extend(params_and_buffers_flat)
        full_args.extend(runtime_args)
        return compiled_fn(full_args)

    # Just for convenience
    forward.zero_grad = mod.zero_grad
    forward.named_parameters = mod.named_parameters
    forward.named_buffers = mod.named_buffers

    return forward