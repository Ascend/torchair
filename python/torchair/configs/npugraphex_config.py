__all__ = []


class _NpuGraphExConfig:
    """
    Config for NpuGraphEx option
    """
    OPTIONS_TO_CONFIG_MAP = {
        "static_kernel_compile": "experimental_config.aclgraph._aclnn_static_shape_kernel",
        "inplace_pass": "debug.aclgraph.disable_reinplace_inplaceable_ops_pass",
        "input_inplace_pass": "debug.aclgraph.disable_reinplace_input_mutated_ops_pass",
        "remove_noop_ops": "experimental_config.remove_noop_ops",
        "graph_dump_type": "debug.graph_dump.type",
        "graph_dump_path": "debug.graph_dump.path",
        "force_eager": "debug.run_eagerly",
        "pattern_fusion_pass": "experimental_config.pattern_fusion_pass",
        "clone_input": "debug.aclgraph.clone_input",
        "frozen_parameter": "experimental_config.frozen_parameter",
        "post_grad_custom_pre_pass": "post_grad_custom_pre_pass",
        "post_grad_custom_post_pass": "post_grad_custom_post_pass",
        "use_graph_pool": "aclgraph_config.use_custom_pool",
        "reuse_graph_pool_in_same_fx": "debug.aclgraph.disable_mempool_reuse_in_same_fx",
        "capture_limit": "debug.aclgraph.static_capture_size_limit",
        "clone_output": "debug.aclgraph.enable_output_clone",
        
        # More mapping relationships can be extended here
    }

    @staticmethod
    def invert_bool(x):
        if isinstance(x, bool):
            return not x
        if isinstance(x, int) and x in (0, 1):
            return 1 if x == 0 else 0
        return x

    OPTIONS_TO_CONFIG_TRANSFORMATIONS = {
        "inplace_pass": invert_bool.__func__,
        "input_inplace_pass": invert_bool.__func__,
        "reuse_graph_pool_in_same_fx": invert_bool.__func__,
    }
    ALLOWED_OPTIONS = set(OPTIONS_TO_CONFIG_MAP.keys())

    def __init__(self) -> None:
        super(_NpuGraphExConfig, self).__init__()
        

def _process_kwargs_options(config, kwargs):
    """
    Processes the "options" parameter to config.

    Args:
        config: Configuration object to be updated with valid option settings
        kwargs: Parameter dictionary that may contain the "options" entry to be processed

    Raises:
        ValueError: Triggered when the "options" entry contains identifiers not specified 
                    in the predefined mapping rules
    """
    
    # Process target parameter
    if "options" in kwargs:
        options = kwargs["options"]
        for option in options:
            if option not in _NpuGraphExConfig.ALLOWED_OPTIONS:
                raise ValueError(f"Invalid option '{option}', allowed options: {sorted(_NpuGraphExConfig.ALLOWED_OPTIONS)}")

            option_value = options[option]
            if option in _NpuGraphExConfig.OPTIONS_TO_CONFIG_TRANSFORMATIONS:
                transform_func = _NpuGraphExConfig.OPTIONS_TO_CONFIG_TRANSFORMATIONS[option]
                processed_value = transform_func(option_value)
            else:
                processed_value = option_value

            # Parse option and assign config value
            current = config
            parts = _NpuGraphExConfig.OPTIONS_TO_CONFIG_MAP[option].split(".")
            for part in parts[:-1]:
                current = getattr(current, part)
            setattr(current, parts[-1], processed_value)