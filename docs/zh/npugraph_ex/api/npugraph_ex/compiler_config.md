# CompilerConfig类

该类用于构造传入torch.compiler backend的config参数，具体定义如下：

```python
class CompilerConfig(NpuBaseConfig):
    """Set CompilerConfig configuration"""

    def __init__(self):
        self.force_eager = OptionValue(False, [True, False])
        self.use_graph_pool = None
        self.reuse_graph_pool_in_same_fx = OptionValue(True, [True, False])
        self.capture_limit = IntRangeValue(64, 1, INT64_MAX)
        self.clone_input = OptionValue(True, [True, False])
        self.clone_output = OptionValue(False, [True, False])
        self.disable_static_kernel_compile_cache = OptionValue(False, [True, False])
        self.static_kernel_compile = OptionValue(False, [True, False])
        self.frozen_parameter = OptionValue(False, [True, False])
        self.remove_noop_ops = OptionValue(True, [True, False])
        self.remove_cat_ops = OptionValue(True, [True, False])
        self.inplace_pass = OptionValue(True, [True, False])
        self.input_inplace_pass = OptionValue(True, [True, False])
        self.pattern_fusion_pass = OptionValue(True, [True, False])
        self.post_grad_custom_pre_pass = CallableValue(None)
        self.post_grad_custom_post_pass = CallableValue(None)
        self.dump_tensor_data = OptionValue(False, [True, False])
        self.data_dump_stage = OptionValue('optimized', ['original', 'optimized'])
        self.data_dump_dir = MustExistedPathValue("./")
        self._vllm_aclnn_static_kernel_sym_index = IntRangeValue(0, 0, INT64_MAX)
        self._vllm_aclnn_static_kernel_sym_range = IntListValue(None)
        self.super_kernel_optimize = OptionValue(False, [True, False])
        self.super_kernel_optimize_options = DictOptionValue(None)
        self.super_kernel_debug_options = DictOptionValue(None)
        self.deadlock_check = OptionValue(False, [True, False])
        self.capture_error_mode = OptionValue("global", ["global", "thread_local", "relaxed"])
        self.mode = OptionValue("npugraph_ex", ["npugraph_ex"])

        super(CompilerConfig, self).__init__()
        self._fixed_attrs.append("post_grad_custom_pre_pass")
        self._fixed_attrs.append("post_grad_custom_post_pass")
        self._fixed_attrs.append("use_graph_pool")
```
