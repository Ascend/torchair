import pkgutil
import importlib
from typing import Callable
import inspect
import os
import unittest
from pathlib import Path

import torchair


def _discover_path_importables(pkg_pth, pkg_name):
    """Yield all importables under a given path and package.

    This is like pkgutil.walk_packages, but does *not* skip over namespace
    packages. Taken from https://stackoverflow.com/questions/41203765/init-py-required-for-pkgutil-walk-packages-in-python3
    """
    for dir_path, _d, file_names in os.walk(pkg_pth):
        pkg_dir_path = Path(dir_path)

        if pkg_dir_path.parts[-1] == '__pycache__':
            continue

        if all(Path(_).suffix != '.py' for _ in file_names):
            continue

        rel_pt = pkg_dir_path.relative_to(pkg_pth)
        pkg_pref = '.'.join((pkg_name,) + rel_pt.parts)
        yield from (
            pkg_path
            for _, pkg_path, _ in pkgutil.walk_packages(
            (str(pkg_dir_path),), prefix=f'{pkg_pref}.', onerror=lambda x: print(f'Failed import {x}')
        )
        )


SKIP_CHECK_MODULES = [  # Do NOT add new modules to this list
    'torchair._ge_concrete_graph.ge_converter',
    'torchair._ge_concrete_graph.ge_apis',
    'torchair._contrib',
    'torchair._tf_concrete_graph',
]

LEGACY_PUBLIC_APIS = [  # Do NOT add new modules to this list
    'torchair.dynamo_export.CompilerConfig',  # should not be public
    'torchair.dynamo_export.ExportSuccess',  # should not be public
    'torchair.dynamo_export.get_npu_backend',  # should not be public
    'torchair.fx_dumper.Any',  # should not be public
    'torchair.fx_dumper.Argument',  # should not be public
    'torchair.fx_dumper.Callable',  # should not be public
    'torchair.fx_dumper.Dict',  # should not be public
    'torchair.fx_dumper.GraphModule',  # should not be public
    'torchair.fx_dumper.Interpreter',  # should not be public
    'torchair.fx_dumper.List',  # should not be public
    'torchair.fx_dumper.Target',  # should not be public
    'torchair.fx_dumper.Tuple',  # should not be public
    'torchair.fx_dumper.Union',  # should not be public
    'torchair.fx_dumper.contextmanager',  # should not be public
    'torchair.fx_dumper.datetime',  # should not be public
    'torchair.fx_dumper.defaultdict',  # should not be public
    'torchair.fx_summary.Any',  # should not be public
    'torchair.fx_summary.Argument',  # should not be public
    'torchair.fx_summary.Callable',  # should not be public
    'torchair.fx_summary.CompilerConfig',  # should not be public
    'torchair.fx_summary.ConcreteGraph',  # should not be public
    'torchair.fx_summary.ConcreteGraphBase',  # should not be public
    'torchair.fx_summary.Dict',  # should not be public
    'torchair.fx_summary.FakeTensor',  # should not be public
    'torchair.fx_summary.Interpreter',  # should not be public
    'torchair.fx_summary.List',  # should not be public
    'torchair.fx_summary.PathManager',  # should not be public
    'torchair.fx_summary.ShapeEnv',  # should not be public
    'torchair.fx_summary.Target',  # should not be public
    'torchair.fx_summary.Tensor',  # should not be public
    'torchair.fx_summary.Tuple',  # should not be public
    'torchair.fx_summary.Union',  # should not be public
    'torchair.fx_summary.ValuePack',  # should not be public
    'torchair.fx_summary.aot_module_simplified',  # should not be public
    'torchair.fx_summary.core_aten_decompositions',  # should not be public
    'torchair.fx_summary.defaultdict',  # should not be public
    'torchair.fx_summary.detect_fake_mode',  # should not be public
    'torchair.fx_summary.get_decompositions',  # should not be public
    'torchair.fx_summary.is_builtin_callable',  # should not be public
    'torchair.fx_summary.no_dispatch',  # should not be public
    'torchair.inference.cache_compile',  # should be public
    'torchair.inference.set_dim_gears',  # should be public
    'torchair.inference.get_dim_gears',  # should be public
    'torchair.inference.readable_cache',  # should be public
    'torchair.inference.cache_compiler.Callable',  # should not be public
    'torchair.inference.cache_compiler.CompilerConfig',  # should not be public
    'torchair.inference.cache_compiler.Dict',  # should not be public
    'torchair.inference.cache_compiler.List',  # should not be public
    'torchair.inference.cache_compiler.Optional',  # should not be public
    'torchair.inference.cache_compiler.Tuple',  # should not be public
    'torchair.inference.cache_compiler.Union',  # should not be public
    'torchair.inference.cache_compiler.aot_module_simplified',  # should not be public
    'torchair.inference.cache_compiler.contextmanager',  # should not be public
    'torchair.inference.cache_compiler.dataclass',  # should not be public
    'torchair.npu_fx_compiler.Any',  # should not be public
    'torchair.npu_fx_compiler.Argument',  # should not be public
    'torchair.npu_fx_compiler.Callable',  # should not be public
    'torchair.npu_fx_compiler.CompilerConfig',  # should not be public
    'torchair.npu_fx_compiler.ConcreteGraph',  # should not be public
    'torchair.npu_fx_compiler.ConcreteGraphBase',  # should not be public
    'torchair.npu_fx_compiler.Dict',  # should not be public
    'torchair.npu_fx_compiler.Interpreter',  # should not be public
    'torchair.npu_fx_compiler.List',  # should not be public
    'torchair.npu_fx_compiler.NpuFxDumper',  # should not be public
    'torchair.npu_fx_compiler.Target',  # should not be public
    'torchair.npu_fx_compiler.Tuple',  # should not be public
    'torchair.npu_fx_compiler.Union',  # should not be public
    'torchair.npu_fx_compiler.ValuePack',  # should not be public
    'torchair.npu_fx_compiler.add_npu_patch',  # should not be public
    'torchair.npu_fx_compiler.aot_module_simplified',  # should not be public
    'torchair.npu_fx_compiler.aot_module_simplified_joint',  # should not be public
    'torchair.npu_fx_compiler.default_partition',  # should not be public
    'torchair.npu_fx_compiler.detect_fake_mode',  # should not be public
    'torchair.npu_fx_compiler.get_decompositions',  # should not be public
    'torchair.npu_fx_compiler.get_used_syms_in_meta',  # should not be public
    'torchair.npu_fx_compiler.is_builtin_callable',  # should not be public
    'torchair.npu_fx_compiler.is_fake',  # should not be public
    'torchair.npu_fx_compiler.is_sym',  # should not be public
    'torchair.npu_fx_compiler.no_dispatch',  # should not be public
    'torchair.npu_fx_compiler.pretty_error_msg',  # should not be public
    'torchair.npu_fx_compiler.record_function',  # should not be public
    'torchair.npu_fx_compiler.summarize_fx_graph',  # should not be public
    'torchair._utils.Callable',  # should not be public
    'torchair._utils.DispatchKey',  # should not be public
    'torchair._utils.InstructionTranslatorBase',  # should not be public
    'torchair._utils.NNModuleVariable',  # should not be public
    'torchair._utils.OpOverload',  # should not be public
    'torchair._utils.OpOverloadPacket',  # should not be public
    'torchair._utils.Optional',  # should not be public
    'torchair._utils.Tensor',  # should not be public
    'torchair._utils.Tuple',  # should not be public
    'torchair._utils.TupleVariable',  # should not be public
    'torchair._utils.Unsupported',  # should not be public
    'torchair._utils.break_graph_if_unsupported',  # should not be public
    'torchair._utils.lru_cache',  # should not be public
    'torchair._utils.out_wrapper',  # should not be public
    'torchair._utils.raw_batch_norm_func',  # should not be public
    'torchair._utils.reduce',  # should not be public
    'torchair._utils.refs_div',  # should not be public
    'torchair._utils.stack_op',  # should not be public
    'torchair._utils.wraps',  # should not be public
    'torchair._utils.custom_aot_functions.Any',  # should not be public
    'torchair._utils.custom_aot_functions.Callable',  # should not be public
    'torchair._utils.custom_aot_functions.Dict',  # should not be public
    'torchair._utils.custom_aot_functions.List',  # should not be public
    'torchair._utils.custom_aot_functions.Optional',  # should not be public
    'torchair._utils.custom_aot_functions.Tuple',  # should not be public
    'torchair._utils.custom_aot_functions.Union',  # should not be public
    'torchair._utils.custom_aot_functions.aot_export_module',  # should not be public
    'torchair._utils.custom_aot_functions.call_func_with_args',  # should not be public
    'torchair._utils.custom_aot_functions.make_boxed_func',  # should not be public
    'torchair._utils.export_utils.GeGraph',  # should not be public
    'torchair._utils.export_utils.GraphDef',  # should not be public
    'torchair._utils.export_utils.ModelDef',  # should not be public
    'torchair._utils.export_utils.PathManager',  # should not be public
    'torchair._utils.export_utils.TypedDict',  # should not be public
    'torchair._utils.export_utils.compat_as_bytes',  # should not be public
    'torchair._utils.export_utils.dump_graph',  # should not be public
    'torchair._utils.export_utils.torch_type_to_ge_type',  # should not be public
    'torchair.inference.cache_compiler.Callable',  # should not be public
    'torchair.inference.cache_compiler.CompilerConfig',  # should not be public
    'torchair.inference.cache_compiler.Dict',  # should not be public
    'torchair.inference.cache_compiler.List',  # should not be public
    'torchair.inference.cache_compiler.Optional',  # should not be public
    'torchair.inference.cache_compiler.Tuple',  # should not be public
    'torchair.inference.cache_compiler.Union',  # should not be public
    'torchair.inference.cache_compiler.aot_module_simplified',  # should not be public
    'torchair.inference.cache_compiler.contextmanager',  # should not be public
    'torchair.inference.cache_compiler.dataclass',  # should not be public
    'torchair.core._backend.Dict',  # should not be public
    'torchair.core._backend.defaultdict',  # should not be public
    'torchair.core._backend.finalize_graph_engine',  # should not be public
    'torchair.core._backend.initialize_graph_engine',  # should not be public
    'torchair.core._backend.pretty_error_msg',  # should not be public
    'torchair.core._concrete_graph.ABC',  # should not be public
    'torchair.core._concrete_graph.Any',  # should not be public
    'torchair.core._concrete_graph.Argument',  # should not be public
    'torchair.core._concrete_graph.Callable',  # should not be public
    'torchair.core._concrete_graph.CompilerConfig',  # should not be public
    'torchair.core._concrete_graph.Dict',  # should not be public
    'torchair.core._concrete_graph.FakeTensor',  # should not be public
    'torchair.core._concrete_graph.List',  # should not be public
    'torchair.core._concrete_graph.Target',  # should not be public
    'torchair.core._concrete_graph.Tensor',  # should not be public
    'torchair.core._concrete_graph.Tuple',  # should not be public
    'torchair.core._concrete_graph.Union',  # should not be public
    'torchair.core._concrete_graph.abstractmethod',  # should not be public
    'torchair.core.utils.lru_cache',  # should not be public
    'torchair.configs.aoe_config.FileValue',  # should not be public
    'torchair.configs.aoe_config.MustExistedPathValue',  # should not be public
    'torchair.configs.aoe_config.NpuBaseConfig',  # should not be public
    'torchair.configs.aoe_config.OptionValue',  # should not be public
    'torchair.configs.compiler_config.AoeConfig',  # should not be public
    'torchair.configs.compiler_config.DataDumpConfig',  # should not be public
    'torchair.configs.compiler_config.DebugConfig',  # should not be public
    'torchair.configs.compiler_config.DeprecatedValue',  # should not be public
    'torchair.configs.compiler_config.ExperimentalConfig',  # should not be public
    'torchair.configs.compiler_config.ExportConfig',  # should not be public
    'torchair.configs.compiler_config.FusionConfig',  # should not be public
    'torchair.configs.compiler_config.NpuBaseConfig',  # should not be public
    'torchair.configs.compiler_config.OptionValue',  # should not be public
    'torchair.configs.debug_config.NpuBaseConfig',  # should not be public
    'torchair.configs.debug_config.OptionValue',  # should not be public
    'torchair.configs.debug_config.datetime',  # should not be public
    'torchair.configs.dump_config.MustExistedPathValue',  # should not be public
    'torchair.configs.dump_config.NpuBaseConfig',  # should not be public
    'torchair.configs.dump_config.OptionValue',  # should not be public
    'torchair.configs.experimental_config.IntRangeValue',  # should not be public
    'torchair.configs.experimental_config.NpuBaseConfig',  # should not be public
    'torchair.configs.experimental_config.OptionValue',  # should not be public
    'torchair.configs.export_config.MustExistedPathValue',  # should not be public
    'torchair.configs.export_config.NpuBaseConfig',  # should not be public
    'torchair.configs.export_config.OptionValue',  # should not be public
    'torchair.configs.fusion_config.FileValue',  # should not be public
    'torchair.configs.fusion_config.NpuBaseConfig',  # should not be public
    'torchair.configs.option_base.pretty_error_msg',  # should not be public
    'torchair._ge_concrete_graph.auto_generated_ge_raw_ops.Any',  # should not be public
    'torchair._ge_concrete_graph.auto_generated_ge_raw_ops.Callable',  # should not be public
    'torchair._ge_concrete_graph.auto_generated_ge_raw_ops.DataType',  # should not be public
    'torchair._ge_concrete_graph.auto_generated_ge_raw_ops.Dict',  # should not be public
    'torchair._ge_concrete_graph.auto_generated_ge_raw_ops.GraphDef',  # should not be public
    'torchair._ge_concrete_graph.auto_generated_ge_raw_ops.List',  # should not be public
    'torchair._ge_concrete_graph.auto_generated_ge_raw_ops.OpDef',  # should not be public
    'torchair._ge_concrete_graph.auto_generated_ge_raw_ops.Optional',  # should not be public
    'torchair._ge_concrete_graph.auto_generated_ge_raw_ops.Tensor',  # should not be public
    'torchair._ge_concrete_graph.auto_generated_ge_raw_ops.TensorDef',  # should not be public
    'torchair._ge_concrete_graph.auto_generated_ge_raw_ops.TensorDescriptor',  # should not be public
    'torchair._ge_concrete_graph.auto_generated_ge_raw_ops.TensorSpec',  # should not be public
    'torchair._ge_concrete_graph.auto_generated_ge_raw_ops.TensorType',  # should not be public
    'torchair._ge_concrete_graph.auto_generated_ge_raw_ops.Tuple',  # should not be public
    'torchair._ge_concrete_graph.auto_generated_ge_raw_ops.Union',  # should not be public
    'torchair._ge_concrete_graph.auto_generated_ge_raw_ops.auto_convert_to_tensor',  # should not be public
    'torchair._ge_concrete_graph.auto_generated_ge_raw_ops.compat_as_bytes',  # should not be public
    'torchair._ge_concrete_graph.auto_generated_ge_raw_ops.compat_as_bytes_list',  # should not be public
    'torchair._ge_concrete_graph.auto_generated_ge_raw_ops.get_default_ge_graph',  # should not be public
    'torchair._ge_concrete_graph.auto_generated_ge_raw_ops.get_invalid_desc',  # should not be public
    'torchair._ge_concrete_graph.auto_generated_ge_raw_ops.next_unique_name',  # should not be public
    'torchair._ge_concrete_graph.auto_generated_ge_raw_ops.trans_to_list_list_float',  # should not be public
    'torchair._ge_concrete_graph.auto_generated_ge_raw_ops.trans_to_list_list_int',  # should not be public
    'torchair._ge_concrete_graph.compiled_model.Any',  # should not be public
    'torchair._ge_concrete_graph.compiled_model.GeGraph',  # should not be public
    'torchair._ge_concrete_graph.compiled_model.InputProcessing',  # should not be public
    'torchair._ge_concrete_graph.compiled_model.ModelDef',  # should not be public
    'torchair._ge_concrete_graph.compiled_model.TorchNpuGraph',  # should not be public
    'torchair._ge_concrete_graph.compiled_model.initialize_graph_engine',  # should not be public
    'torchair._ge_concrete_graph.compiled_model.unserialize_dict_attr',  # should not be public
    'torchair._ge_concrete_graph.dynamic_output_ops.Any',  # should not be public
    'torchair._ge_concrete_graph.dynamic_output_ops.Callable',  # should not be public
    'torchair._ge_concrete_graph.dynamic_output_ops.DataType',  # should not be public
    'torchair._ge_concrete_graph.dynamic_output_ops.Dict',  # should not be public
    'torchair._ge_concrete_graph.dynamic_output_ops.GraphDef',  # should not be public
    'torchair._ge_concrete_graph.dynamic_output_ops.List',  # should not be public
    'torchair._ge_concrete_graph.dynamic_output_ops.OpDef',  # should not be public
    'torchair._ge_concrete_graph.dynamic_output_ops.Optional',  # should not be public
    'torchair._ge_concrete_graph.dynamic_output_ops.Tensor',  # should not be public
    'torchair._ge_concrete_graph.dynamic_output_ops.TensorDef',  # should not be public
    'torchair._ge_concrete_graph.dynamic_output_ops.TensorDescriptor',  # should not be public
    'torchair._ge_concrete_graph.dynamic_output_ops.TensorSpec',  # should not be public
    'torchair._ge_concrete_graph.dynamic_output_ops.Tuple',  # should not be public
    'torchair._ge_concrete_graph.dynamic_output_ops.Union',  # should not be public
    'torchair._ge_concrete_graph.dynamic_output_ops.auto_convert_to_tensor',  # should not be public
    'torchair._ge_concrete_graph.dynamic_output_ops.compat_as_bytes',  # should not be public
    'torchair._ge_concrete_graph.dynamic_output_ops.compat_as_bytes_list',  # should not be public
    'torchair._ge_concrete_graph.dynamic_output_ops.get_default_ge_graph',  # should not be public
    'torchair._ge_concrete_graph.dynamic_output_ops.next_unique_name',  # should not be public
    'torchair._ge_concrete_graph.dynamic_output_ops.trans_to_list_list_float',  # should not be public
    'torchair._ge_concrete_graph.dynamic_output_ops.trans_to_list_list_int',  # should not be public
    'torchair._ge_concrete_graph.export_config_generete.GraphDef',  # should not be public
    'torchair._ge_concrete_graph.export_config_generete.List',  # should not be public
    'torchair._ge_concrete_graph.export_config_generete.PathManager',  # should not be public
    'torchair._ge_concrete_graph.export_config_generete.Set',  # should not be public
    'torchair._ge_concrete_graph.export_config_generete.get_export_rank_file_name',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.Any',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.Argument',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.Callable',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.CompilerConfig',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.ConcreteGraphBase',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.DataType',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.Dict',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.FakeTensor',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.Format',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.GeGraph',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.GeTensor',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.GraphDef',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.InputProcessing',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.List',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.OpDef',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.OpOverload',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.OpOverloadPacket',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.Placement',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.Support',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.Target',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.Tensor',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.TensorDef',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.TensorDescriptor',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.TensorSpec',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.TorchNpuGraph',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.Tuple',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.Union',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.ValuePack',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.assert_args_checkout',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.attr_scope',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.compat_as_bytes',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.compute_value_of_sym',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.contextmanager',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.convert_to_tensorboard',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.datetime',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.default_ge_graph',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.defaultdict',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.dump_graph',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.force_op_unknown_shape',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.frozen_data_by_constplaceholder',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.generate_config',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.generate_shape_from_tensor',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.generate_sym_exper',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.get_export_file_name',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.get_frozen_flag',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.get_sym_int_value',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.get_used_sym_value_mapping',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.initialize_graph_engine',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.is_fake',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.is_host_data_tensor',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.is_sym',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.make_export_graph',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.optimize_reference_op_redundant_copy',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.optimize_sym_pack',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.remove_dead_data_and_gen_input_mapping',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.replace_data_to_refdata',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.serialize_int_dict_attr',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.serialize_save_graph',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.serialize_str_dict_attr',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.sym_to_ge_dtype',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.torch_type_to_ge_proto_type',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.torch_type_to_ge_type',  # should not be public
    'torchair._ge_concrete_graph.fx2ge_converter.update_op_input_name_from_mapping',  # should not be public
    'torchair._ge_concrete_graph.ge_graph.Any',  # should not be public
    'torchair._ge_concrete_graph.ge_graph.Argument',  # should not be public
    'torchair._ge_concrete_graph.ge_graph.AttrDef',  # should not be public
    'torchair._ge_concrete_graph.ge_graph.Callable',  # should not be public
    'torchair._ge_concrete_graph.ge_graph.Dict',  # should not be public
    'torchair._ge_concrete_graph.ge_graph.Enum',  # should not be public
    'torchair._ge_concrete_graph.ge_graph.GraphDef',  # should not be public
    'torchair._ge_concrete_graph.ge_graph.List',  # should not be public
    'torchair._ge_concrete_graph.ge_graph.ModelDef',  # should not be public
    'torchair._ge_concrete_graph.ge_graph.OpDef',  # should not be public
    'torchair._ge_concrete_graph.ge_graph.Target',  # should not be public
    'torchair._ge_concrete_graph.ge_graph.TensorDescriptor',  # should not be public
    'torchair._ge_concrete_graph.ge_graph.Tuple',  # should not be public
    'torchair._ge_concrete_graph.ge_graph.Union',  # should not be public
    'torchair._ge_concrete_graph.ge_graph.defaultdict',  # should not be public
    'torchair._ge_concrete_graph.ge_graph.no_dispatch',  # should not be public
    'torchair._ge_concrete_graph.ge_ir_by_protoc_3_13_pb2.AttrDef',  # should not be public
    'torchair._ge_concrete_graph.ge_ir_by_protoc_3_13_pb2.GraphDef',  # should not be public
    'torchair._ge_concrete_graph.ge_ir_by_protoc_3_13_pb2.ModelDef',  # should not be public
    'torchair._ge_concrete_graph.ge_ir_by_protoc_3_13_pb2.NamedAttrs',  # should not be public
    'torchair._ge_concrete_graph.ge_ir_by_protoc_3_13_pb2.OpDef',  # should not be public
    'torchair._ge_concrete_graph.ge_ir_by_protoc_3_13_pb2.ShapeDef',  # should not be public
    'torchair._ge_concrete_graph.ge_ir_by_protoc_3_13_pb2.TensorDef',  # should not be public
    'torchair._ge_concrete_graph.ge_ir_by_protoc_3_13_pb2.TensorDescriptor',  # should not be public
    'torchair._ge_concrete_graph.ge_ir_by_protoc_3_19_pb2.AttrDef',  # should not be public
    'torchair._ge_concrete_graph.ge_ir_by_protoc_3_19_pb2.GraphDef',  # should not be public
    'torchair._ge_concrete_graph.ge_ir_by_protoc_3_19_pb2.ModelDef',  # should not be public
    'torchair._ge_concrete_graph.ge_ir_by_protoc_3_19_pb2.NamedAttrs',  # should not be public
    'torchair._ge_concrete_graph.ge_ir_by_protoc_3_19_pb2.OpDef',  # should not be public
    'torchair._ge_concrete_graph.ge_ir_by_protoc_3_19_pb2.ShapeDef',  # should not be public
    'torchair._ge_concrete_graph.ge_ir_by_protoc_3_19_pb2.TensorDef',  # should not be public
    'torchair._ge_concrete_graph.ge_ir_by_protoc_3_19_pb2.TensorDescriptor',  # should not be public
    'torchair._ge_concrete_graph.ge_ir_pb2.AttrDef',  # should not be public
    'torchair._ge_concrete_graph.ge_ir_pb2.GraphDef',  # should not be public
    'torchair._ge_concrete_graph.ge_ir_pb2.ModelDef',  # should not be public
    'torchair._ge_concrete_graph.ge_ir_pb2.OpDef',  # should not be public
    'torchair._ge_concrete_graph.ge_ir_pb2.ShapeDef',  # should not be public
    'torchair._ge_concrete_graph.ge_ir_pb2.TensorDef',  # should not be public
    'torchair._ge_concrete_graph.ge_ir_pb2.TensorDescriptor',  # should not be public
    'torchair._ge_concrete_graph.graph_pass.Any',  # should not be public
    'torchair._ge_concrete_graph.graph_pass.Callable',  # should not be public
    'torchair._ge_concrete_graph.graph_pass.Dict',  # should not be public
    'torchair._ge_concrete_graph.graph_pass.GeGraph',  # should not be public
    'torchair._ge_concrete_graph.graph_pass.GeTensor',  # should not be public
    'torchair._ge_concrete_graph.graph_pass.GraphDef',  # should not be public
    'torchair._ge_concrete_graph.graph_pass.List',  # should not be public
    'torchair._ge_concrete_graph.graph_pass.OpDef',  # should not be public
    'torchair._ge_concrete_graph.graph_pass.Placement',  # should not be public
    'torchair._ge_concrete_graph.graph_pass.TensorDef',  # should not be public
    'torchair._ge_concrete_graph.graph_pass.TensorDescriptor',  # should not be public
    'torchair._ge_concrete_graph.graph_pass.Tuple',  # should not be public
    'torchair._ge_concrete_graph.graph_pass.Union',  # should not be public
    'torchair._ge_concrete_graph.graph_pass.compat_as_bytes',  # should not be public
    'torchair._ge_concrete_graph.graph_pass.defaultdict',  # should not be public
    'torchair._ge_concrete_graph.graph_pass.generate_shape_from_tensor',  # should not be public
    'torchair._ge_concrete_graph.graph_pass.namedtuple',  # should not be public
    'torchair._ge_concrete_graph.graph_pass.torch_type_to_ge_type',  # should not be public
    'torchair._ge_concrete_graph.graph_pass.update_op_input_name_from_mapping',  # should not be public
    'torchair._ge_concrete_graph.utils.Any',  # should not be public
    'torchair._ge_concrete_graph.utils.Callable',  # should not be public
    'torchair._ge_concrete_graph.utils.DataType',  # should not be public
    'torchair._ge_concrete_graph.utils.Dict',  # should not be public
    'torchair._ge_concrete_graph.utils.GraphDef',  # should not be public
    'torchair._ge_concrete_graph.utils.List',  # should not be public
    'torchair._ge_concrete_graph.utils.PathManager',  # should not be public
    'torchair._ge_concrete_graph.utils.Tensor',  # should not be public
    'torchair._ge_concrete_graph.utils.Tuple',  # should not be public
    'torchair._ge_concrete_graph.utils.Union',  # should not be public
    'torchair._ge_concrete_graph.utils.compat_as_bytes',  # should not be public
    'torchair._ge_concrete_graph.utils.is_sym',  # should not be public
    'torchair._ge_concrete_graph.utils.torch_type_to_ge_type',  # should not be public
    'torchair._utils.custom_aot_functions.Any',  # should not be public
    'torchair._utils.custom_aot_functions.Callable',  # should not be public
    'torchair._utils.custom_aot_functions.Dict',  # should not be public
    'torchair._utils.custom_aot_functions.List',  # should not be public
    'torchair._utils.custom_aot_functions.Optional',  # should not be public
    'torchair._utils.custom_aot_functions.Tuple',  # should not be public
    'torchair._utils.custom_aot_functions.Union',  # should not be public
    'torchair._utils.custom_aot_functions.aot_export_module',  # should not be public
    'torchair._utils.custom_aot_functions.call_func_with_args',  # should not be public
    'torchair._utils.custom_aot_functions.make_boxed_func',  # should not be public
    'torchair._utils.export_utils.GeGraph',  # should not be public
    'torchair._utils.export_utils.GraphDef',  # should not be public
    'torchair._utils.export_utils.ModelDef',  # should not be public
    'torchair._utils.export_utils.PathManager',  # should not be public
    'torchair._utils.export_utils.TypedDict',  # should not be public
    'torchair._utils.export_utils.compat_as_bytes',  # should not be public
    'torchair._utils.export_utils.dump_graph',  # should not be public
    'torchair._utils.export_utils.torch_type_to_ge_type',  # should not be public
    'torchair._utils.npu_fx_passes.joint_graph.init_once_fakemode',  # should be public
    'torchair._utils.npu_fx_passes.joint_graph_passes.npu_fusion_attention.filter_nodes',  # should be public
    'torchair._utils.npu_fx_passes.joint_graph_passes.npu_fusion_attention.inference_graph',  # should be public
    'torchair._utils.npu_fx_passes.joint_graph_passes.npu_fusion_attention.register_joint_graph_pass',
    # should be public
    'torchair._utils.npu_fx_passes.joint_graph_passes.npu_fusion_attention.register_replacement',  # should be public
    'torchair._utils.npu_fx_passes.joint_graph_passes.npu_fusion_attention.training_graph',  # should be public
    'torchair._utils.npu_fx_passes.joint_graph_passes.npu_rms_norm.filter_nodes',  # should be public
    'torchair._utils.npu_fx_passes.joint_graph_passes.npu_rms_norm.inference_graph',  # should be public
    'torchair._utils.npu_fx_passes.joint_graph_passes.npu_rms_norm.register_joint_graph_pass',  # should be public
    'torchair._utils.npu_fx_passes.joint_graph_passes.npu_rms_norm.register_replacement',  # should be public
    'torchair._utils.npu_fx_passes.joint_graph_passes.npu_rms_norm.training_graph',  # should be public
    'torchair._utils.npu_fx_passes.joint_graph_passes.npu_rotary_mul.filter_nodes',  # should be public
    'torchair._utils.npu_fx_passes.joint_graph_passes.npu_rotary_mul.inference_graph',  # should be public
    'torchair._utils.npu_fx_passes.joint_graph_passes.npu_rotary_mul.register_joint_graph_pass',  # should be public
    'torchair._utils.npu_fx_passes.joint_graph_passes.npu_rotary_mul.register_replacement',  # should be public
    'torchair._utils.npu_fx_passes.joint_graph_passes.npu_rotary_mul.training_graph',  # should be public
    'torchair._utils.npu_fx_passes.joint_graph_passes.npu_fusion_attention.filter_nodes',  # should be public
    'torchair._utils.npu_fx_passes.joint_graph_passes.npu_fusion_attention.inference_graph',  # should be public
    'torchair._utils.npu_fx_passes.joint_graph_passes.npu_fusion_attention.register_joint_graph_pass',
    # should be public
    'torchair._utils.npu_fx_passes.joint_graph_passes.npu_fusion_attention.register_replacement',  # should be public
    'torchair._utils.npu_fx_passes.joint_graph_passes.npu_fusion_attention.training_graph',  # should be public
    'torchair._utils.npu_fx_passes.joint_graph_passes.npu_rms_norm.filter_nodes',  # should be public
    'torchair._utils.npu_fx_passes.joint_graph_passes.npu_rms_norm.inference_graph',  # should be public
    'torchair._utils.npu_fx_passes.joint_graph_passes.npu_rms_norm.register_joint_graph_pass',  # should be public
    'torchair._utils.npu_fx_passes.joint_graph_passes.npu_rms_norm.register_replacement',  # should be public
    'torchair._utils.npu_fx_passes.joint_graph_passes.npu_rms_norm.training_graph',  # should be public
    'torchair._utils.npu_fx_passes.joint_graph_passes.npu_rotary_mul.filter_nodes',  # should be public
    'torchair._utils.npu_fx_passes.joint_graph_passes.npu_rotary_mul.inference_graph',  # should be public
    'torchair._utils.npu_fx_passes.joint_graph_passes.npu_rotary_mul.register_joint_graph_pass',  # should be public
    'torchair._utils.npu_fx_passes.joint_graph_passes.npu_rotary_mul.register_replacement',  # should be public
    'torchair._utils.npu_fx_passes.joint_graph_passes.npu_rotary_mul.training_graph',  # should be public
]


class TestPublicBindings(unittest.TestCase):
    @staticmethod
    def _is_mod_public(modname):
        split_strs = modname.split('.')
        for elem in split_strs:
            if elem.startswith("_"):
                return False
        return True

    @staticmethod
    def _is_legacy_public(modname):
        for mod in SKIP_CHECK_MODULES:
            if modname.startswith(mod):
                return True

        if modname in LEGACY_PUBLIC_APIS:
            return True
        return False

    def test_correct_module_names(self):
        '''
        An API is considered public, if  its  `__module__` starts with `torchair.`
        and there is no name in `__module__` or the object itself that starts with “_”.
        Each public package should either:
        - (preferred) Define `__all__` and all callables and classes in there must have their
         `__module__` start with the current submodule's path. Things not in `__all__` should
          NOT have their `__module__` start with the current submodule.
        - (for simple python-only modules) Not define `__all__` and all the elements in `dir(submod)` must have their
          `__module__` that start with the current submodule.
        '''
        failure_list = []

        def test_module(modname):
            try:
                if "__main__" in modname:
                    return
                mod = importlib.import_module(modname)
            except Exception:
                # It is ok to ignore here as we have a test above that ensures
                # this should never happen
                return

            if not self._is_mod_public(modname):
                return

                # verifies that each public API has the correct module name and naming semantics

            def check_one_element(elem, modname, mod, *, is_public, is_all):
                if self._is_legacy_public(f'{modname}.{elem}'):
                    return
                obj = getattr(mod, elem)
                if not (isinstance(obj, Callable) or inspect.isclass(obj)):
                    return
                elem_module = getattr(obj, '__module__', None)
                # Only used for nice error message below
                why_not_looks_public = ""
                if elem_module is None:
                    why_not_looks_public = "because it does not have a `__module__` attribute"
                elem_modname_starts_with_mod = elem_module is not None and \
                                               elem_module.startswith(modname) and \
                                               '._' not in elem_module
                if not why_not_looks_public and not elem_modname_starts_with_mod:
                    why_not_looks_public = f"because its `__module__` attribute (`{elem_module}`) is not within the " \
                                           f"torch library or does not start with the submodule where it is " \
                                           f"defined (`{modname}`)"
                # elem's name must NOT begin with an `_` and it's module name
                # SHOULD start with it's current module since it's a public API
                looks_public = not elem.startswith('_') and elem_modname_starts_with_mod
                if not why_not_looks_public and not looks_public:
                    why_not_looks_public = f"because it starts with `_` (`{elem}`)"

                if is_public != looks_public:
                    if is_public:
                        why_is_public = f"it is inside the module's (`{modname}`) `__all__`" if is_all else \
                            "it is an attribute that does not start with `_` on a module that " \
                            "does not have `__all__` defined"
                        fix_is_public = f"remove it from the modules's (`{modname}`) `__all__`" if is_all else \
                            f"either define a `__all__` for `{modname}` or add a `_` at the beginning of the name"
                    else:
                        assert is_all
                        why_is_public = f"it is not inside the module's (`{modname}`) `__all__`"
                        fix_is_public = f"add it from the modules's (`{modname}`) `__all__`"

                    if looks_public:
                        why_looks_public = "it does look public because it follows the rules from the doc above " \
                                           "(does not start with `_` and has a proper `__module__`)."
                        fix_looks_public = "make its name start with `_`"
                    else:
                        why_looks_public = why_not_looks_public
                        if not elem_modname_starts_with_mod:
                            fix_looks_public = "make sure the `__module__` is properly set and points to a submodule " \
                                               f"of `{modname}`"
                        else:
                            fix_looks_public = "remove the `_` at the beginning of the name"

                    failure_list.append(f"# {modname}.{elem}:")
                    is_public_str = "" if is_public else " NOT"
                    failure_list.append(f"  - Is{is_public_str} public: {why_is_public}")
                    looks_public_str = "" if looks_public else " NOT"
                    failure_list.append(f"  - Does{looks_public_str} look public: {why_looks_public}")
                    # Swap the str below to avoid having to create the NOT again
                    failure_list.append("  - You can do either of these two things to fix this problem:")
                    failure_list.append(f"    - To make it{looks_public_str} public: {fix_is_public}")
                    failure_list.append(f"    - To make it{is_public_str} look public: {fix_looks_public}")

            if hasattr(mod, '__all__'):
                public_api = mod.__all__
                all_api = dir(mod)
                for elem in all_api:
                    check_one_element(elem, modname, mod, is_public=elem in public_api, is_all=True)
            else:
                all_api = dir(mod)
                for elem in all_api:
                    if not elem.startswith('_'):
                        check_one_element(elem, modname, mod, is_public=True, is_all=False)

        for modname in _discover_path_importables(str(torchair.__path__[0]), "torchair"):
            test_module(modname)

        test_module('torchair')

        msg = "All the APIs below do not meet our guidelines for public API from " \
              "https://github.com/pytorch/pytorch/wiki/Public-API-definition-and-documentation.\n"
        msg += "Make sure that everything that is public is expected (in particular that the module " \
               "has a properly populated `__all__` attribute) and that everything that is supposed to be public " \
               "does look public (it does not start with `_` and has a `__module__` that is properly populated)."
        msg += "\n\nFull list:\n"
        msg += "\n".join(map(str, failure_list))

        # empty lists are considered false in python
        self.assertTrue(not failure_list, msg)


if __name__ == '__main__':
    unittest.main()
