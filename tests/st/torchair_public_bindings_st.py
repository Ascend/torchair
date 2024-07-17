import pkgutil
import importlib
from typing import Callable
import inspect
import os
import json
import unittest
from pathlib import Path

import torchair


def _discover_path_importables(pkg_pth, pkg_name):
    """Yield all importables under a given path and package.

    This is like pkgutil.walk_packages, but does *not* skip over namespace
    packages.
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


def _read_allow_api_json():
    with open(os.path.join(os.path.dirname(__file__), 'allowlist_for_publicAPI.json')) as allow_file_json:
        allow_api_dict = json.load(allow_file_json)
        allow_api = []
        for key, values in allow_api_dict.items():
            allow_api.extend([f'{key}.{value}' for value in values])
    return allow_api


def _is_alias(public_api_fun_name, public_api):

    if public_api.split('.')[-1] in public_api_fun_name:
        return True
    return False


SKIP_CHECK_MODULES = [
    "torchair.ge_concrete_graph.ge_converter.aten",
    "torchair.ge_concrete_graph.ge_converter.prims",
    "torchair.ge_concrete_graph.auto_generated_ge_raw_ops",
    "torchair.ge_concrete_graph.ge_converter.quantized",
    "torchair.ge_concrete_graph.ge_apis",
    "ge_concrete_graph.ge_converter.builtin_converters",
    "torchair.ge_concrete_graph.ge_converter.c10d_functional.c10d_functional",
    "torchair.ge_concrete_graph.ge_converter.higher_order",
]


LEGACY_PUBLIC_APIS = [
    "torchair.ge_concrete_graph.continguous.DataType",
    "torchair.ge_concrete_graph.continguous.Tensor",
    "torchair.ge_concrete_graph.continguous.ValuePack",
    "torchair.ge_concrete_graph.continguous.force_op_unknown_shape",
    "torchair.ge_concrete_graph.continguous.hint_int",
    "torchair.ge_concrete_graph.continguous.is_host_data_tensor",
    "torchair.ge_concrete_graph.continguous.is_sym",
    "torchair.ge_concrete_graph.continguous_utils.Any",
    "torchair.ge_concrete_graph.continguous_utils.Argument",
    "torchair.ge_concrete_graph.continguous_utils.Callable",
    "torchair.ge_concrete_graph.continguous_utils.Dict",
    "torchair.ge_concrete_graph.continguous_utils.FakeTensor",
    "torchair.ge_concrete_graph.continguous_utils.List",
    "torchair.ge_concrete_graph.continguous_utils.Target",
    "torchair.ge_concrete_graph.continguous_utils.Tensor",
    "torchair.ge_concrete_graph.continguous_utils.Tuple",
    "torchair.ge_concrete_graph.continguous_utils.Union",
    "torchair.ge_concrete_graph.continguous_utils.detect_fake_mode",
    "torchair.ge_concrete_graph.continguous_utils.gen_contiguous_storagesize",
    "torchair.ge_concrete_graph.continguous_utils.gen_contiguous_stride",
    "torchair.ge_concrete_graph.continguous_utils.is_sym",
    "torchair.ge_concrete_graph.continguous_utils.optimize_view",
    "torchair.ge_concrete_graph.dynamic_output_ops.Any",
    "torchair.ge_concrete_graph.dynamic_output_ops.Callable",
    "torchair.ge_concrete_graph.dynamic_output_ops.DataType",
    "torchair.ge_concrete_graph.dynamic_output_ops.Dict",
    "torchair.ge_concrete_graph.dynamic_output_ops.GraphDef",
    "torchair.ge_concrete_graph.dynamic_output_ops.List",
    "torchair.ge_concrete_graph.dynamic_output_ops.OpDef",
    "torchair.ge_concrete_graph.dynamic_output_ops.Optional",
    "torchair.ge_concrete_graph.dynamic_output_ops.Tensor",
    "torchair.ge_concrete_graph.dynamic_output_ops.TensorDef",
    "torchair.ge_concrete_graph.dynamic_output_ops.TensorDescriptor",
    "torchair.ge_concrete_graph.dynamic_output_ops.TensorSpec",
    "torchair.ge_concrete_graph.dynamic_output_ops.Tuple",
    "torchair.ge_concrete_graph.dynamic_output_ops.Union",
    "torchair.ge_concrete_graph.dynamic_output_ops.auto_convert_to_tensor",
    "torchair.ge_concrete_graph.dynamic_output_ops.compat_as_bytes",
    "torchair.ge_concrete_graph.dynamic_output_ops.compat_as_bytes_list",
    "torchair.ge_concrete_graph.dynamic_output_ops.get_default_ge_graph",
    "torchair.ge_concrete_graph.dynamic_output_ops.next_unique_name",
    "torchair.ge_concrete_graph.dynamic_output_ops.trans_to_list_list_float",
    "torchair.ge_concrete_graph.dynamic_output_ops.trans_to_list_list_int",
    "torchair.ge_concrete_graph.export_config_generete.Dict",
    "torchair.ge_concrete_graph.export_config_generete.List",
    "torchair.ge_concrete_graph.export_config_generete.PathManager",
    "torchair.ge_concrete_graph.export_config_generete.Set",
    "torchair.ge_concrete_graph.export_config_generete.get_export_rank_file_name",
    "torchair.ge_concrete_graph.fx2ge_converter.Any",
    "torchair.ge_concrete_graph.fx2ge_converter.Argument",
    "torchair.ge_concrete_graph.fx2ge_converter.Callable",
    "torchair.ge_concrete_graph.fx2ge_converter.CompilerConfig",
    "torchair.ge_concrete_graph.fx2ge_converter.ConcreteGraphBase",
    "torchair.ge_concrete_graph.fx2ge_converter.DataType",
    "torchair.ge_concrete_graph.fx2ge_converter.Dict",
    "torchair.ge_concrete_graph.fx2ge_converter.FakeTensor",
    "torchair.ge_concrete_graph.fx2ge_converter.Format",
    "torchair.ge_concrete_graph.fx2ge_converter.GeGraph",
    "torchair.ge_concrete_graph.fx2ge_converter.GeTensor",
    "torchair.ge_concrete_graph.fx2ge_converter.GraphDef",
    "torchair.ge_concrete_graph.fx2ge_converter.List",
    "torchair.ge_concrete_graph.fx2ge_converter.OpDef",
    "torchair.ge_concrete_graph.fx2ge_converter.OpOverload",
    "torchair.ge_concrete_graph.fx2ge_converter.OpOverloadPacket",
    "torchair.ge_concrete_graph.fx2ge_converter.Placement",
    "torchair.ge_concrete_graph.fx2ge_converter.Support",
    "torchair.ge_concrete_graph.fx2ge_converter.Target",
    "torchair.ge_concrete_graph.fx2ge_converter.Tensor",
    "torchair.ge_concrete_graph.fx2ge_converter.TensorDef",
    "torchair.ge_concrete_graph.fx2ge_converter.TensorDescriptor",
    "torchair.ge_concrete_graph.fx2ge_converter.TensorSpec",
    "torchair.ge_concrete_graph.fx2ge_converter.Tuple",
    "torchair.ge_concrete_graph.fx2ge_converter.Union",
    "torchair.ge_concrete_graph.fx2ge_converter.ValuePack",
    "torchair.ge_concrete_graph.fx2ge_converter.assert_args_checkout",
    "torchair.ge_concrete_graph.fx2ge_converter.attr_scope",
    "torchair.ge_concrete_graph.fx2ge_converter.compat_as_bytes",
    "torchair.ge_concrete_graph.fx2ge_converter.compute_value_of_sym",
    "torchair.ge_concrete_graph.fx2ge_converter.contextmanager",
    "torchair.ge_concrete_graph.fx2ge_converter.convert_to_tensorboard",
    "torchair.ge_concrete_graph.fx2ge_converter.datetime",
    "torchair.ge_concrete_graph.fx2ge_converter.default_ge_graph",
    "torchair.ge_concrete_graph.fx2ge_converter.defaultdict",
    "torchair.ge_concrete_graph.fx2ge_converter.dump_graph",
    "torchair.ge_concrete_graph.fx2ge_converter.force_op_unknown_shape",
    "torchair.ge_concrete_graph.fx2ge_converter.frozen_data_by_constplaceholder",
    "torchair.ge_concrete_graph.fx2ge_converter.generate_config",
    "torchair.ge_concrete_graph.fx2ge_converter.generate_dynamic_dims_option",
    "torchair.ge_concrete_graph.fx2ge_converter.generate_shape_from_tensor",
    "torchair.ge_concrete_graph.fx2ge_converter.generate_sym_exper",
    "torchair.ge_concrete_graph.fx2ge_converter.get_dim_gears",
    "torchair.ge_concrete_graph.fx2ge_converter.get_export_file_name",
    "torchair.ge_concrete_graph.fx2ge_converter.get_frozen_flag",
    "torchair.ge_concrete_graph.fx2ge_converter.get_sym_int_value",
    "torchair.ge_concrete_graph.fx2ge_converter.get_used_sym_value_mapping",
    "torchair.ge_concrete_graph.fx2ge_converter.guard_view_input",
    "torchair.ge_concrete_graph.fx2ge_converter.initialize_graph_engine",
    "torchair.ge_concrete_graph.fx2ge_converter.is_fake",
    "torchair.ge_concrete_graph.fx2ge_converter.is_host_data_tensor",
    "torchair.ge_concrete_graph.fx2ge_converter.is_sym",
    "torchair.ge_concrete_graph.fx2ge_converter.make_export_graph",
    "torchair.ge_concrete_graph.fx2ge_converter.optimize_reference_op_redundant_copy",
    "torchair.ge_concrete_graph.fx2ge_converter.optimize_sym_pack",
    "torchair.ge_concrete_graph.fx2ge_converter.record_pg_to_graph",
    "torchair.ge_concrete_graph.fx2ge_converter.replace_data_to_refdata",
    "torchair.ge_concrete_graph.fx2ge_converter.sym_to_ge_dtype",
    "torchair.ge_concrete_graph.fx2ge_converter.torch_type_to_ge_proto_type",
    "torchair.ge_concrete_graph.fx2ge_converter.torch_type_to_ge_type",
    "torchair.ge_concrete_graph.fx2ge_converter.update_op_input_name_from_mapping",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.Any",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.Callable",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.ContextManager",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.Device",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.Generator",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.Iterator",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.List",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.Literal",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.NamedTuple",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.Number",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.Optional",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.Sequence",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.SymInt",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.Tensor",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.TensorSpec",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.Tuple",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.TypeVar",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.Union",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.conveter_math_floor",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.conveter_operator_add",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.conveter_operator_floordiv",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.conveter_operator_mul",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.conveter_operator_pow",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.conveter_operator_sub",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.conveter_operator_truediv",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.conveter_sym_float",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.dtype_promote",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.is_sym",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.overload",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.register_fx_node_ge_converter",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Any",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Callable",
    "torchair.ge_concrete_graph.ge_converter.prim.device.ContextManager",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Device",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Generator",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Iterator",
    "torchair.ge_concrete_graph.ge_converter.prim.device.List",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Literal",
    "torchair.ge_concrete_graph.ge_converter.prim.device.NamedTuple",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Number",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Optional",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Sequence",
    "torchair.ge_concrete_graph.ge_converter.prim.device.SymInt",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Tensor",
    "torchair.ge_concrete_graph.ge_converter.prim.device.TensorSpec",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Tuple",
    "torchair.ge_concrete_graph.ge_converter.prim.device.TypeVar",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Union",
    "torchair.ge_concrete_graph.ge_converter.prim.device.conveter_prim_device_default",
    "torchair.ge_concrete_graph.ge_converter.prim.device.overload",
    "torchair.ge_concrete_graph.ge_converter.prim.device.register_fx_node_ge_converter",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Any",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Callable",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.ContextManager",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Device",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Generator",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Iterator",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.List",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Literal",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.NamedTuple",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Number",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Optional",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Sequence",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.SymInt",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Tensor",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.TensorSpec",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Tuple",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.TypeVar",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Union",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.conveter_rngprims_philox_rand_default",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.overload",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.register_fx_node_ge_converter",
    "torchair.ge_concrete_graph.ge_ir_by_protoc_3_13_pb2.AttrDef",
    "torchair.ge_concrete_graph.ge_ir_by_protoc_3_13_pb2.GraphDef",
    "torchair.ge_concrete_graph.ge_ir_by_protoc_3_13_pb2.ModelDef",
    "torchair.ge_concrete_graph.ge_ir_by_protoc_3_13_pb2.NamedAttrs",
    "torchair.ge_concrete_graph.ge_ir_by_protoc_3_13_pb2.OpDef",
    "torchair.ge_concrete_graph.ge_ir_by_protoc_3_13_pb2.ShapeDef",
    "torchair.ge_concrete_graph.ge_ir_by_protoc_3_13_pb2.TensorDef",
    "torchair.ge_concrete_graph.ge_ir_by_protoc_3_13_pb2.TensorDescriptor",
    "torchair.ge_concrete_graph.ge_ir_by_protoc_3_19_pb2.AttrDef",
    "torchair.ge_concrete_graph.ge_ir_by_protoc_3_19_pb2.GraphDef",
    "torchair.ge_concrete_graph.ge_ir_by_protoc_3_19_pb2.ModelDef",
    "torchair.ge_concrete_graph.ge_ir_by_protoc_3_19_pb2.NamedAttrs",
    "torchair.ge_concrete_graph.ge_ir_by_protoc_3_19_pb2.OpDef",
    "torchair.ge_concrete_graph.ge_ir_by_protoc_3_19_pb2.ShapeDef",
    "torchair.ge_concrete_graph.ge_ir_by_protoc_3_19_pb2.TensorDef",
    "torchair.ge_concrete_graph.ge_ir_by_protoc_3_19_pb2.TensorDescriptor",
    "torchair.ge_concrete_graph.ge_ir_pb2.AttrDef",
    "torchair.ge_concrete_graph.ge_ir_pb2.GraphDef",
    "torchair.ge_concrete_graph.ge_ir_pb2.ModelDef",
    "torchair.ge_concrete_graph.ge_ir_pb2.OpDef",
    "torchair.ge_concrete_graph.ge_ir_pb2.ShapeDef",
    "torchair.ge_concrete_graph.ge_ir_pb2.TensorDef",
    "torchair.ge_concrete_graph.ge_ir_pb2.TensorDescriptor",
    "torchair.ge_concrete_graph.graph_pass.Any",
    "torchair.ge_concrete_graph.graph_pass.Callable",
    "torchair.ge_concrete_graph.graph_pass.Dict",
    "torchair.ge_concrete_graph.graph_pass.GeTensor",
    "torchair.ge_concrete_graph.graph_pass.GraphDef",
    "torchair.ge_concrete_graph.graph_pass.List",
    "torchair.ge_concrete_graph.graph_pass.OpDef",
    "torchair.ge_concrete_graph.graph_pass.Placement",
    "torchair.ge_concrete_graph.graph_pass.TensorDef",
    "torchair.ge_concrete_graph.graph_pass.TensorDescriptor",
    "torchair.ge_concrete_graph.graph_pass.Tuple",
    "torchair.ge_concrete_graph.graph_pass.Union",
    "torchair.ge_concrete_graph.graph_pass.compat_as_bytes",
    "torchair.ge_concrete_graph.graph_pass.defaultdict",
    "torchair.ge_concrete_graph.graph_pass.generate_shape_from_tensor",
    "torchair.ge_concrete_graph.graph_pass.namedtuple",
    "torchair.ge_concrete_graph.graph_pass.torch_type_to_ge_type",
    "torchair.ge_concrete_graph.graph_pass.update_op_input_name_from_mapping",
    "torchair.ge_concrete_graph.utils.Any",
    "torchair.ge_concrete_graph.utils.Callable",
    "torchair.ge_concrete_graph.utils.DataType",
    "torchair.ge_concrete_graph.utils.Dict",
    "torchair.ge_concrete_graph.utils.GraphDef",
    "torchair.ge_concrete_graph.utils.List",
    "torchair.ge_concrete_graph.utils.PathManager",
    "torchair.ge_concrete_graph.utils.Tensor",
    "torchair.ge_concrete_graph.utils.Tuple",
    "torchair.ge_concrete_graph.utils.Union",
    "torchair.ge_concrete_graph.utils.compat_as_bytes",
    "torchair.ge_concrete_graph.utils.get_default_ge_graph",
    "torchair.ge_concrete_graph.utils.is_sym",
    "torchair.ge_concrete_graph.utils.torch_type_to_ge_type",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.Any",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.Callable",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.ContextManager",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.Device",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.Generator",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.Iterator",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.List",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.Literal",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.NamedTuple",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.Number",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.Optional",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.Sequence",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.SymInt",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.Tensor",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.TensorSpec",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.Tuple",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.TypeVar",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.Union",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.conveter_math_floor",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.conveter_operator_add",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.conveter_operator_floordiv",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.conveter_operator_mul",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.conveter_operator_pow",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.conveter_operator_sub",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.conveter_operator_truediv",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.conveter_sym_float",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.dtype_promote",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.is_sym",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.overload",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.register_fx_node_ge_converter",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Any",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Callable",
    "torchair.ge_concrete_graph.ge_converter.prim.device.ContextManager",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Device",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Generator",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Iterator",
    "torchair.ge_concrete_graph.ge_converter.prim.device.List",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Literal",
    "torchair.ge_concrete_graph.ge_converter.prim.device.NamedTuple",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Number",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Optional",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Sequence",
    "torchair.ge_concrete_graph.ge_converter.prim.device.SymInt",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Tensor",
    "torchair.ge_concrete_graph.ge_converter.prim.device.TensorSpec",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Tuple",
    "torchair.ge_concrete_graph.ge_converter.prim.device.TypeVar",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Union",
    "torchair.ge_concrete_graph.ge_converter.prim.device.conveter_prim_device_default",
    "torchair.ge_concrete_graph.ge_converter.prim.device.overload",
    "torchair.ge_concrete_graph.ge_converter.prim.device.register_fx_node_ge_converter",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Any",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Callable",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.ContextManager",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Device",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Generator",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Iterator",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.List",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Literal",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.NamedTuple",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Number",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Optional",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Sequence",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.SymInt",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Tensor",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.TensorSpec",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Tuple",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.TypeVar",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Union",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.conveter_rngprims_philox_rand_default",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.overload",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.register_fx_node_ge_converter",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Any",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Callable",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.ContextManager",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Device",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Generator",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Iterator",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.List",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Literal",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.NamedTuple",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Number",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Optional",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Sequence",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.SymInt",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Tensor",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.TensorSpec",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Tuple",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.TypeVar",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.Union",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.conveter_rngprims_philox_rand_default",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.overload",
    "torchair.ge_concrete_graph.ge_converter.rngprims.philox_rand.register_fx_node_ge_converter",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Any",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Callable",
    "torchair.ge_concrete_graph.ge_converter.prim.device.ContextManager",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Device",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Generator",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Iterator",
    "torchair.ge_concrete_graph.ge_converter.prim.device.List",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Literal",
    "torchair.ge_concrete_graph.ge_converter.prim.device.NamedTuple",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Number",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Optional",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Sequence",
    "torchair.ge_concrete_graph.ge_converter.prim.device.SymInt",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Tensor",
    "torchair.ge_concrete_graph.ge_converter.prim.device.TensorSpec",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Tuple",
    "torchair.ge_concrete_graph.ge_converter.prim.device.TypeVar",
    "torchair.ge_concrete_graph.ge_converter.prim.device.Union",
    "torchair.ge_concrete_graph.ge_converter.prim.device.conveter_prim_device_default",
    "torchair.ge_concrete_graph.ge_converter.prim.device.overload",
    "torchair.ge_concrete_graph.ge_converter.prim.device.register_fx_node_ge_converter",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allgather.Any",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allgather.Callable",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allgather.Dict",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allgather.Iterator",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allgather.List",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allgather.Optional",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allgather.Tensor",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allgather.Tuple",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allgather.Union",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allgather.conveter_allgather_in_tensor",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allgather.get_group_name_and_record",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allgather.register_decomposition",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allgather.register_fx_node_ge_converter",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce.Any",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce.Callable",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce.Dict",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce.Iterator",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce.Library",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce.List",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce.Optional",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce.Tensor",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce.Tuple",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce.Union",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce.conveter_allreduce",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce.get_group_name_and_record",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce.normalize_reduceop_type",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce.register_fx_node_ge_converter",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce.torch_all_reduce",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_alltoall.Any",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_alltoall.Callable",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_alltoall.DataType",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_alltoall.Dict",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_alltoall.Iterator",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_alltoall.List",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_alltoall.Optional",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_alltoall.Tensor",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_alltoall.Tuple",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_alltoall.Union",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_alltoall.convert_all_to_all_single_npu",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_alltoall.dtype_promote",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_alltoall.get_group_name_and_record",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_alltoall.register_decomposition",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_alltoall.register_fx_node_ge_converter",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_broadcast.Any",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_broadcast.Callable",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_broadcast.Dict",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_broadcast.Iterator",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_broadcast.List",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_broadcast.Optional",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_broadcast.Tuple",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_broadcast.Union",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_broadcast.conveter_broadcast",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_broadcast.get_group_name_and_record",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_broadcast.register_fx_node_ge_converter",
    "torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce.backup_custom_all_reduce",
    "torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce.get_npu_all_reduce",
    "torchair.ge_concrete_graph.continguous.gen_contiguous_storagesize",
    "torchair.ge_concrete_graph.continguous.gen_contiguous_stride",
    "torchair.ge_concrete_graph.continguous.get_sym_node_from_graph",
    "torchair.ge_concrete_graph.continguous.is_contiguous",
    "torchair.ge_concrete_graph.continguous.optimize_view",
    "torchair.ge_concrete_graph.continguous_utils.ViewFakeTensor",
    "torchair.ge_concrete_graph.continguous_utils.guard_view_input",
    "torchair.ge_concrete_graph.continguous_utils.is_view_case",
    "torchair.ge_concrete_graph.continguous_utils.set_fake_mapsym",
    "torchair.ge_concrete_graph.continguous_utils.set_ge_outputs",
    "torchair.ge_concrete_graph.continguous_utils.set_meta_tensor_info",
    "torchair.ge_concrete_graph.dynamic_output_ops.AssistHelp",
    "torchair.ge_concrete_graph.dynamic_output_ops.BarrierTakeMany",
    "torchair.ge_concrete_graph.dynamic_output_ops.Batch",
    "torchair.ge_concrete_graph.dynamic_output_ops.BoostedTreesBucketize",
    "torchair.ge_concrete_graph.dynamic_output_ops.CTCBeamSearchDecoder",
    "torchair.ge_concrete_graph.dynamic_output_ops.ConcatOffset",
    "torchair.ge_concrete_graph.dynamic_output_ops.ConcatOffsetD",
    "torchair.ge_concrete_graph.dynamic_output_ops.Copy",
    "torchair.ge_concrete_graph.dynamic_output_ops.DynamicPartition",
    "torchair.ge_concrete_graph.dynamic_output_ops.DynamicRNNGrad",
    "torchair.ge_concrete_graph.dynamic_output_ops.GetNext",
    "torchair.ge_concrete_graph.dynamic_output_ops.GroupedMatmul",
    "torchair.ge_concrete_graph.dynamic_output_ops.HcomBroadcast",
    "torchair.ge_concrete_graph.dynamic_output_ops.IdentityN",
    "torchair.ge_concrete_graph.dynamic_output_ops.IteratorGetNext",
    "torchair.ge_concrete_graph.dynamic_output_ops.MapPeek",
    "torchair.ge_concrete_graph.dynamic_output_ops.MapUnstage",
    "torchair.ge_concrete_graph.dynamic_output_ops.MapUnstageNoKey",
    "torchair.ge_concrete_graph.dynamic_output_ops.OrderedMapPeek",
    "torchair.ge_concrete_graph.dynamic_output_ops.OrderedMapUnstage",
    "torchair.ge_concrete_graph.dynamic_output_ops.OrderedMapUnstageNoKey",
    "torchair.ge_concrete_graph.dynamic_output_ops.QueueDequeue",
    "torchair.ge_concrete_graph.dynamic_output_ops.QueueDequeueMany",
    "torchair.ge_concrete_graph.dynamic_output_ops.QueueDequeueUpTo",
    "torchair.ge_concrete_graph.dynamic_output_ops.ScatterList",
    "torchair.ge_concrete_graph.dynamic_output_ops.ShapeN",
    "torchair.ge_concrete_graph.dynamic_output_ops.SparseSplit",
    "torchair.ge_concrete_graph.dynamic_output_ops.Split",
    "torchair.ge_concrete_graph.dynamic_output_ops.SplitD",
    "torchair.ge_concrete_graph.dynamic_output_ops.SplitV",
    "torchair.ge_concrete_graph.dynamic_output_ops.SplitVD",
    "torchair.ge_concrete_graph.dynamic_output_ops.StagePeek",
    "torchair.ge_concrete_graph.dynamic_output_ops.SwitchN",
    "torchair.ge_concrete_graph.dynamic_output_ops.Unpack",
    "torchair.ge_concrete_graph.dynamic_output_ops.Unstage",
    "torchair.ge_concrete_graph.export_config_generete.generate_config",
    "torchair.ge_concrete_graph.fx2ge_converter.Converter",
    "torchair.ge_concrete_graph.fx2ge_converter.ExecutorType",
    "torchair.ge_concrete_graph.fx2ge_converter.ExportSuccess",
    "torchair.ge_concrete_graph.fx2ge_converter.GeConcreteGraph",
    "torchair.ge_concrete_graph.fx2ge_converter.SymOutput",
    "torchair.ge_concrete_graph.fx2ge_converter.ViewOfInput",
    "torchair.ge_concrete_graph.fx2ge_converter.declare_supported",
    "torchair.ge_concrete_graph.fx2ge_converter.empty_function",
    "torchair.ge_concrete_graph.fx2ge_converter.get_checkpoint_func",
    "torchair.ge_concrete_graph.fx2ge_converter.get_meta_outputs",
    "torchair.ge_concrete_graph.fx2ge_converter.register_checkpoint_func",
    "torchair.ge_concrete_graph.fx2ge_converter.set_ge_outputs",
    "torchair.ge_concrete_graph.graph_pass.frozen_data_by_constplaceholder",
    "torchair.ge_concrete_graph.graph_pass.get_frozen_flag",
    "torchair.ge_concrete_graph.graph_pass.optimize_reference_op_redundant_copy",
    "torchair.ge_concrete_graph.graph_pass.optimize_sym_pack",
    "torchair.ge_concrete_graph.graph_pass.ref_op_info",
    "torchair.ge_concrete_graph.graph_pass.remove_dead_data_and_reorder_data_index",
    "torchair.ge_concrete_graph.graph_pass.replace_data_to_refdata",
    "torchair.ge_concrete_graph.supported_declaration.BF16",
    "torchair.ge_concrete_graph.supported_declaration.BOOL",
    "torchair.ge_concrete_graph.supported_declaration.F16",
    "torchair.ge_concrete_graph.supported_declaration.F32",
    "torchair.ge_concrete_graph.supported_declaration.F64",
    "torchair.ge_concrete_graph.supported_declaration.I16",
    "torchair.ge_concrete_graph.supported_declaration.I32",
    "torchair.ge_concrete_graph.supported_declaration.I64",
    "torchair.ge_concrete_graph.supported_declaration.I8",
    "torchair.ge_concrete_graph.supported_declaration.Support",
    "torchair.ge_concrete_graph.supported_declaration.T",
    "torchair.ge_concrete_graph.supported_declaration.U8",
    "torchair.ge_concrete_graph.utils.Placement",
    "torchair.ge_concrete_graph.utils.compute_value_of_sym",
    "torchair.ge_concrete_graph.utils.convert_to_pbtxt",
    "torchair.ge_concrete_graph.utils.convert_to_tensorboard",
    "torchair.ge_concrete_graph.utils.dtype_promote",
    "torchair.ge_concrete_graph.utils.dump_graph",
    "torchair.ge_concrete_graph.utils.force_op_unknown_shape",
    "torchair.ge_concrete_graph.utils.generate_shape_from_tensor",
    "torchair.ge_concrete_graph.utils.generate_sym_exper",
    "torchair.ge_concrete_graph.utils.get_graph_input_placements",
    "torchair.ge_concrete_graph.utils.get_group_name_and_record",
    "torchair.ge_concrete_graph.utils.get_sym_int_value",
    "torchair.ge_concrete_graph.utils.get_used_sym_value_mapping",
    "torchair.ge_concrete_graph.utils.get_used_syms_in_meta",
    "torchair.ge_concrete_graph.utils.is_host_data_tensor",
    "torchair.ge_concrete_graph.utils.is_integral_type",
    "torchair.ge_concrete_graph.utils.normalize_reduceop_type",
    "torchair.ge_concrete_graph.utils.record_pg_to_graph",
    "torchair.ge_concrete_graph.utils.specific_op_input_layout",
    "torchair.ge_concrete_graph.utils.specific_op_output_layout",
    "torchair.ge_concrete_graph.utils.update_op_input_name_from_mapping",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allgather.allgather_decomposition",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allgather.allgather_in_different_size",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allgather.allgather_in_same_size",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allgather.allgather_in_tensor_meta",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allgather.allgather_in_tensor_npu",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allgather.allgather_meta",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allgather.allgather_npu",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allgather.check_same_size",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allgather.npu_all_gather_patch_dist",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allgather.npu_allgather_in_tensor_patch_dist",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce.adapter_functional_collectives_all_reduce",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce.allreduce_cpu",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce.allreduce_meta",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce.allreduce_npu",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce.backup_custom_all_reduce",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce.functional_collectives_context",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce.get_npu_all_reduce",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce.npu_all_reduce",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_alltoall.all_to_all_decomposition",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_alltoall.all_to_all_single_decomposition",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_alltoall.npu_all_to_all",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_alltoall.npu_all_to_all_meta",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_alltoall.npu_all_to_all_patch_dist",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_alltoall.npu_all_to_all_single",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_alltoall.npu_all_to_all_single_meta",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_alltoall.npu_all_to_all_single_npu_meta",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_alltoall.npu_all_to_all_single_patch_dist",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_broadcast.broadcast_meta",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_broadcast.broadcast_npu",
    "torchair.ge_concrete_graph.ge_converter.experimental.hcom_broadcast.npu_broadcast_patch_dist",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.is_sym",
    "torchair.ge_concrete_graph.utils.is_sym",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.is_sym",
    "torchair.ge_concrete_graph.fx2ge_converter.torch_type_to_ge_proto_type",
    "torchair.ge_concrete_graph.fx2ge_converter.sym_to_ge_dtype",
    "torchair.ge_concrete_graph.continguous.is_sym",
    "torchair.ge_concrete_graph.continguous_utils.is_sym",
    "torchair.ge_concrete_graph.continguous.is_sym",
    "torchair.ge_concrete_graph.dynamic_output_ops.trans_to_list_list_float",
    "torchair.ge_concrete_graph.fx2ge_converter.assert_args_checkout",
    "torchair.ge_concrete_graph.fx2ge_converter.attr_scope",
    "torchair.ge_concrete_graph.fx2ge_converter.default_ge_graph",
    "torchair.ge_concrete_graph.fx2ge_converter.is_sym",
    "torchair.ge_concrete_graph.fx2ge_converter.sym_to_ge_dtype",
    "torchair.ge_concrete_graph.fx2ge_converter.torch_type_to_ge_proto_type",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.is_sym",
    "torchair.ge_concrete_graph.utils.is_sym",
    "torchair.ge_concrete_graph.ge_converter.builtin_converters.is_sym",
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
        allow_api = _read_allow_api_json()
        public_api_fun_name = set()
        for api in allow_api:
            public_api_fun_name.add(api.split('.')[-1])

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

                if is_public and looks_public:
                    public_api = f"{modname}.{elem}"
                    if public_api not in allow_api and not _is_alias(public_api_fun_name, public_api):
                        failure_list.append(f"# {public_api} is public api, "
                                            f"please add it to allowlist_for_publicAPI.json.")

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
              "pytorch wiki Public-API-definition-and-documentation.\n"
        msg += "Make sure that everything that is public is expected (in particular that the module " \
               "has a properly populated `__all__` attribute) and that everything that is supposed to be public " \
               "does look public (it does not start with `_` and has a `__module__` that is properly populated)."
        msg += "\n\nFull list:\n"
        msg += "\n".join(map(str, failure_list))

        # empty lists are considered false in python
        self.assertTrue(not failure_list, msg)


if __name__ == '__main__':
    unittest.main()
