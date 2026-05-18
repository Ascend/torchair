__all__ = []

from typing import Dict, Any, List, Optional
from torch.export import ExportedProgram
import torch.fx
from torchair._ge_concrete_graph.ge_ir_pb2 import GraphDef


class _NodeIR:
    """Ascend IR 节点数据（内部实现）"""
    def __init__(self, source: str, ops: List[Dict[str, Any]], mapping: Dict[str, Any] = None):
        self.source = source
        self.ops = ops
        self.mapping = mapping or {}


class _GeGraphAscend:
    """Ascend IR 转换结果容器（纯数据，内部实现）"""

    def __init__(
        self,
        proto: GraphDef,
        ascend_ir_map: Dict[str, _NodeIR],
        original: ExportedProgram,
        optimized: ExportedProgram
    ):
        self._proto = proto
        self._ascend_ir_map = ascend_ir_map
        self._original = original
        self._optimized = optimized

    @property
    def proto(self) -> GraphDef:
        return self._proto

    @property
    def ascend_ir_map(self) -> Dict[str, _NodeIR]:
        return self._ascend_ir_map

    @property
    def original(self) -> ExportedProgram:
        return self._original

    @property
    def optimized(self) -> ExportedProgram:
        return self._optimized

    def to_json_dict(self) -> dict:
        """将 proto 转为 JSON（用于 epair 序列化）"""
        from google.protobuf.json_format import MessageToDict

        def post_process(data):
            if isinstance(data, bytes):
                return data.hex()
            elif isinstance(data, dict):
                return {k: post_process(v) for k, v in data.items()}
            elif isinstance(data, (list, tuple)):
                return [post_process(item) for item in data]
            return data

        return post_process(MessageToDict(self._proto))


def _optimize_and_convert(
    ep: ExportedProgram,
    config = None,
    custom_decompositions = {}
) -> _GeGraphAscend:
    """
    对 ExportedProgram 执行优化并转换为 Ascend IR（内部实现）

    Args:
        ep: ExportedProgram
        config: 编译器配置（CompilerConfig，可选）
        custom_decompositions: 自定义 decomposition 算子集合

    Returns:
        _GeGraphAscend: 包含 proto + ascend_ir_map + original + optimized

    流程：
    1. decomposition（npu_decomps + custom_decompositions）
    2. optimize_fx
    3. GE Graph 转换（record_ascend_ir=True）
    """
    from torchair.configs.compiler_config import CompilerConfig
    from torchair._utils import add_npu_patch, get_npu_default_decompositions
    from torchair.npu_fx_compiler import _optimize_fx, _valid_graph, _NpuGraphConverter
    from torchair._ge_concrete_graph.fx2ge_converter import GeConcreteGraph
    from torchair._utils.graph_transform_observer import GraphTransformObserver
    from torch.fx.passes.infra.pass_base import PassResult

    if config is None:
        config = CompilerConfig()

    npu_decomps = get_npu_default_decompositions()
    add_npu_patch(npu_decomps, config)

    if custom_decompositions:
        for k, v in custom_decompositions.items():
            npu_decomps[k] = v

    ep = ep.run_decompositions(decomp_table=npu_decomps)
    args, kwargs = ep.example_inputs if ep.example_inputs else ((), {})
    flat_args = ep._graph_module_flat_inputs(args, kwargs)

    ge_graph = GeConcreteGraph(config, name="ascend_ir_graph", record_ascend_ir=True)

    def _optimize_pass(gm):
        observer = GraphTransformObserver(gm, flat_args, config)
        optimized_gm = _optimize_fx(gm, config, observer)
        _valid_graph(optimized_gm)
        ge_graph.save_fx_graph(optimized_gm)
        return PassResult(graph_module=optimized_gm, modified=True)

    optimized_ep = ep._transform_do_not_use(_optimize_pass)

    converter = _NpuGraphConverter(optimized_ep.graph_module, graph=ge_graph)

    converter.run(*flat_args)

    return _GeGraphAscend(
        proto=ge_graph.graph._proto,
        ascend_ir_map=ge_graph.get_ascend_ir_map(),
        original=ep,
        optimized=optimized_ep
    )
