from npu_extension_for_inductor.common.asc_graph import ASCGraph


def codegen_kernel_def(graph: ASCGraph, var_name=None) -> str:
    from ._asc_codegen import codegen_kernel_def as _codegen_kernel_def
    return _codegen_kernel_def(graph, var_name)


def codegen_pgo_def(graph: ASCGraph) -> str:
    from ._asc_codegen import codegen_pgo_def as _codegen_pgo_def
    return _codegen_pgo_def(graph)
