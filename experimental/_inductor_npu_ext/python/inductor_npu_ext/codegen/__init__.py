from inductor_npu_ext.common.asc_graph import FusedASCGraph


def codegen_kernel_def(graph: FusedASCGraph) -> str:
    from ._asc_codegen import codegen_kernel_def as _codegen_kernel_def
    return _codegen_kernel_def(graph)


def codegen_cpp_wrapper(graph: FusedASCGraph) -> str:
    from ._asc_codegen import codegen_cpp_wrapper as _codegen_cpp_wrapper
    return _codegen_cpp_wrapper(graph)
