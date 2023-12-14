import functools
import itertools
from typing import List

import sympy

import torch  # noqa

from torch._inductor.codegen.wrapper import WrapperCodeGen
from torch._inductor.scheduler import BaseSchedulerNode, BaseScheduling, SchedulerNode
from torch._inductor.virtualized import V
from torch._inductor.codegen.common import (
    IndentedBuffer,
    SizeArg, Kernel, OpOverrides,
)


class NPUOverrides(OpOverrides):
    """Map element-wise ops to NPU Triton backend"""

    def __getattr__(self, item):
        return getattr(V.kernel.graph, item)

    @staticmethod
    def abs(x):
        return V.kernel.graph.abs(x)


class ASCGraph:
    IR = 'ascir'

    def __init__(self, buffer, name="graph"):
        super().__init__()
        self._buffer = buffer
        self._name = name
        self._call = f"{name}()"
        self._index = 0

    def next_op_name(self):
        name = f"ops_{self._index}"
        self._index += 1
        return name

    def __getattr__(self, item):
        return functools.partial(_default_op, graph=self, op=item)

    def str(self):
        return self._buffer.getvalue()


def _default_op(*args, graph: ASCGraph, op, **kwargs):
    arg_str = ", ".join(itertools.chain(map(str, args), [f"{k}={v}" for k, v in kwargs.items()]))
    name = graph.next_op_name()
    graph._buffer.writeline(f"{name} = {ASCGraph.IR}.{op}({arg_str})")
    return name


class NPUKernel(Kernel):
    overrides = NPUOverrides
    _index = 0

    @classmethod
    def next_kernel_name(cls):
        name = f"npu_kernel_{cls._index}"
        cls._index += 1
        return name

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._graph = ASCGraph(self.compute)
        self._kernel = NPUKernel.next_kernel_name()

    @property
    def kernel_name(self):
        return self._kernel

    @property
    def code(self):
        code = IndentedBuffer()
        arg_def, _, _ = self.args.python_argdefs()
        arg_def = ', '.join(arg_def)
        code.writeline(f"def {self._kernel}_graph():")
        with code.indent():
            code.writeline("return None")  # TODO: remove this once ascir ready
            code.writeline("import ascir")
            code.writeline(f"with {ASCGraph.IR}.Graph() as graph:")
            with code.indent():
                code.splice(self._graph.str())
                code.writeline("return graph")
        code.writeline(
            f"{self._kernel}_compiled = npu_compiler.aclnn(npu_fuser.auto_fuse({self._kernel}_graph()))")
        code.writeline(f"def {self._kernel}({arg_def}):")
        with code.indent():
            code.writeline(f"{self._kernel}_compiled({arg_def})")
        return code.getvalue()

    @property
    def graph(self):
        return self._graph

    def load(self, name: str, index: sympy.Expr):
        name = self.args.input(name)
        data = self._graph.input_buffer(f"'{name}'", index)
        return self._graph.load(data)

    def store_reduction(self, name, index, value):
        name = self.args.output(name)
        data = self._graph.output_buffer(f"'{name}'", index)
        return self._graph.store_reduction(data, value)

    def store(self, name, index, value, mode=None):
        name = self.args.output(name)
        data = self._graph.output_buffer(f"'{name}'", index)
        return self._graph.store(data, value)

    def reduction(self, dtype, src_dtype, reduction_type, value):
        return self._graph.reduction(dtype, src_dtype, f"'{reduction_type}'", value)


class NPUScheduling(BaseScheduling):
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler

    def can_fuse_vertical(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        return False

    def can_fuse_horizontal(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        return False

    def group_fn(self, sizes):
        # 这个函数是用来处理迭代大小的，例如将[s0, s1, s2]的sizes变换为[s0, s1*s2]
        return sizes

    def codegen_template(
            self, template_node: SchedulerNode, epilogue_nodes: List[SchedulerNode]
    ):
        raise NotImplementedError()

    def codegen_nodes(self, nodes: List[BaseSchedulerNode]):
        """
        Generate a kernel given a list of pre-fused nodes.
        """
        print(f"{'-' * 5} codegen_nodes {'-' * 5}")
        for i, node in enumerate(nodes):
            print(f"Node {i}: {type(node).__name__}")
            print(f"Ranges: {node.get_ranges()}")
            print(f"Body: \n{node._body.debug_str()}")
            kernel = NPUKernel()

            print(f"{'-' * 5} Start build graph {'-' * 5}")
            ranges = []
            for k, _ in dict(node._body.var_ranges).items():
                ranges.append([sympy.Symbol(f"{k}")])

            with V.set_kernel_handler(kernel), kernel:
                node.codegen(ranges)

            wrapper: WrapperCodeGen = V.graph.wrapper_code
            wrapper.header.splice("\n\n")
            wrapper.header.splice(kernel.code)

            _, call_args, _ = kernel.args.python_argdefs()

            node.mark_run()
            wrapper.generate_kernel_call(
                kernel.kernel_name,
                call_args,
                [None],
                V.graph.scheduler.current_device.index,
                cuda=False,
                triton=False,
            )

    def codegen_sync(self):
        raise NotImplementedError()

    def flush(self):
        pass

    def benchmark_fused_nodes(self, nodes):
        raise NotImplementedError()


class NpuWrapperCodeGen(WrapperCodeGen):
    def __init__(self):
        super().__init__()
        self.header.splice(
            f"""
                from npu_extension_for_inductor import compiler as npu_compiler
                from npu_extension_for_inductor import fuser as npu_fuser
            """
        )
