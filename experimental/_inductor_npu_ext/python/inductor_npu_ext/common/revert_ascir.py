# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import subprocess
import tempfile
from typing import List

import torch
import sympy

from torch._inductor.utils import IndentedBuffer
from inductor_npu_ext.common import logger
from inductor_npu_ext.common.asc_graph import ASCGraph, _Track


class AscDtypes:
    def __getattr__(self, item):
        return getattr(torch, item)


class _TrackedSubgraph:
    def __init__(self, op: _Track):
        self.track = op
        self.y = [getattr(op, f'y{i}') for i in range(100)]

    def __getattr__(self, item):
        if item == "y":
            return self.y
        if item == "x":
            return self.track.x
        raise AttributeError(f"Unknown attribute {item}")


class AscOps:
    def __init__(self):
        self.holder = ASCGraph()
        self.holder.set_current_loop(_Track(''))
        self.graph: HintGraph = HintGraph('')

    def __getattr__(self, item):
        if item == "AscBackend":
            return self.graph.subgraph
        if item == "Data":
            return self.graph.data
        if item == "Output":
            return self.graph.output
        return lambda name, graph: self.holder.add_op(item, name=name)

    def set_graph(self, graph):
        self.graph = graph


class HintGraph:
    def __init__(self, name):
        self.name = name
        self.symbols = set()
        self.args = []
        self.holder = ASCGraph()
        self.holder.set_current_loop(_Track(''))

    @property
    def tiling_def(self):
        code = IndentedBuffer()
        code.splice(f"""
        #include <cstdint>
        #include <iostream>
        struct AutofuseTilingData {{uint32_t block_dim;}};
        """)
        return code.getvalue()

    @property
    def tiling(self):
        used_symbols = sorted(self.symbols)
        signature = [f"int64_t {str(v)}" for v in used_symbols]
        signature.append(f"AutofuseTilingData *tiling_data")
        signature.append(f"uint32_t *workspace_size")
        signature.append(f"uint32_t *block_dim")
        signature.append(f"void *resource_limit")
        debug_code = '\n    '.join(
            [f'std::cerr << "[STUB]Tiling for {self.name} {v} = " << {v} << std::endl;' for v in used_symbols])
        code = IndentedBuffer()
        code.splice(f"""
extern "C" int64_t AutofuseTiling({', '.join(signature)}) {{
    *block_dim = 24;
    *workspace_size = 1024 * 1024;
    {debug_code}
    return 0;
}}
        """)
        return code.getvalue()

    @property
    def kernel(self):
        signature = ["uint32_t block_dim", "void *stream"]
        buf_names = self.args + ['workspace']
        signature.extend([f"void *{v}" for v in buf_names])
        signature.append(f"AutofuseTilingData *tiling_data")
        debug_names = buf_names + ['block_dim', 'stream']
        debug_code = '\n    '.join(
            [f'std::cerr << "[STUB]Launch for {self.name} {v} = " << {v} << std::endl;' for v in debug_names])
        code = IndentedBuffer()
        code.splice(f"""
extern "C" int64_t AutofuseLaunch({', '.join(signature)}) {{
    {debug_code}
    return 0;
}}
        """)
        return code.getvalue()

    def codegen(self):
        return self.tiling_def, self.tiling, self.kernel

    def create_size(self, name):
        self.symbols.add(name)
        return sympy.Symbol(name)

    @staticmethod
    def create_axis(name, _):
        return sympy.Symbol(name)

    @staticmethod
    def infer_dtypes():
        pass

    def data(self, name, _):
        self.args.append(name)
        return self.holder.add_op("Data", name=name)

    def output(self, name, _):
        self.args.append(name)
        return self.holder.add_op("Output", name=name)

    def subgraph(self, name, graph: 'HintGraph', _):
        self.symbols |= graph.symbols

        return _TrackedSubgraph(self.holder.add_op("AscGraph", name=name))


class RevertAscir:
    def __init__(self):
        self.dtypes = AscDtypes()
        self.ops = AscOps()
        self.__dict__['SizeExpr'] = self.size_expr
        self.__dict__['HintGraph'] = self.hint_graph
        self.__dict__['FusedGraph'] = self.hint_graph

    def hint_graph(self, name):
        graph = HintGraph(name)
        self.ops.set_graph(graph)
        return graph

    def size_expr(self, digit):
        return sympy.Symbol(str(digit))


class AutofuserOptions:
    def __init__(self, *args, **kwargs):
        pass


class Autofuser:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def schedule(graph):
        return graph

    @staticmethod
    def codegen(fused_graph):
        return fused_graph.codegen()


class PyAutofuseStub:
    def __init__(self):
        self.ascir = RevertAscir()
        self.__dict__['Autofuser'] = Autofuser
        self.__dict__['AutofuserOptions'] = AutofuserOptions


class AscCompilerStub:
    def __init__(self):
        pass

    @staticmethod
    def jit_compile(tiling_def, host_tiling, op_kernel, args: List[str]):
        output_file = './kernel.so'
        for arg in args:
            if arg.startswith('--output_file='):
                output_file = arg.split('=')[1]
        with tempfile.NamedTemporaryFile(suffix='.cpp', mode='w+', delete=True) as temp_file:
            temp_file.write('\n'.join([tiling_def, host_tiling, op_kernel]))
            temp_file.flush()
            args = ["g++", "-shared", "-std=c++17", "-fPIC", "-Wall", "-O2", "-o", output_file, temp_file.name]
            logger.debug(' '.join(args))
            subprocess.run(args, check=True)
