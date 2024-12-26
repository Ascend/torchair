import functools
import itertools
import json
import os
from typing import List, Dict

import torch
import sympy
from npu_extension_for_inductor.common.asc_graph import ASCGraph
from npu_extension_for_inductor.common.utils import camel_to_snake
from npu_extension_for_inductor.ir import _Track, _Op, _Tensor
from torch._inductor.utils import IndentedBuffer


class Tiling:
    def __init__(self, offset, size, pipe):
        self.offset = offset
        self.size = size
        self.pipe = pipe


class AscOpBase:
    def __init__(self, op: _Op):
        self.op = op
        self.name = op.name
        self.op_type = op.op_type
        self.order = op.order
        self.str_type = 'float16'
        self.type = 'half'

        self.q = f"que_{self.name}"
        self.local = f"local_{self.name}"
        self.gm = f"{self.name}Gm"
        self.inputs: List[AscOpBase] = []
        self.outputs: List[AscOpBase] = []


class AscOp(AscOpBase):
    def __init__(self, op: _Op):
        super().__init__(op)

    def compute(self, vals: List["StackVal"], td: Tiling, code: IndentedBuffer) -> "StackVal":
        code.writeline(f'// code gen for {self.op_type} {self.name}')
        if self.op.op_type in ['Data', 'Output', 'Workspace']:
            return self.global_memory(vals, td, code)

        if hasattr(self, self.op.op_type.lower()):
            return getattr(self, self.op.op_type.lower())(vals, td, code)

        output = self.outputs[0]
        if output.op_type == "Store":
            buf = output.outputs[0]
            code.writeline(f"auto {self.local} = {buf.q}.AllocTensor<{self.type}>();")
            code.writeline(
                f"{self.op_type.capitalize()}({', '.join([str(v) for v in [self.local] + vals + [td.size]])});")
            code.writeline(f"{buf.q}.EnQue({self.local});")
        else:
            code.writeline(f"TBuf<TPosition::VECCALC> {self.q};")
            code.writeline(f"{td.pipe}.InitBuffer({self.q}, {td.size} * sizeof({self.type}));")
            code.writeline(f"auto {self.local} = {self.q}.Get<{self.type}>({td.size});")
            code.writeline(
                f"{self.op_type.capitalize()}({', '.join([str(v) for v in [self.local] + vals + [td.size]])});")
        for v in vals:
            code.writeline(v.free())
        return StackVal(f"{self.local}", ref=len(self.outputs))

    def load(self, vals, td: Tiling, code: IndentedBuffer):
        data = self.inputs[0]
        code.writeline('{')
        with code.indent():
            code.writeline(f"auto {self.local} = {data.q}.AllocTensor<{data.type}>();")
            code.writeline(f"DataCopy({self.local}, {vals[0]}, {td.size});")
            code.writeline(f"{vals[0].free()}")
            code.writeline(f"{data.q}.EnQue({self.local});")
        code.writeline('}')
        code.writeline(f"auto {self.local} = {data.q}.DeQue<{self.type}>();")
        return StackVal(f"{self.local}", free=f"{data.q}.FreeTensor({self.local});", ref=len(self.outputs))

    def store(self, vals, td: Tiling, code: IndentedBuffer):
        buf = self.outputs[0]
        code.writeline('{')
        with code.indent():
            code.writeline(f"auto {self.local} = {buf.q}.DeQue<{self.type}>();")
            code.writeline(f"DataCopy({buf.gm}[{td.offset}], {self.local}, {td.size});")
            code.writeline(f"{buf.q}.FreeTensor({self.local});")
        code.writeline('}')

    def global_memory(self, vals, td: Tiling, code: IndentedBuffer):
        return StackVal(f"{self.gm}[{td.offset}]")


class AscIO(AscOpBase):
    def __init__(self, op: _Op, is_input=True):
        super().__init__(op)
        self.format = 'ND'
        self.is_input = is_input
        self.proto_type = self.str_type.replace('float', 'fp')
        self.def_type = f"ge::DT_{self.str_type.replace('float32', 'float').upper()}"

    def desc(self):
        desc = {}
        desc['name'] = self.name
        desc['param_type'] = 'required'
        desc['format'] = [self.format]
        desc['type'] = [self.proto_type]
        return desc

    def io_def(self):
        return f"""
        this->Input("{self.name}")
        .ParamType(REQUIRED)
        .DataType({{{self.def_type}}})
        .Format({{ge:: FORMAT_{self.format}}})
        .UnknownShapeFormat({{ge:: FORMAT_{self.format}}});
        """

    def global_buffer(self, buffer_size):
        return f"""
        GlobalTensor <{self.type}> {self.name}Gm;
        {self.name}Gm.SetGlobalBuffer((__gm__ {self.type} *){self.name} + GetBlockIdx() * {buffer_size}, {buffer_size});
        """

    def global_pipe(self, pipe, tile_size, buffer_size=2):
        return f"""
        TQue<TPosition::{'VECIN' if self.is_input else 'VECOUT'}, {buffer_size}> {self.q};
        {pipe}.InitBuffer({self.q}, {buffer_size}, {tile_size} * sizeof({self.type}));
        """


class StackVal:
    def __init__(self, val, *, free=None, ref=None):
        if free:
            assert ref is not None, f"Error ref {ref} for {val} with free {free}"
        self._val = val
        self._free = free if free else f'// No need free {self._val}'
        self._ref = ref

    def __str__(self):
        return self._val

    def free(self):
        if self._ref is None:
            return ''
        self._ref -= 1
        if self._ref == 0:
            return self._free
        assert self._ref > 0, f"Error ref {self._ref} for {self._val}"
        return f'// Ref count of {self._val} is now {self._ref}'


class AscDtypes:
    def __getattr__(self, item):
        return getattr(torch, item)


class AscOps:
    def __init__(self, graph: ASCGraph):
        super().__init__()
        self.graph = graph
        self.graph.ops.clear()
        self.graph.inputs.clear()

    def __getattr__(self, item):
        return lambda name, graph: self.graph.add_op(item, name=name)


class ASCHintGraph:
    BLOCK_DIM = 8  # 计算核心数量
    BLOCK_SIZE = 512  # 单个核心数据块大小
    TILE_SIZE = 128  # 单指令操作数据量

    def __init__(self, name, graph: ASCGraph):
        self.name = name
        self._graph = graph
        self.inputs: List[AscIO] = []
        self.outputs: List[AscIO] = []
        self.ordered_op = dict()

    @property
    def proto(self):
        proto = {}
        proto['op'] = self.name
        proto['language'] = 'cpp'
        proto['input_desc'] = [op.desc() for op in self.inputs]
        proto['output_desc'] = [op.desc() for op in self.outputs]
        return proto

    @property
    def tiling_def(self):
        code = IndentedBuffer()
        code.splice(f"""
        #include <iostream>
        struct {self.name}TilingData {{
            uint32_t total_size; // 数据总量
            uint32_t blob_size; // 数据内存块大小
            uint32_t blob_num; // 数据内存块数量
            uint32_t blob_num_per_core; // 单个核心处理的内存块数量 = (blob_num / core_num(=8))
            uint32_t tile_size; // 单指令操作数据量
            uint32_t tile_num_per_core; // 单个核心指令数量 = blob_num_per_core * blob_size / tile_size
        }};
        """)
        return code.getvalue()

    @property
    def tiling(self):
        signature = [f"int64_t {str(v)}" for v in sorted(self.graph.size_vars)]
        signature.append(f"{self.name}TilingData *tiling_data")
        signature.append(f"int64_t *workspace_size")
        signature.append(f"int64_t *block_dim")
        debug_code = '\n'.join(
            [f'std::cerr << "[STUB]{self.name}TilingFunc {v} = " << {v} << std::endl;' for v in self.graph.size_vars])
        code = IndentedBuffer()
        code.splice(f"""
        extern "C" int {self.name}TilingFunc({', '.join(signature)}) {{
            {debug_code}
            return 0;
        }}
        """)
        return code.getvalue()

    @property
    def kernel(self):
        signature = ["int64_t block_dim", "void *stream"]
        buf_names = self.graph.inputs + self.graph.outputs + ['workspace']
        signature.extend([f"int64_t *{v}" for v in buf_names])
        signature.append(f"{self.name}TilingData *tiling_data")
        debug_names = buf_names + ['block_dim', 'stream']
        debug_code = '\n'.join(
            [f'std::cerr << "[STUB]aclrtlaunch_{self.name} {v} = " << {v} << std::endl;' for v in debug_names])
        code = IndentedBuffer()
        code.splice(f"""
        extern "C" int aclrtlaunch_{camel_to_snake(self.name)}({', '.join(signature)}) {{
            {debug_code}
            return 0;
        }}
        """)
        return code.getvalue()

    def codegen(self):
        self.inputs: List[AscIO] = []
        self.outputs: List[AscIO] = []
        self.ordered_op: Dict[int, AscOp] = {}

        inputs = []
        for op in self._graph.ops:
            assert op.order not in self.ordered_op, f"Duplicate order {op.order} from {op}"
            asc_op = AscOp(op)
            self.ordered_op[op.order] = asc_op
            if op.name in self._graph.inputs:
                inputs.append(asc_op)
                self.inputs.append(AscIO(op))
            if op.name in self._graph.outputs:
                self.outputs.append(AscIO(op, is_input=False))

            for k, v in op.attrs.items():
                if isinstance(v, _Track) and isinstance(v.parent, _Op):
                    i_order = v.parent.order
                    assert i_order in self.ordered_op, f"Error order {i_order} for {v.parent}"
                    asc_op.inputs.append(self.ordered_op[i_order])
                    self.ordered_op[i_order].outputs.append(asc_op)

        for data in inputs:
            if not len(data.outputs) > 1:
                continue
            reserved = data.outputs[0]
            for load in data.outputs[1:]:
                for op in load.outputs:
                    op.inputs = [v if v != load else reserved for v in op.inputs]
                    reserved.outputs.append(op)
                self.ordered_op.pop(load.order)
            data.outputs = [reserved]

        return self.tiling_def, self.tiling, self.kernel

    def create_size(self, name):
        self._graph.size(name)
        return sympy.Symbol(name)

    def create_axis(self, name, size_expr):
        self._graph.axis(name, size_expr)
        return sympy.Symbol(name)

    def set_inputs(self, inputs):
        for i in inputs:
            self._graph.input(i.name)

    def set_outputs(self, outputs):
        for o in outputs:
            self._graph.output(o.name)

    @property
    def graph(self):
        for op in self._graph.ops:
            for k, v in op.attrs.items():
                if isinstance(v, _Track) and isinstance(v.parent, _Op):
                    op.attrs[k] = _Tensor(v)
        return self._graph


class RevertAscir(_Track):
    def __init__(self):
        super().__init__('')
        self.__dict__['dtypes'] = AscDtypes()
        self.__dict__['SizeExpr'] = lambda digit: sympy.Symbol(str(digit))
        self.__dict__['HintGraph'] = self.hint_graph

    def hint_graph(self, name):
        graph = ASCGraph()
        graph.set_current_loop(_Track(''))
        self.__dict__['ops'] = AscOps(graph)
        return ASCHintGraph(name, graph)


class AutofuserOptions:
    def __init__(self, *args, **kwargs):
        pass


class Autofuser:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def autofuse(graph):
        return graph

    @staticmethod
    def codegen(graph, fused_graph):
        return None, *graph.codegen()


class AutofuseStub:
    def __init__(self):
        self.ascir = RevertAscir()
        self.__dict__['Autofuser'] = Autofuser
        self.__dict__['AutofuserOptions'] = AutofuserOptions


class AscCompilerStub:
    def __init__(self):
        self.jit_compile = lambda *args, **kwargs: None
