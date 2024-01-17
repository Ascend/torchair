import functools
import json
import os
from typing import List, Dict

import torch
import sympy
from npu_extension_for_inductor.common.asc_graph import ASCGraph
from npu_extension_for_inductor.common.op_code import OpCode, OpProto
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
        torch_dtype = op.attrs[f'{op.name}.y.dtype']
        assert isinstance(torch_dtype, type(torch.float)), f"Unsupported dtype {torch_dtype}"
        if torch_dtype == torch.float:
            torch_dtype = torch.float32
        self.str_type = str(torch_dtype).replace('torch.', '')
        self.type = self.str_type.replace('float16', 'half').replace('float32', 'float')

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
        return lambda name: self.graph.add_op(item, name=name)


class HintGraph:
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
        return OpProto(json.dumps(proto, indent=4, sort_keys=True))

    @property
    def tiling_def(self):
        code = IndentedBuffer()
        code.splice(f"""
        # include "register/tilingdata_base.h"
        namespace optiling {{
            BEGIN_TILING_DATA_DEF(TilingData)
            TILING_DATA_FIELD_DEF(uint32_t, total_size); // 数据总量
            TILING_DATA_FIELD_DEF(uint32_t, blob_size); // 数据内存块大小
            TILING_DATA_FIELD_DEF(uint32_t, blob_num); // 数据内存块数量
            TILING_DATA_FIELD_DEF(uint32_t, blob_num_per_core); // 单个核心处理的内存块数量 = (blob_num / core_num(=8))
    
            TILING_DATA_FIELD_DEF(uint32_t, tile_size); // 单指令操作数据量
            TILING_DATA_FIELD_DEF(uint32_t, tile_num_per_core); // 单个核心指令数量 = blob_num_per_core * blob_size / tile_size
    
            END_TILING_DATA_DEF;
            REGISTER_TILING_DATA_CLASS({self.name}, TilingData)
        }}
        """)
        return code.getvalue()

    @property
    def tiling(self):
        core_type = os.getenv("NPU_CORE_TYPE", "ai_core-ascend910B1")
        core_type = core_type.split('ai_core-')[-1].lower()
        inputs_def = '\n'.join([op.io_def() for op in self.inputs])
        outputs_def = '\n'.join([op.io_def() for op in self.outputs])
        code = IndentedBuffer()
        code.splice(f"""
        # include "{camel_to_snake(self.name)}_tiling.h"
        # include "register/op_def_registry.h"
        namespace optiling {{
            static ge::graphStatus TilingFunc(gert::TilingContext * ctx) {{
                uint32_t BLOCK_DIM = {HintGraph.BLOCK_DIM};
                uint32_t BLOCK_SIZE = {HintGraph.BLOCK_SIZE};
                uint32_t TILE_SIZE = {HintGraph.TILE_SIZE};
    
                TilingData tiling;
                const auto & syms = ctx->GetInputShape(0)->GetStorageShape();
                uint32_t data_size = 1U;
                for (int i = 0; i < syms.GetDimNum(); ++i) {{
                    data_size *= syms.GetDim(i);
                }}
                tiling.set_total_size(data_size);
                tiling.set_blob_size(BLOCK_SIZE);
                tiling.set_blob_num((data_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
                tiling.set_blob_num_per_core(tiling.get_blob_num() / BLOCK_DIM);
                tiling.set_tile_size(TILE_SIZE);
                tiling.set_tile_num_per_core(tiling.get_blob_num_per_core() * BLOCK_SIZE / TILE_SIZE);
        
                ctx->SetBlockDim({HintGraph.BLOCK_DIM});
                tiling.SaveToBuffer(ctx->GetRawTilingData()->GetData(), ctx->GetRawTilingData()->GetCapacity());
                ctx->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
                
                return ge::GRAPH_SUCCESS;
            }}
        }}
        namespace ops {{
            class {self.name}: public OpDef {{
            public:
                explicit {self.name}(const char * name): OpDef(name) {{
                    {inputs_def}
                    {outputs_def}
                    this->AICore().SetTiling(optiling::TilingFunc);
                    this->AICore().AddConfig("{core_type}");
                }}
            }};
            OP_ADD({self.name});
        }}
        """)
        return code.getvalue()

    @property
    def kernel_args(self):
        args = [f"GM_ADDR {op.name}" for op in self.inputs + self.outputs]
        args += ["GM_ADDR workspace", "GM_ADDR tiling"]
        return args

    @property
    def kernel(self):
        pipe = "tpipe"
        td = "tiling_data"
        buffer_size = f"{td}.blob_num_per_core * {td}.blob_size"
        offset = "offset"
        tile_size = f"{td}.tile_size"
        tile_num = f"{td}.tile_num_per_core"
        inputs = self.inputs[1:]
        init_global_buffer = '\n'.join([op.global_buffer(buffer_size) for op in inputs + self.outputs])
        init_global_pipe = '\n'.join([op.global_pipe(pipe, f"{tile_size}") for op in inputs + self.outputs])

        compute = IndentedBuffer()
        stack = {}
        tiling = Tiling(offset, tile_size, pipe)
        for order, op in list(self.ordered_op.items())[1:]:  # Skip size vars
            op: AscOp = op
            op_vals = []
            for i in op.inputs:
                op_vals.append(stack[i.order])
            stack[order] = op.compute(op_vals, tiling, compute)

        code = IndentedBuffer()
        code.splice(f"""
        # include "kernel_operator.h"
        using namespace AscendC;

        extern "C" __global__ __aicore__ void {camel_to_snake(self.name)}({', '.join(self.kernel_args)}) {{
            GET_TILING_DATA({td}, tiling);
            if (GetBlockIdx() >= GetBlockNum()) return;
    
            TPipe {pipe};
            {init_global_buffer}
            {init_global_pipe}
            for (int i = 0; i < {tile_num}; ++i) {{
                uint32_t {offset} = i * {tile_size};
                {compute.getvalue()}
            }}
        }}
        """)
        return code.getvalue()

    def codegen(self):
        self.inputs: List[AscIO] = []
        self.outputs: List[AscIO] = []
        self.ordered_op: Dict[int, AscOp] = {}

        for op in self._graph.ops:
            assert op.order not in self.ordered_op, f"Duplicate order {op.order} from {op}"
            asc_op = AscOp(op)
            self.ordered_op[op.order] = asc_op
            if op.name in self._graph.inputs:
                self.inputs.append(AscIO(op))
            if op.name in self._graph.outputs:
                self.outputs.append(AscIO(op, is_input=False))

            for k, v in op.attrs.items():
                if isinstance(v, _Track) and isinstance(v.parent, _Op):
                    i_order = v.parent.order
                    assert i_order in self.ordered_op, f"Error order {i_order} for {v.parent}"
                    asc_op.inputs.append(self.ordered_op[i_order])
                    self.ordered_op[i_order].outputs.append(asc_op)

        return OpCode(self.proto, self.tiling_def, self.tiling, self.kernel)

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
        self.__dict__['graph'] = ASCGraph()
        self.__dict__['ops'] = AscOps(self.graph)
        self.__dict__['dtypes'] = AscDtypes()
        self.__dict__['HintGraph'] = lambda name: HintGraph(name, self.graph)
        self.__dict__['SizeExpr'] = lambda syms: functools.reduce(lambda x, y: x * y, syms + [sympy.S.One])
