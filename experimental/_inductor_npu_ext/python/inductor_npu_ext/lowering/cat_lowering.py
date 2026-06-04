"""自定义 cat lowering：所有输入走类型提升、realize，然后包成一个 NPUConcatBuffer。
NPUConcatBuffer 继承 ir.TemplateBuffer，scheduler 通过 codegen_template 派发到 NPUScheduling。
"""
import sympy
import torch

from torch.utils._ordered_set import OrderedSet
from torch._inductor.ir import (
    FixedLayout,
    FlexibleLayout,
    InputsKernel,
    TensorBox,
    TritonTemplateBuffer,
)
from torch._inductor.lowering import register_lowering, to_dtype
from .common import _LoweringGuard, float_dtypes

aten = torch.ops.aten

_LoweringGuard.support(aten.cat, float_dtypes())


def _logical_layout_for_cat_input(x):
    try:
        return x.get_layout()
    except NotImplementedError:
        size = list(x.get_size())
        return FixedLayout(
            device=x.get_device(),
            dtype=x.get_dtype(),
            size=size,
            stride=FlexibleLayout.contiguous_strides(size),
        )


class NPUConcatBuffer(TritonTemplateBuffer):
    """N 路输入沿 axis 拼接的 IR 节点。
    - inputs：每路输入是一个 Buffer/IRNode，concat 轴 size 各自独立
    - axis：concat 维度
    - 不挂 inner_fn / body；scheduler 通过 isinstance 在 codegen_template 里派发到
      NPUKernel.concat。

    继承 TritonTemplateBuffer（不是直接的 TemplateBuffer）的原因：
    inductor scheduler 在 prologue fusion 决策处硬编码
    `if not isinstance(template, ir.TritonTemplateBuffer): return False`，
    必须戴这顶帽子才能让 cat 上游 pointwise 真正融进同一个 NPUKernel。
    我们用不到 Triton 模板的 make_kernel_render / mutated_inputs / subgraph 字段，
    只借它的 isinstance 通过权 + allowed_prologue_inps。
    """

    def __init__(self, *, layout, inputs, input_layouts, axis):
        # 所有 input 都允许做 prologue fuse，这样上游 pointwise 才能并入。
        allowed = OrderedSet(
            inp.get_name() for inp in InputsKernel.unwrap_storage(list(inputs))
        )
        super().__init__(
            layout=layout,
            inputs=inputs,
            make_kernel_render=None,
            allowed_prologue_inps=allowed,
        )
        self.axis = axis
        self.input_layouts = input_layouts

    def is_multi_outputs_template(self) -> bool:
        # NPUConcatBuffer is a pseudo template. Returning True lets concat prologue
        # fusion skip Triton's low-precision template profitability heuristic.
        return True


# 注意：必须直接对 OpOverload 注册而不是对 packet 注册，否则 inductor 的 get_overloads()
# 会因为 "other_fn not in lowerings" 而跳过已被默认 cat 占住的 overloads，
# 导致我们只覆盖了 packet，dispatch 实际命中的 default overload 还是默认实现。
@register_lowering([aten.cat.default, aten.cat.names], type_promotion_kind=None)
def _npu_cat(inputs, dim=0):
    # 过滤 None；保留输入顺序
    inputs = [x for x in inputs if x is not None]
    assert len(inputs) > 0, "aten.cat got empty inputs"

    # ① dtype 自洽：自己 promote，保证进 NPUConcatBuffer 的输入 dtype 全一致
    target_dtype = inputs[0].get_dtype()
    for x in inputs[1:]:
        target_dtype = torch.promote_types(target_dtype, x.get_dtype())
    inputs = [to_dtype(x, target_dtype) if x.get_dtype() != target_dtype else x
              for x in inputs]

    # ② 保留 cat 语义下的 logical layout；unwrap_storage 会丢掉 view/slice 的 size/stride。
    input_layouts = [_logical_layout_for_cat_input(x) for x in inputs]

    # ③ realize 所有输入，让每个上游成为稳定 buffer，方便后续融合判断
    for x in inputs:
        x.realize()

    # ④ 计算输出 layout：concat 轴 = ΣM_i，其它维取第一个输入（lowering 期已保证一致）
    first_size = inputs[0].get_size()
    ndim = len(first_size)
    if dim < 0:
        dim += ndim
    out_size = list(first_size)
    out_size[dim] = sympy.Add(*[x.get_size()[dim] for x in inputs])
    out_stride = FlexibleLayout.contiguous_strides(out_size)

    layout = FixedLayout(
        device=inputs[0].get_device(),
        dtype=target_dtype,
        size=out_size,
        stride=out_stride,
    )

    # ⑤ inputs 解包成 Buffer/ReinterpretView，TemplateBuffer.__init__ 里也会再做一次
    unwrapped = InputsKernel.unwrap_storage(inputs)

    return TensorBox.create(NPUConcatBuffer(
        layout=layout,
        inputs=unwrapped,
        input_layouts=input_layouts,
        axis=dim,
    ))
