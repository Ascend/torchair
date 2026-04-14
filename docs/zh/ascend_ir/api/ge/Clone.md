# Clone

## 功能说明

算子Converter中的构图元素，表示一个Clone节点，该节点可实现图上任意单个Tensor的拷贝。

## 函数原型

```python
Clone(x: Tensor, *, dependencies=[], node_name=None) -> Tensor
```

## 参数说明

|参数|输入/输出|说明|
|--|--|--|
|x|输入|待拷贝的Tensor。|
|dependencies|输入|指定Clone节点的执行顺序，标识该节点在图中的拓扑依赖关系，传入列表中为Tensor类型。表明Clone节点在Tensor列表后执行。|
|node_name|输入|自定义Clone节点的名字，默认为None，例如'const_1'，同一张图中节点名不允许重复。|
|*|输入|预留参数项，用于后续功能扩展。|

## 返回值说明

正常情况下，返回一个Tensor，表示图上新增的一个TensorMove节点，否则失败报错。

## 约束说明

无

## 调用示例

```python
import torch_npu, torchair

# 实现Ascend Converter
@register_fx_node_ge_converter(torch.ops.npu.npu_scatter_nd_update.default)
def converter_npu_scatter_nd_update_default(
    x: Tensor,
    indices: Tensor,
    updates: Tensor,
    meta_outputs: TensorSpec = None,
):
    # func: scatter_nd_update(Tensor x, Tensor indices, Tensor updates) -> Tensor
    # copy是对x的clone，indices和updates是外部某算子的输出Tensor 
    copy = torchair.ge.Clone(x, dependencies=[indices, updates], node_name='Const0')
    return scatterNdUpdateFunc(copy, indices, updates)
```
