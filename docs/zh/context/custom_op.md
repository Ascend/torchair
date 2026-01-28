# custom\_op

## 功能说明

基于算子原型（IR）实现算子Converter函数，完成PyTorch IR与GE IR的转换。

> **说明：** 
>Converter功能是将PyTorch FX图的节点转换为NPU GE图的节点。对于自定义算子，需要先实现对应的Converter，否则会出现报错。

## 函数原型

```python
custom_op(op_type: str, *args, inputs: Optional[Dict[str, Optional[Union['Tensor', List['Tensor']]]]] = None, outputs: Optional[List[Union[str, Tuple[str, int]]]] = None, attrs: Optional[Dict[str, '_Attr']] = None, node_name: Optional[str] = None) -> Tensor
```

## 参数说明


| 参数 | 输入/输出 | 说明 | 是否必选 |
| --- | --- | --- | --- |
| op_type | 输入 | 自定义算子类型，字符串类型，取值与算子原型REG_OP保持一致，例如"MyOp"。 | 是 |
| args | 输入 | 算子原型的参数，可按顺序输入算子的参数后自动解析，但不能和inputs、outputs、attrs同时使用，同时存在时会以args的结果为准。 | 否 |
| inputs | 输入 | 算子原型的输入参数，为Dict类型，取值和顺序需与算子原型REG_OP保持一致。 | 否 |
| outputs | 输入 | 算子原型的输出参数，为List类型。取值和顺序需与算子原型REG_OP保持一致。 | 否 |
| attrs | 输入 | 算子原型的属性参数，为Dict类型，默认值为None。取值和顺序需与算子原型REG_OP保持一致。 | 否 |
| node_name | 输入 | 常量节点名，支持字符串类型，如“a_1”；若不设置，框架会自动生成节点名。<br>同一张图中节点名不允许重复。 | 否 |
| * | 输入 | 预留参数项，用于后续功能扩展。 | 否 |

## 返回值说明

正常情况下，返回输出Tensor，否则失败报错。

## 约束说明

- 本接口仅适用于max-autotune模式。

- attrs一般通过ge.attr.Xxx显式确定类型，当前支持如下类型：

    ```
    "torchair.ge.attr": [
    "Bool",
    "DataType",
    "Float",
    "Int",
    "ListBool",
    "ListDataType",
    "ListFloat",
    "ListInt",
    "ListListFloat",
    "ListListInt",
    "ListStr",
    "Str"]
    ```
- 关于算子原型介绍，请参考[《CANN 图模式开发指南》](https://hiascend.com/document/redirect/CannCommunityGraphguide)中“构建Graph>使用图开发接口全新构建Graph>什么是算子原型”章节。

## 调用示例

> **说明：** 
>简单示例如下，如需深入了解请参考[自定义算子入图](自定义算子入图.md)不同样例中“实现Converter”章节。

假设算子原型定义如下：

```C
REG_OP(MyOp)
 .INPUT(x, TensorType::ALL())                  //x为必选输入，TensorType类型
 .OPTIONAL_INPUT(y, TensorType::ALL())         //y为可选输入，TensorType类型
 .OPTIONAL_INPUT(z, TensorType::ALL())         //z为可选输入，TensorType类型
 .DYNAMIC_INPUT(dy, TensorType::ALL())         //dy为动态输入，TensorType类型，构图时该输入为一组Tensor，数量在构图时确定，数量可以为0
 .REQUIRED_ATTR(rattr, ListInt)                //rattr为必选属性，ListInt类型，属性值为整数数组，无默认值
 .ATTR(oattr, Float)                           //oattr为可选属性，Float类型，默认值为1.0，构图时可不传入，则图上节点该属性值为1.0
 .OUTPUT(m, TensorType::ALL())                 //m为单输出，TensorType类型，节点输出为一个Tensor
 .DYNAMIC_OUTPUT(n, TensorType::ALL())         //（可选，若包含动态输出）若n为动态数量输出，TensorType类型，节点输出为一组Tensor，构图时这组Tensor数量可确定
```

假设构图不包含动态输出，即y传入、z不传、rattr值为\[1,2,3\]、oattr值为1（与默认值相同），接口返回的输出数量与outputs列表长度相同，接口调用支持**传参方式1和传参方式2**。

假设构图包含动态输出，即y传入、z不传、rattr值为\[1,2,3\]、oattr值为1（与默认值相同），**动态输出n的数量为3**，接口返回的输出数量与outputs列表长度相同，接口调用**仅支持传参方式2**。

-   传参方式1：使用args传入所有参数

    ```python
    import torch, torch_npu, torchair
    def my_op(x, y, z, dy, *, rattr, oattr=1.0):
        return torchair.ge.custom_op("MyOp", x, y, z, dy, rattr, oattr)     # 自定义算子类型名称
    t1, t2, t3, t4 = ......                                                 # t1、t2、t3、t4为converter接收到的入参信息
    m = my_op(t1, t2, None, [t3, t4], rattr=[1,2,3])
    ```

-   传参方式2：使用单独的输入/属性/输出参数传入

    ```python
    import torch, torch_npu, torchair
    def my_op(x, y, z, dy, *, rattr, oattr=1.0):
        num_of_n = ...                                               # 对于动态输出，这里要根据输入和属性计算出n的输出个数
        return torchair.ge.custom_op(
            "MyOp",                                                  # 自定义算子类型名称
            inputs={                                                 # 顺序必须与IR定义的输入顺序一致
                "x" : x,
                "y" : y,
                "z" : z,                                             # None表示该输入未传入
                "dy": dy
            },
            attrs={
                "rattr" : torchair.ge.attr.ListInt(rattr),           # 需要通过ge.attr.T显式确定类型
                "oattr" : torchair.ge.attr.Float(oattr)              # 即使与默认值相同也要传入
            },
            outputs=[
                "m",                                                 # 非动态输出写对应IR输出名称即可
                ("n", num_of_n)                                      # 动态输出需要（对应IR输出名称，输出数量）的元组
            ]
        )
    t1, t2, t3, t4 = ......                                          # t1、t2、t3、t4为converter接收到的入参信息
    m, n = my_op(t1, t2, None, [t3, t4], rattr=[1,2,3])
    ```
