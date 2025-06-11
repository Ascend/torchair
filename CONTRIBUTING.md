# 关于CI编译

工程目录下的build.sh用于对接CI编译，但是您也可以通过其在本地执行编译和UT/ST测试。

- 查看帮助
    ```shell
    ./build.sh -h
    ```

- 编译安装包
    ```shell
    ./build.sh -c
    ```
    编译完成后，会在`output/`目录下生成名为torchair-{version}-py3-none-any.whl的安装包文件。用户需要使用pip3手动安装。
    ```
    pip3 install output/torchair-{version}-py3-none-any.whl
    ```
    如需要保存安装日志，可在pip3 install命令后面加上参数`--log <PATH>`，并对您指定的目录`<PATH>`做好权限管控。

- 编译并安装
    ```shell
    ./build.sh -i
    ```
    编译完成后，会在`output/`目录下生成名为torchair-{version}-py3-none-any.whl的安装包文件，并且自动调用pip3进行安装。

- 执行UT测试
    > 本地执行UT测试需要设置环境变量`ASCEND_CUSTOM_PATH`,将其指定至Ascend sdk的安装路径（指定至ai_cann_x86目录）
    ```shell
    ./build.sh -u
    ```

- 执行ST测试
    > 本地执行ST测试需要设置环境变量`ASCEND_CUSTOM_PATH`,将其指定至Ascend sdk的安装路径（指定至ai_cann_x86目录）
    ```shell
    ./build.sh -s
    ```

- 其它参数

    `-j[n]` : 指定编译CANN使用的线程数量为n，其中n默认为8。

    `-v` : 启动make的VERBOSE=1选项，用于显示具体的编译链接命令。

    `-g path` : 将编译过程中使用的编译器指定为path路径下的g++编译器。

- 卸载

    torchair的卸载只需要执行命令：

    ```
    pip3 uninstall torchair
    ```
    如需要保存卸载日志，可在pip3 uninstall命令后面加上参数`--log <PATH>`，并对您指定的目录`<PATH>`做好权限管控。

- 查看覆盖率报告
    UT或ST执行通过后，会在coverage目录下生成覆盖率文件coverage.info，如果您想要查看覆盖率报告，可以执行如下命令
    ```shell
    genhtml coverage.info -o coverage_report
    ```

# 基于昇腾NPU和torch_npu使用TorchAir示例

> 使能图模式之前，请先将模型迁移至昇腾NPU，确保模型能够在单算子模式下正确执行，具体请参考[PyTorch模型迁移和训练指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha003/devguide/moddevg/ptmigr/AImpug_0001.html)。

```python
# 导入torchair框架
import torch
import torch_npu
import torchair as tng

# 定义模型model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, z):
        return torch.add(x, y*z)

model = Model()

# torchair提供了一些额外的编译选项，您可以在此处指定，也可以设置为None使用默认选项
config = CompilerConfig()

# 从torchair框架获取npu提供的 默认backend
npu_backend = tng.get_npu_backend(compiler_config=config)
torch.compile(model, backend=npu_backend)

# 使用torchair的backend去调用compile接口编译模型
model = torch.compile(model, backend=npu_backend, dynamic=False)

# 执行编译后的model
in1 = torch.randn(4, 1).float().npu()
in2 = torch.randn(4, 4).float().npu()
in3 = torch.randn(4, 4).int().npu()
graph_result = model(in1, in2, in3)

# 打印执行结果
print(graph_result)
```
> TorchAir在更多不同场景下的应用，请参考[示例代码](https://gitee.com/ascend/torchair/tree/master/examples).


# converter补齐
1. **确定需要补齐的converter**
    
    torchair提供配置项config.debug.fx_summary开关来确定FX图中涉及需要补齐的converter，您可以通过如下方式来配置
    ```python
    config = torchair.CompilerConfig()
    config.debug.fx_summary.type = "csv"
    npu_backend = torchair.get_npu_backend(compiler_config=config)

    torch.compile(model, backend=npu_backend)
    ```
    执行后，会生成summary_{timestamp}.csv文件，您可以通过excel等工具来查看。
    通过导出的csv文件，您可以看到当前模型fx图中涉及的所有converter
    - 对于`支持状态`为`未支持`的converter，您需要在`_ge_concrete_graph/ge_converter`目录下对应文件中补齐实现
    > 当前已经为aten下的op提供了一个壳子实现（固定抛出未支持的异常），您应当在此基础上补齐实现

    - 对于`支持状态`为`未注册`的converter，您需要在`_ge_concrete_graph/ge_converter`目录下对应文件中新增注册并补齐实现

    - 对于`支持状态`为`部分支持`的converter，您需要查看位于`_ge_concrete_graph/ge_converter`目录下的实现，并根据输入数据列决定是否需要补齐场景实现

2. **实现converter**
    
    您可以参考`_ge_concrete_graph/ge_converter`目录下的实现，实现对应的converter。
    我们以torch.ops.aten.add.Tensor的converter实现为例说明converter实现时的一些细节：
    ```python
    @declare_supported([
        Support(F32(2, 2), F32(2, 2)),
        Support(F32(2, 2), F32(2, 1)),
        Support(F32(2, 2), F16(2, 1)),
        Support(F32(2, 2), F16(2, 2), alpha=2),
        Support(F32(2, 2), 2.0),
        Support(F32(2, 2), 2),
        Support(F32(2, 2), 2, alpha=2.0),
    ])
    @register_fx_node_ge_converter(torch.ops.aten.add.Tensor)
    def conveter_aten_add_Tensor(
            self: Tensor,
            other: Tensor,
            *,
            alpha: Union[Number, Tensor] = 1,
            meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
        """ NB: aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor """
        if not isinstance(alpha, Tensor) and alpha == 1:
            # just for better permance
            self, other = dtype_promote(self, other, target_dtype = meta_outputs.dtype)
            return ge.Add(self, other)
        else:
            self, other, alpha = dtype_promote(self, other, alpha, target_dtype = meta_outputs.dtype)
            return ge.AxpyV2(self, other, alpha)
    ```
    1. **声明converter支持的场景**
        
        您应该在开头声明converter需要支持的全部场景，支持场景应该穷举aten.add.Tensor所支持的全部传参方式，
        需要注意，`不应当有不支持的场景`，如果您发现有无法支持的传参方式，需要在converter实现中抛出异常。
        ```python
        @declare_supported([
            Support(F32(2, 2), F32(2, 2)), # 支持基础的torch.add()
            Support(F32(2, 2), F32(2, 1)), # 支持f32类型间的广播
            Support(F32(2, 2), F16(2, 1)), # 支持f32与f16类型的广播
            Support(F32(2, 2), F16(2, 2), alpha=2), # 支持带alpha入参场景
            Support(F32(2, 2), 2.0), # 支持与浮点常量的加法
            Support(F32(2, 2), 2), # 支持f32类型与整型常量的加法
            Support(F32(2, 2), 2, alpha=2.0), # 支持带alpha入参场景
        ])
        ```
        当您实现了您的converter后，我们会根据您声明支持的场景，生成对应的测试用例，您可以通过如下方式来测试您的converter是否正确：
        > 需要在真实NPU环境测试，并确保已经实现了对应的converter及正确安装了torchair
        ```shell
        python3 smoke/converter_test.py
        ```

        可以通过入参控制只测试满足某个prefix的converter，如下所示：
        ```shell
        python3 smoke/converter_test.py aten.add.Tensor
        ```

    2. **converter的函数签名**
        
        ```python
        from torchair.ge._ge_graph import Tensor, TensorSpec

        @register_fx_node_ge_converter(torch.ops.aten.add.Tensor)
        def conveter_aten_add_Tensor(
                self: Tensor,
                other: Tensor,
                *,
                alpha: Union[Number, Tensor] = 1,
                meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
            """ NB: aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor """
        ```
        我们逐行解释下上述代码片段：
        ```python
        from torchair.ge._ge_graph import Tensor, TensorSpec
        ```
        文件的开头从GE graph的文件中导入了Tensor和TensorSpec，这两个类分别表示GE图上的Tensor和TensorSpec。
        需要特别注意的是，Tensor和TensorSpec绝不是运行时的真实数据，你只能从Tensor上获得dtype和rank信息，而不能从Tensor上获得shape和数据。

        ```python
        @register_fx_node_ge_converter(torch.ops.aten.add.Tensor)
        ```
        表示要为torch.ops.aten.add.Tensor注册converter函数，实现将fx图上的aten.add.Tensor节点，转换为GE图上的节点。对应的实现函数则是conveter_aten_add_Tensor。
        ```python
        self: Tensor,
        other: Tensor,
        ```
        表示fx图上的aten.add.Tensor节点的两个输入self和other`对应的GE图输入`，这两个输入都是GE图上的Tensor类型。
        ```python
        *,
        ```
        这是python3的语法，表示后面的参数都是关键字参数，即必须使用参数名来传参。

        ```python
        alpha: Union[Number, Tensor] = 1,
        ```
        表示fx图上的aten.add.Tensor节点的alpha入参，其类型为Number(字面值)或者GE图上的Tensor，其默认值为1。

        ```python
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
        ```
        可以发现conveter_aten_add_Tensor函数的入参与torch.ops.aten.add.Tensor几乎完全一致，但是多一个名为meta_outputs的入参外完全一致。
        meta_outputs是GE graph下定义的`TensorSpec`或`List[TensorSpec]`(aten节点有动态输出时)类型，TensorSpec是对一个节点输出的描述信息。meta_outputs是在指示converter，最终你应该输入一个什么样的Tensor。meta_outputs由原始fx节点输出的aten.Tensor转换而来，包含dtype和rank信息。
        什么时候应该使用meta_outputs呢，典型的场景包括：
        - 通过meta_outputs可以确定输出数量，有些aten的节点的输出数量是不确定的，比如aten.split，会根据输入shape的不同得到不同的输出数量。
        - 确定输出的dtype，用于类型提升，比如aten.add，会根据输入的dtype的不同得到不同的输出dtype，这时候通过meta_outputs上的dtype可以精确地确定应该把输入提升成何种类型。


    3. **实现converter**
        
        alpha的输入可能是Tensor或者Number，我们需要对其进行判断，如果是Number且为1，则可以直接使用ge.Add来实现，否则需要使用ge.AxpyV2来实现，下面的代码片段展示了aten.add.Tensor的converter实现：
        ```python
        @register_fx_node_ge_converter(torch.ops.aten.add.Tensor)
        def conveter_aten_add_Tensor(
                self: Tensor,
                other: Tensor,
                *,
                alpha: Union[Number, Tensor] = 1,
                meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
            """ NB: aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor """
            if not isinstance(alpha, Tensor) and alpha == 1:
                # just for better permance
                self, other = dtype_promote(self, other, target_dtype = meta_outputs.dtype)
                return ge.Add(self, other)
            else:
                self, other, alpha = dtype_promote(self, other, alpha, target_dtype = meta_outputs.dtype)
                return ge.AxpyV2(self, other, alpha)
        ```
        主要关注其中的dtype_promote（类型提升）函数，这个函数接收任意个输入，并返回等量的输出。其作用是将传入的多个输出，提升为target_dtype指定的类型。
        例如测试用例TestInput(F32(2, 2), 2)，将一个f32类型与一个int类型的常量相加，如果不执行dtype_promote，最终生成的图上，会出现一个ge::Add节点，
        其两个输入分别为F32和INT64类型，当前这种图无法编译通过（图编译不支持类型提升）。
        dtype_promote（类型提升）的关键，在于converter实现时，需要根据算子语义，将需要保证类型一致的GE图节点输入进行类型提升，以保证最终生成的GE图可以编译通过。

    4. **注意事项**
        
        需要特别注意，所实现的converter必须支持动态shape，不应该试图从输入的Tensor上获取任何shape信息，Tensor也不会提供任何shape信息。
        > 如果您的converter依赖shape才能工作，这通常意味着实现错误，或者没有选择正确的Ascend IR映射。

    5. **新增自定义算子入图注册**
        
        此功能可以让用户注册新增自定义算子，增加自定义算子的入图能力。
        > 注意：在开发自定义算子入图前，需要确保自定义算子已经在torch框架中完成注册。自定义算子指：区别与原生torch算子，为了实现用户自定义计算逻辑而开发注册的算子。
        > 在C++中注册自定义算子参考：https://pytorch.org/tutorials/advanced/dispatcher.html

        新增算子入图步骤
        > 1 自定义算子在torch框架中注册（假如：您已完成自定义算子注册，请忽略此步骤，下面代码是为了完成示例）。
        ```python
        import torch
        from torch.library import Library, impl
        # 实例化torch.library，完成"custom_op"自定义算子在"npu_define"的namespace的注册，并通过define方法完成schema格式的算子原型定义。
        # 注意："DEF"方式注册不允许namespace重名，python和c++注册不能使用同一个namespace。
        m = Library("npu_define", "DEF")
        m.define("custom_op(Tensor input1, Tensor input2) -> Tensor")

        # 通过impl装饰器完成算子实现的注册，示例中使用custom_op这个算子实现Add的功能，"PrivateUse1"表示注册在npu后端。
        @impl(m, "custom_op", "PrivateUse1")
        def plug_custom_op(
                x: torch.Tensor,
                y: torch.Tensor,
        ):
            return x + y
        ```
        > 2 向torch注册自定义算子meta后端实现，用来完成图模式下的shape推导。
        ```python

        # '@impl(m, "custom_op", "Meta")'表示: 通过Library实例m，为"custom_op"这个自定义算子注册Meta实现。
        # 注：若自定义算子原型的注册发生在C++，无法直接获得实例化的m。使用'm = Library("npu_define", "IMPL", "Meta")'方式获取实例化m,"IMPL"表示为任何操作符添加实现。
        # 'def custom_op_meta(x, y)'为算子的infershape函数，其入参需要保持与自定义算子一致。
        # 注：此处需要保证输出tensor的device为meta，torch.empty_like(x)可以保证输出与输入x的device相同，皆为meta，其他生成输出tensor的方式需要注意是否需要显式指定device为meta。
        @impl(m, "custom_op", "Meta")
        def custom_op_meta(x, y):
            return torch.empty_like(x)
        ```

        > 3 codegen生成ge构图api

        假设`npu.custom_op`转换为`ge.Add`这个GE IR，生成`ge.Add`接口步骤如下(注：ge.Add为演示使用，一般新增自定义算子会有对于新增的GE IR)：

        （1）将REG_OP算子原型放置到codegen/custom_op/custom_reg_op.h文件中，替换原来示例的REG_OP；
        ```c++
        #include "graph/operator_reg.h"

        namespace ge {
        REG_OP(Add)
            .INPUT(x1, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16,
                                DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                                DT_COMPLEX64, DT_STRING}))
            .INPUT(x2, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16,
                                DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                                DT_COMPLEX64, DT_STRING}))
            .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16,
                                DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                                DT_COMPLEX64, DT_STRING}))
            .OP_END_FACTORY_REG(Add)
        }
        ```
        （2）执行编译命令
        ```make
        mkdir build
        cd build
        cmake ..
        make generate_ge_raw_custom_ops
        ```
        生成的ge.api函数在codegen/custom_op/auto_generated_ge_raw_custom_ops.py文件中, 内容如下所示
        ```
        from typing import Any, Dict, List, Tuple, Union, Callable, Optional
        from torchair.ge._ge_graph import auto_convert_to_tensor, TensorType
        from torchair.ge import Tensor, DataType, attr
        from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef


        # This api is auto-generated from IR Add
        @auto_convert_to_tensor([False, False], [False, False])
        def Add(x1: Tensor, x2: Tensor, *, dependencies=[], node_name=None):
            """REG_OP(Add)\n
        .INPUT(x1, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128, DT_COMPLEX64, DT_STRING}))\n
        .INPUT(x2, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128, DT_COMPLEX64, DT_STRING}))\n
        .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128, DT_COMPLEX64, DT_STRING}))\n
        """

            # process inputs
            inputs = {
                "x1": x1,
                "x2": x2,
            }

            # process attrs
            attrs = {
            }

            # process outputs
            outputs = [
            "y",
            ]

            return ge_op(
                op_type="Add",
                inputs=inputs,
                attrs=attrs,
                outputs=outputs,
                ir=IrDef("Add") \
                .input("x1", "DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128, DT_COMPLEX64, DT_STRING") \
                .input("x2", "DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128, DT_COMPLEX64, DT_STRING") \
                .output("y" , "DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128, DT_COMPLEX64, DT_STRING")
            )
        ```
        （3）将您生成的文件内容拷贝至工程目录python/torchair/_ge_concrete_graph/ge_converter/custom中合适的文件中。

        > 4 向torchair注册自定义算子的converter，完成自定义算子的torch IR到CANN软件图中的GE IR的转化(此步骤为npu入图独有的操作)。

        converter如何开发参考本文章的前序章节。
        需要保证converter调用装饰器`@register_fx_node_ge_converter(torch.ops.npu_define.custom_op.default)`，完成converter注册。其中
        `torch.ops.npu_define.custom_op.default`为自定义算子生成的python函数的函数签名。
        ```python
        @register_fx_node_ge_converter(torch.ops.npu_define.custom_op.default)
        def conveter_custom_op(
                input1: Tensor,
                input2: Tensor,
                out: Tensor = None,
                meta_outputs: Any = None):
            # 将输入的数据类型提升至与输出一致
            input1, input2 = dtype_promote(input1, input2, target_dtype=meta_outputs.dtype)
            # 调用ge构图api
            return ge.Add(input1, input2)
        ```

        至此完成全部自定义算子入图适配工作，您可以运行参考用例中的示例验证。
        > 注意： 您在开发您自己的REG_OP(xxx)的自定义算子时，需要向GE注册infershape函数，否则执行时会出错。

3. **导出gegraph**
    
    torchair提供配置项config.debug.graph_dump开关来导出gegraph，您可以通过如下方式来配置
    
    ```python
    config = torchair.CompilerConfig()
    config.debug.graph_dump.type = 'txt' # ['txt', 'pbtxt', 'py']
    npu_backend = torchair.get_npu_backend(compiler_config=config)

    model = torch.compile(model, backend=npu_backend)
    ```

    执行后，会生成dynamo_{timestamp}.{graph_dump.type}文件，当前支持 ```['txt', 'pbtxt', 'py']``` 三种导出方式
    - 导出的txt文件是cann最终接收到的torchair的构图结果，为protobuf格式，您可以通过vscode等查看
    - 导出的pbtxt文件是可以被tensorboard读取的构图结果，您可以通过tensorboard等查看
    - 导出的py文件是torch代码经由converter转化后的GEIR代码，支持运行，您可以通过vscode等查看

    > **需要注意:**
    > 本段接口可以导出图的相关信息，请用户加强对相关数据的保护。
    > 请在接口的功能完成之后及时关闭相关选项。

# lite导出场景下覆盖主线的converter
**背景：**
在lite导出的场景下，想使用不同于当前主线版本的算子，可以通过后注册相同的aten_op的converter来覆盖当前主线上已注册的converter。

**步骤：**
在python/torchair/_ge_concrete_graph/ge_converter/lite，新增需要覆盖的相同aten_op的Python文件，本次示例用add算子演示, 覆盖主线的converter，则新增add.py文件如下：

```python
from typing import (
    Union,
)
import torch
from torch.types import Number
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec

@register_fx_node_ge_converter(torch.ops.aten.add.Tensor)
def conveter_aten_add_Tensor(
    self: Tensor,
    other: Tensor,
    *,
    alpha: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None
):
    return ge.Add(self, other)
```

**注意：**
存放于该路径python/torchair/_ge_concrete_graph/ge_converter/lite下的converter，需要打开配置enable_lite_export才能生效。

```python
from torchair.configs.compiler_config import CompilerConfig
config = CompilerConfig()
config.export.experimental.enable_lite_export = True
```

# 调整interpolate下的默认decomposition
  **动机：**
    pytorch中存在一些默认的decomposition旨在将算子decompose为更小的[算子集合](https://pytorch.org/docs/stable/torch.compiler_ir.html)，例如aten.upsample_nearest2d算子会在torch.compile的模式下被拆解为一系列小算子。
    该行为有时不符合预期：
    - 拆解为小算子后，对于特定的backend可能造成性能的裂化。
    - 在export场景，不希望被拆解。
    因此提供一种方式，在torchair中屏蔽原生pytorch中的某些decomposition。
  **decomposition实现原理：**
    1、pytorch使用装饰器[register_decomposition](https://github.com/pytorch/pytorch/blob/v2.1.0/torch/_decomp/__init__.py#L99)将特定的atenIR拆解为被装饰函数所调用的小算子。
    2、[特殊场景]某些atenIR存在特殊的dispatch key，如DispatchKey.CompositeImplicitAutograd。该dispatch key对应的C++
    实现，是无法进行dynamic=True时torch.compile模式下的sym符号推导。此时，通过aten.upsample_nearest2d.vec.py_impl装饰器
    将该dispatch key注册python实现解决该问题。
  **调整默认decomposition：**
    调整算子IR的默认decomposition一般有两种策略，禁止算子的默认decomposition与替换算子的默认decomposition。
    > 其中替换decomposition需要关注, 被替换的算子是否还存在decomposition，以及是否需要被禁止。
    如果您有屏蔽某些算子的decomposition的需求，欢迎贡献相关代码至python/torchair/_utils/adjust_implicit_decomposition.py。
  **提醒：**
    确认替换后的算子是否具有完备的meta infershape函数。
    > 替换算子本身的meta infershape是否完整，其中若只有C++ TORCH_META_FUNC宏注册的infershape可能无法在dynamic=True时工作，需要额外注册python的infershape函数。