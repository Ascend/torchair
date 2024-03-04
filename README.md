torchair实现将torch的FX图转换为GE计算图，并提供了GE计算图的编译与执行接口。

# 编译准备

配置编译环境，执行配置命令
> 配置只需要进行一次，用于获取pytorch的编译选项（如当前的torch是否开启ABI）及Ascend sdk的安装路径（如果需要在本地CPU上进行调试）。
```shell
bash ./configure
```
默认情况下，执行上述命会弹出如下的交互式会话窗口
> 您的会话可能有所不同。

```BASH
Please specify the location of python with available torch 2.1.x installed. [Default is /usr/bin/python3]
(You can make this quiet by set env [TARGET_PYTHON_PATH]):
```

此时，要求您输入安装了 Torch 2.1 版本的python解释器路径，如果默认路径是正确的，直接回车，否则请输入正确的 python 解释器路径。
> 您可以通过设置 `TARGET_PYTHON_PATH` 环境变量，来抑制交互式窗口弹出，但是要确保路径是有效的，否则，仍然会要求您输入正确的 python 解释器路径。

键入后，会耗费几秒钟以确保您的输入是有效的，接着，会弹出下面的交互式窗口

```BASH
Specify the location of ascend sdk for debug on localhost or leave empty.
(You can make this quiet by set env [ASCEND_SDK_PATH]):
```

如果您不需要在本地CPU调试，可以直接回车跳过。否则，需要输入昇腾处理器开发套件的安装路径（需指定至opensdk/opensdk目录）。

> 您可以通过设置 `ASCEND_SDK_PATH` 环境变量指定 SDK 目录 或 设置 `NO_ASCEND_SDK` 环境变量指定不需要 SDK 来抑制交互式窗口弹出。

键入后，等待配置完成。

# 编译与安装

```shell
mkdir build
cd build
cmake ..
make torchair -j8
```

编译完成后，会在`build/dist/dist/`目录下生成名为torchair-{version}-py3-none-any.whl的安装包文件。

您可以直接使用pip安装该安装包，或者使用make命令安装至您configure时指定的python环境中。
```shell
make install_torchair
```

# 执行测试

> 如果您在配置时未指定Ascend sdk的安装路径，则无法执行CPU上的调试，需要在NPU环境上进行测试。

CPU调试时，需要设置LD_LIBRARY_PATH到生成的fake so文件目录以及sdk目录

> tools/env.sh会根据配置生成对应的LD_LIBRARY_PATH（如果您在配置时指定了Ascend sdk安装路径）
```shell
source tools/env.sh

python3 examples/example.py
```

# 关于CI编译

工程目录下的build.sh用于对接CI编译，但是您也可以通过其在本地执行编译和UT/ST测试。

## 查看帮助
```shell
./build.sh -h
```

## 编译安装包
```shell
./build.sh -c
```
编译完成后，会在`output/`目录下生成名为torchair-{version}-py3-none-any.whl的安装包文件。
## 执行UT测试
> 本地执行UT测试需要设置环境变量`ASCEND_CUSTOM_PATH`,将其指定至Ascend sdk的安装路径（指定至ai_cann_x86目录）
```shell
./build.sh -u
```

## 执行ST测试
> 本地执行ST测试需要设置环境变量`ASCEND_CUSTOM_PATH`,将其指定至Ascend sdk的安装路径（指定至ai_cann_x86目录）
```shell
./build.sh -s
```

## 查看覆盖率报告
UT或ST执行通过后，会在coverage目录下生成覆盖率文件coverage.info，如果您想要查看覆盖率报告，可以执行如下命令
```shell
genhtml coverage.info -o coverage_report
```

# 基于torchair调试pytorch模型

## 在合适的位置添加torch.compile，并指定npu后端
```python
import torchair
# torchair提供了一些额外的编译选项，您可以在此处指定，也可以设置为None使用默认选项
npu_backend = torchair.get_npu_backend(compiler_config=None)
torch.compile(model, backend=npu_backend)
```

`注意`，如果您基于torch_npu执行，可以使用torch_npu提供的"npu"后端。
> torch_npu目前正在集成torchair作为其torch.compile后端。
```python
torch.compile(model, backend="npu")
```

## 确定需要补齐的converter
torchair提供配置项config.debug.fx_summary开关来确定FX图中涉及需要补齐的converter，您可以通过如下方式来配置
```python
config = torchair.CompilerConfig()
config.debug.fx_summary.type = "csv"
npu_backend = torchair.get_npu_backend(compiler_config=config)

torch.compile(model, backend=npu_backend)
```
执行后，会生成summary_{timestamp}.csv文件，您可以通过excel等工具来查看。
通过导出的csv文件，您可以看打当前模型fx图中涉及的所有converter
- 对于`支持状态`为`未支持`的converter，您需要在`ge_concrete_graph/ge_converter`目录下对应文件中补齐实现
> 当前已经为aten下的op提供了一个壳子实现（固定抛出未支持的异常），您应当在此基础上补齐实现

- 对于`支持状态`为`未注册`的converter，您需要在`ge_concrete_graph/ge_converter`目录下对应文件中新增注册并补齐实现

- 对于`支持状态`为`部分支持`的converter，您需要查看位于`ge_concrete_graph/ge_converter`目录下的实现，并根据输入数据列决定是否需要补齐场景实现

## 实现converter
您可以参考`ge_concrete_graph/ge_converter`目录下的实现，实现对应的converter。
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
### 声明converter支持的场景
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

### converter的函数签名
```python
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec

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
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
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


### 实现converter
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

### 注意事项
需要特别注意，所实现的converter必须支持动态shape，不应该试图从输入的Tensor上获取任何shape信息，Tensor也不会提供任何shape信息。
> 如果您的converter依赖shape才能工作，这通常意味着实现错误，或者没有选择正确的Ascend IR映射。

### 自定义算子入图插件化注册
此功能可以让用户注册自定义算子，增加自定义算子的入图能力，且无需重新编译torch_npu与torchair。
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
生成的ge.api函数在codegen/custom_op/auto_generated_ge_raw_custom_ops.py文件中

（3）将您生成的文件import至您的工程中或者拷贝源码至您的调用文件，保证converter能够调用到即可。

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
    return Add(input1, input2)
```

至此完成全部自定义算子入图适配工作，您可以运行参考用例中的示例验证。
> 注意： 您在开发您自己的REG_OP(xxx)的自定义算子时，需要向GE注册infershape函数，否则执行时会出错。


> 参考用例 

（1）examples/example_custom_op_register/example_custom_op_register_in_one_file.py  一个文件内部完成注册使用。

（2）examples/example_custom_op_register/new_custom_op.py 注册新算子
examples/example_custom_op_register/example_use_import_new_custom_op.py  import新算子模块，使用注册的新算子。

## 导出gegraph

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

**需要注意：**
本段接口可以导出图的相关信息，请用户加强对相关数据的保护。
请在接口的功能完成之后及时关闭相关选项。


# torchair常用类和公开接口介绍

## TORCHAIR.GET_NPU_BACKEND
`torchair.get_npu_backend(*, compiler_config=None, aot_config=None, custom_decompositions={})`

获取能够在NPU上运行的图编译后端npu_backend，可以作为backend参数传入torch.compile。

### Keyword Arguments
- **compiler_config**(*CompilerConfig*)- 配置项，具体可见torchair.CompilerConfig条目。
- **aot_config**(*AotConfig*)- 配置是否将前反向图以一个完整图的方式运行。
- **custom_decomposition**(*Dict*)- 手动指定模型运行时用到的decomposition。

## TORCHAIR.GET_COMPILER
`torchair.get_compiler(compiler_config=None)`

获取能够在NPU上运行的图编译器。torchair.get_npu_backend()获取的图编译后端默认使用由本接口获取的图编译器。用户也可将获取的图编译器传入自定义的后端中。

### Parameters
- **compiler_config**(*CompilerConfig*)- 配置项，具体可见torchair.CompilerConfig条目。

## TORCHAIR.COMPILERCONFIG
`torchair.CompilerConfig`

配置类。用户可以通过CompilerConfig配置以下功能：
- **debug** 用于配置图dump、Converter支持状态导出等选项。
- **aoe_config** 用于配置自动调优工具aoe的模式。
- **export** 用于配置export图时的选项。
- **fusion_config** 用于配置融合选项。
- **experimental_config** 用于配置实验功能。

## TORCHAIR.DYNAMO_EXPORT
`torchair.dynamo_export(*args, model, export_path="export_file", export_name="export", dynamic= False, config=CompilerConfig())`

导出由torchair生成的离线图。

### Parameters
- **args** 模型的入参。
- **model** 用户的自定义模型。
- **export_path** 离线图导出的位置。
- **export_name** 离线图导出的名字。
- **dynamic** 是否是动态图。
- **config** 配置项，具体可见torchair.CompilerConfig条目。

## TORCHAIR.USE_INTERNAL_FORMAT_WEIGHT
`torchair.use_internal_format_weight(model)`

将模型的权重转换为NPU的私有格式。

### Parameters
- **model** 用户的自定义模型。

# 系统安全配置
用户可在运行系统配置时开启ASLR（级别2）以提高系统安全性，保护系统随机化开启。

可参考以下方式进行配置：
```python
echo 2 > /proc/sys/kernel/randomize_va_space
```

# 公网地址说明
代码涉及公网地址参考public_address_statement.md