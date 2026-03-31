# 非In-place算子开发和入图样例

## 确定算子原型

本样例目标是自定义实现一个非In-place类PyTorch算子，使其在Eager和TorchAir图模式下正常工作。

假设目标PyTorch[算子Schema](../overview.md#常用概念)定义如下：

> [!NOTE]说明
>定义非In-place类算子时，需要遵循PyTorch原型定义惯例：
>
>- Tensor类型输入在前，基本类型输入在后。

```python
- func: my_op(Tensor x, Tensor? y, Tensor[] z, float attr1, int attr2) -> Tensor
```

- my\_op：算子名，对应调用方式为torch.ops.npu.my\_op。
- Tensor x：表示x为Tensor类型输入。
- Tensor? y：表示y为Tensor类型输入，但是允许传None。
- Tensor\[\] z：表示z为List\[Tensor\]类型输入。
- float attr1：表示attr1为float类型的值输入。
- int attr2：表示attr2为int类型的值输入。
- -\> Tensor：表示返回一个Tensor。

输出Tensor的shape与dtype与算子逻辑相关，假设本样例的输出Tensor shape和dtype与输入x完全相同。

请先完成<u>[环境准备](./overview.md#环境准备)</u>，确定好算子原型后，实现目标算子入图的步骤如下：

![](../figures/custom_op_10.png)

## 算子NPU实现（Ascend C）

自定义PyTorch算子的NPU实现一般采用Ascend C编程语言实现，推荐“工程化方式”实现算子，其开发的NPU算子简称为“Ascend C算子”。

工程化开发是标准的算子开发流程，其简化了NPU适配过程，同时会自动注册Ascend C算子对应的Ascend IR，以保证PyTorch算子能与TorchAir max-autotune模式（Ascend IR）配合工作。

> [!NOTE]说明
>使用TorchAir npugraph\_ex后端使能（aclgraph）模式时，您可以使用任意方式开发算子NPU实现，**只需要保证Torch算子能在Eager模式下正常工作**。
>使用TorchAir max-autotune模式时，需要确保算子NPU实现是以**Ascend C算子工程化方式**开发的aclnnXxx接口，以**Kernel直调方式开**发的NPU实现不会生成Ascend IR注册逻辑，没有对应的Ascend IR，无法完成PyTorch算子到Ascend IR的转换。

本章**仅提供基于Ascend C工程化开发算子的关键步骤说明**，详细的操作请参考《CANN Ascend C算子开发指南》中的“算子实现\>工程化算子开发”章节，例如算子原型json文件的参数含义、msOpGen工具的命令说明等。

### 创建自定义算子工程与原型

1. 编写算子的原型定义json文件，用于生成算子开发工程。

    > [!NOTE]说明
    >算子定义时，注意名称为Torch算子名的**大驼峰格式**，同时入参顺序与类型应当与[PyTorch算子原型](#确定算子原型)完全一致。

    例如MyOp算子的原型json文件名为my\_op.json，文件内容如下：

    ```txt
    [
        {
            "op": "MyOp",
            "input_desc": [
                {
                    "name": "x",
                    "param_type": "required",
                    "format": ["ND"],
                    "type": ["float"]
                },
                {
                    "name": "y",
                    "param_type": "optional",
                    "format": ["ND"],
                    "type": ["float"]
                },
                {
                    "name": "z",
                    "param_type": "dynamic",
                    "format": ["ND"],
                    "type": ["float"]
                }
            ],
            "attr": [
                {
                    "name": "attr1",
                    "param_type": "required",
                    "type": "float"
                },
                {
                    "name": "attr2",
                    "param_type": "required",
                    "type": "int"
                }
            ],
            "output_desc": [
                {
                    "name": "out",
                    "param_type": "required",
                    "format": ["ND"],
                    "type": ["float"]
                }
            ]
        }
    ]
    ```

2. 使用msOpGen工具生成算子的开发工程。

    ```bash
    ${INSTALL_DIR}/python/site-packages/bin/msopgen gen -i my_op.json -c ai_core-<soc_version> -f pytorch -lan cpp -out ./MyOp
    ```

    - $\{INSTALL\_DIR\}为CANN软件安装后文件存储路径，请根据实际环境进行替换。
    - -i：指定算子原型定义的json文件所在路径，请根据实际情况修改。
    - -c：ai\_core-_<soc\_version\>_代表算子在AI Core上执行，_<soc\_version\>_为昇腾AI处理器的型号，请与实际环境保持一致。
    - -lan：参数cpp代表算子基于Ascend C编程框架，使用C/C++编程语言开发。
    - -out：生成文件所在路径，可配置为绝对路径或者相对路径，并且工具执行用户对路径具有可读写权限。若不配置，则默认生成在执行命令的当前路径。

3. 生成的算子核心工程目录结构如下：

    ```txt
    MyOp
    ├── build.sh                   // 算子包编译脚本
    ├── ......
    ├── op_host                    // Host侧实现
    │  ├── my_op.cpp              // 算子定义、Tiling、InferShape、InferDataType实现文件
    │  └── my_op_tiling.h         // 算子Tiling定义文件
    └── op_kernel                  // Kernel侧实现
        └── my_op.cpp              // Kernel代码实现文件
    ```

### 实现Kernel与Tiling

为方便演示，本样例直接使用默认生成的Kernel和Tiling空实现，不影响后续的编译与执行。

实际业务场景下，您可以参考《CANN Ascend C算子开发指南》中的“算子实现\>工程化算子开发”章节下“**Kernel侧算子实现**”和“**Host侧Tiling实现**”，进行核心代码开发。

### 实现InferShape与InferDataType（可选）

本步骤为可选操作，仅当需要使用TorchAir max-autotune模式（Ascend IR）提供的能力时，才需要实现InferShape与InferDataType。

前文msOpGen生成的自定义算子工程会在“op\_host/my\_op.cpp”中生成一份简单但通常合适的默认实现。

- **方式1**：如果自定义算子满足如下条件，TorchAir会自动生成InferShape与InferDataType函数。

    当Ascend C算子输入输出与PyTorch算子可以一一对应时，可以删除算子工程生成的默认实现，TorchAir会在算子执行过程中自动生成InferShape与InferDataType函数。

- **方式2**：如果不满足方式1，支持手动实现推导函数，请参考《CANN Ascend C算子开发指南》中“算子入图（GE图）开发”章节下“开发流程”完成DataType推导与Shape推导。

本样例中Ascend C算子MyOp与PyTorch算子my\_op符合方式1，可以删除默认实现，使用TorchAir自动生成能力。

### 自定义算子包编译部署

1. 在自定义算子工程目录下执行如下命令，进行算子工程编译。

    ```bash
    cd ./MyOp
    bash build.sh
    ```

    编译完成后，会在当前目录下创建build\_out目录，并在build\_out目录下生成自定义算子安装包`custom_opp_<target os>_<target architecture>.run`。

    需要注意的是，自定义算子包的默认vendor名为customize，相同vendor名称的算子包会互相覆盖。

2. 自定义算子包安装。

    ```bash
    bash build_out/custom_opp_<target os>_<target architecture>.run
    cd ..
    ```

## 注册并适配Eager模式

完成算子NPU实现后，可对接PyTorch的Eager模式进行适配。Ascend Extension for PyTorch提供了OpPlugin算子插件，用来实现**PyTorch算子注册**和**Eager模式适配。**

本章**仅提供OpPlugin适配的关键步骤说明**，详细的操作请参考《PyTorch 框架特性指南》中的“基于OpPlugin算子适配开发”章节，例如算子yaml配置、算子适配等实现。

### 注册PyTorch算子

PyTorch官方提供的native\_functions.yaml文件定义了PyTorch Native Functions的具体算子定义和分发细节，定义则通过\*.cpp文件实现。OpPlugin库与原生库类似，也使用yaml文件定义了NPU适配的算子，算子具体适配则存放在\*.cpp文件中。

请确保已按[环境准备](./overview.md#环境准备)下载torch\_npu源码，算子的ATen IR定义位于third\_party/op-plugin/op\_plugin/config/op\_plugin\_functions.yaml文件中，在“**custom字段**”下添加[目标PyTorch算子](#确定算子原型)Schema定义：

```yaml
custom:   
- func: my_op(Tensor x, Tensor? y, Tensor[] z, float attr1, int attr2) -> Tensor     
  op_api: all_version
```

上述原型定义对应的PyTorch算子为torch.ops.npu.my\_op，其中torch.ops是PyTorch算子固定开头，npu为torch\_npu自定义算子库的名称，my\_op为自定义算子名。

### 基于OpPlugin适配Eager模式

完成算子NPU实现（Ascend C）和PyTorch算子注册后，需要在PyTorch的Eager模式适配层调用Ascend C算子。

借助OpPlugin插件提供的工程化适配能力，简化Eager模式适配层开发，在third\_party/op-plugin/op\_plugin/config/op\_plugin\_functions.yaml的my\_op算子原型注册下，追加“**gen\_opapi**”字段（表示对应可结构化的API）：

```yaml
- func: my_op(Tensor x, Tensor? y, Tensor[] z, float attr1, int attr2) -> Tensor
  op_api: all_version
  gen_opapi:
    out:
      size: x
      dtype: x
    exec: aclnnMyOp                       # 等价于aclnnMyOp,x,y,z,attr1,attr2
```

- out：表示函数的输出，包含size和dtype字段，如果包含多个输出，可配置成out0、out1等。对于out类接口，此字段不可自定义，需要与Aten IR定义的输出参数名相同。对于inplace类接口，不需要配置此字段。本样例中输出的size和dtype与x相同。
- exec：配置对应的EXEC\_NPU\_CMD接口，一般指aclnnXxx前缀接口。本样例配置为aclnnMyOp，aclnn为固定前缀，MyOp为Ascend C算子名，表示调用Ascend C算子MyOp实现PyTorch算子my\_op。

## 实现Meta推导函数

PyTorch原生要求所有能与torch.compile配合工作的算子需要实现Meta推导函数，又称为“符号化推导”。Meta函数表示了PyTorch算子输出与输入shape、dtype以及内存的关系，它是PyTorch入图的前提条件，借助符号化和符号guard可静态化控制流和形状信息，从而确定图结构。关于Meta函数的详细介绍请参考PyTorch官网[符号化手册](https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs/edit#heading=h.fh8zzonyw8ng)。

> [!NOTE]说明
>
>- Meta推导函数**必须在torch.compile执行前**完成注册。
>- torch.library.Library接口介绍请参考[PyTorch官网](https://docs.pytorch.org/docs/stable/library.html#torch.library.Library)。

进入third\_party/op-plugin/op\_plugin/python/meta/\_meta\_registrations.py实现Meta推导函数：

```python
import torch
from torch.library import Library, impl
# meta register implementation
m = Library("npu", "IMPL", "Meta")

@impl(m, "my_op")
def my_op_meta(x, y, z, attr1, attr2):
    return torch.empty_like(x)
```

- my\_op\_meta：Meta函数名，通常以PyTorch算子名+"\_meta"后缀命名。
- m：表示NPU算子的Meta实现库，通常定义在文件开头“m=Library\("npu", "IMPL", "Meta"\)”。

## 实现Converter（可选）

如果您希望使用GE图模式提供的高阶能力例如SuperKernel等，需要额外实现Ascend Converter（使用Ascend IR表达算子的计算逻辑）。

在Eager模式下，my\_op调用Ascend C算子MyOp；而对应到Converter实现，调用Ascend IR MyOp。

> [!NOTE]说明
>
>- 在Ascend C算子工程编译时，除了生成aclnnXxx接口外，还会同步生成同名Ascend IR的注册代码。
>- 接口介绍参见[register\_fx\_node\_ge\_converter](../ascend_ir/api/torchair/register_fx_node_ge_converter.md)和[custom\_op](../ascend_ir/api/ge/custom_op.md)。

通常不需要手动实现Converter，TorchAir会自动完成PyTorch算子到同名（大驼峰）Ascend IR的转换。例如本样例中的my\_op算子，会自动转换为Ascend IR MyOp。

如果自动转换无法完成，TorchAir的编译报错信息会给出原因，原因一般如下：

- PyTorch算子的名字无法与Ascend IR名字通过大驼峰格式对应，例如my\_op实际对应Ascend IR的名字为MyOp或OtherOp等。
- PyTorch算子与Ascend IR的输入输出顺序或数量不一致。
- PyTorch算子原型定义中存在Scalar类型入参。

您可以修改PyTorch算子原型使其满足条件，让TorchAir自动完成转换，或者手动实现Converter：

在third\_party/torchair/torchair/python/torchair/\_ge\_concrete\_graph/ge\_converter/custom目录下，新建MyOp算子对应的my\_op.py文件，添加如下代码实现Converter：

```python
import torch
import torchair

# torch.ops.npu.my_op.default为自定义算子生成的Python函数签名，注意default后缀
@torchair.register_fx_node_ge_converter(torch.ops.npu.my_op.default)  
def convert_npu_my_op(x, y, z, attr1, attr2):           # 函数入参与Torch算子入参一致
    return torchair.ge.custom_op("MyOp", x, y, z, attr1, attr2)
```

## 功能验证

1. 编译自定义算子包。

    参考<u>[环境准备](./overview.md#环境准备)</u>准备好环境，执行如下命令重新编译、安装自定义算子torch.ops.npu.my\_op的torch\_npu包。请注意与当前运行环境的Python版本匹配，以Python3.8版本为例：

    ```bash
    bash ci/build.sh --python=3.8
    pip3 install dist/torch*.whl --force-reinstall --no-deps
    ```

2. 验证自定义算子在Eager模式、TorchAir npugraph\_ex后端使能（aclgraph）模式、TorchAir max-autotune模式下功能是否正常

    ```python
    import torch
    import torch_npu
    import torchair
    
    def test_eager(x, y, z, attr1, attr2):
        return torch.ops.npu.my_op(x, y, z, attr1, attr2)
    
    # aclgraph模式
    @torch.compile(backend="npugraph_ex")
    def test_torchair_npugraph_ex(x, y, z, attr1, attr2):
        return torch.ops.npu.my_op(x, y, z, attr1, attr2)
    
    config = torchair.CompilerConfig()
    config.mode = "max-autotune"          # 表示Ascend IR模式
    @torch.compile(backend=torchair.get_npu_backend(compiler_config=config))
    def test_torchair_max_autotune(x, y, z, attr1, attr2):
        return torch.ops.npu.my_op(x, y, z, attr1, attr2)
    
    x = torch.ones(4, 8).npu()
    y = None
    z = [torch.ones(4, 8).npu(), torch.ones(4, 8).npu()]
    attr1 = 2.0
    attr2 = 5
    
    test_eager(x, y, z, attr1, attr2)
    torch.npu.synchronize()
    print("Eager ok")
    test_torchair_npugraph_ex(x, y, z, attr1, attr2)
    torch.npu.synchronize()
    print("TorchAir npugraph_ex ok")
    test_torchair_max_autotune(x, y, z, attr1, attr2)
    torch.npu.synchronize()
    print("TorchAir max-autotune ok")
    ```

## 样例总结

非In-place算子在入图过程中存在一些注意事项，在实际操作时请注意：

- 定义PyTorch算子时，Tensor类型输入在前，基本类型输入在后。
- 以算子工程方式开发PyTorch算子的Ascend C实现，Ascend C算子名保证为PyTorch算子的大驼峰格式，同时入参顺序、类型与PyTorch算子一致。
- 使用OpPlugin工程化适配能力，简化PyTorch算子Eager模式NPU适配层开发。
