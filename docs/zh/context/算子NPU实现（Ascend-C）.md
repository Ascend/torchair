# 算子NPU实现（Ascend C）

自定义PyTorch算子的NPU实现一般采用[Ascend C](简介.md#常用概念)编程语言实现，推荐“**工程化方式**”实现算子，其开发的NPU算子简称为“Ascend C算子”。

工程化开发是标准的算子开发流程，其简化了NPU适配过程，同时会自动注册Ascend C算子对应的Ascend IR，以保证PyTorch算子能与TorchAir max-autotune模式（Ascend IR）配合工作。

> **说明：** 
>使用TorchAir reduce-overhead模式时，您可以使用任意方式开发算子NPU实现，**只需要保证Torch算子能在Eager模式下正常工作**。
>使用TorchAir max-autotune模式时，需要确保算子NPU实现是以**Ascend C算子工程化方式**开发的aclnnXxx接口，以**Kernel直调方式开**发的NPU实现不会生成Ascend IR注册逻辑，没有对应的Ascend IR，无法完成PyTorch算子到Ascend IR的转换。

本章**仅提供基于Ascend C工程化开发算子的关键步骤说明**，详细的操作请参考《CANN Ascend C算子开发指南》中的“算子实现\>工程化算子开发”章节，例如算子原型json文件的参数含义、msOpGen工具的命令说明等。

1.  [创建自定义算子工程与原型](#创建自定义算子工程与原型)
2.  [实现Kernel与Tiling](#实现Kernel与Tiling)
3.  [实现InferShape与InferDataType（可选）](#sec3)
4.  [自定义算子包编译部署](#自定义算子包编译部署)

## 创建自定义算子工程与原型

1.  编写算子的原型定义json文件，用于生成算子开发工程。

    > **说明：** 
    >-   算子定义时，注意名称为Torch算子名的**大驼峰格式**，同时入参顺序与类型应当与[PyTorch算子原型](确定算子原型-0.md)完全一致。
    >-   特别注意的是，Ascend C算子定义时，**被修改的输入必须定义一个同名输出**，表达算子执行时对该输入的修改。

    例如MyInplace算子的原型json文件名为my\_inplace.json，文件内容如下：

    ```json
    [
        {
            "op": "MyInplace",
            "input_desc": [
                {
                    "name": "x",
                    "param_type": "required",
                    "format": ["ND"],
                    "type": ["float"]
                },
                {
                    "name": "y",
                    "param_type": "required",
                    "format": ["ND"],
                    "type": ["float"]
                }
            ],
            "output_desc": [
                {
                    "name": "x",
                    "param_type": "required",
                    "format": ["ND"],
                    "type": ["float"]
                },
                {
                    "name": "z",
                    "param_type": "required",
                    "format": ["ND"],
                    "type": ["float"]
                }
            ]
        }
    ]
    ```

2.  使用msOpGen工具生成算子的开发工程。

    ```bash
    ${INSTALL_DIR}/python/site-packages/bin/msopgen gen -i my_inplace.json -c ai_core-<soc_version> -f pytorch -lan cpp -out ./MyInplace
    ```

    -   \$\{INSTALL\_DIR\}为CANN软件安装后文件存储路径，请根据实际环境进行替换。
    -   -i：指定算子原型定义的json文件所在路径，请根据实际情况修改。
    -   -c：ai\_core-_\<soc\_version\>_代表算子在AI Core上执行，_\<soc\_version\>_为昇腾AI处理器的型号，请与实际环境保持一致。
    -   -lan：参数cpp代表算子基于Ascend C编程框架，使用C/C++编程语言开发。
    -   -out：生成文件所在路径，可配置为绝对路径或者相对路径，并且工具执行用户对路径具有可读写权限。若不配置，则默认生成在执行命令的当前路径。

3.  生成的算子核心工程目录结构如下：

    ```bash
    MyInplace
    ├── build.sh                       // 算子包编译脚本
    ├── ......
    ├── op_host                        // Host侧实现
    │  ├── my_inplace.cpp             // 算子定义、Tiling、InferShape、InferDataType实现文件
    │  └── my_inplace_tiling.h        // 算子Tiling定义文件
    └── op_kernel                      // Kernel侧实现
        └── my_inplace.cpp             // Kernel代码实现文件
    ```

## 实现Kernel与Tiling

为方便演示，本样例直接使用默认生成的Kernel和Tiling空实现，不影响后续的编译与执行。

实际业务场景下，您可以参考[《CANN Ascend C算子开发指南》](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)中的“算子实现\>工程化算子开发”章节下“**Kernel侧算子实现**”和“**Host侧Tiling实现**”，进行核心代码开发。

## 实现InferShape与InferDataType（可选）<a name="sec3"></a>

本步骤为可选操作，仅当需要使用TorchAir max-autotune模式（Ascend IR）提供的能力时，才需要实现InferShape与InferDataType。

前文msOpGen生成的自定义算子工程会在“op\_host/my\_inplace.cpp”中生成一份简单但通常合适的默认实现。

-   **方式1**：如果自定义算子满足如下条件，TorchAir会自动生成InferShape与InferDataType函数。

    当Ascend C算子输入输出与PyTorch算子可以一一对应时，可以删除算子工程生成的默认实现，TorchAir会在算子执行过程中自动生成InferShape与InferDataType函数。

-   **方式2**：如果不满足方式1，支持手动实现推导函数，请参考[《CANN Ascend C算子开发指南》](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)中“算子入图（GE图）开发”章节下“开发流程”完成DataType推导与Shape推导。

本样例中Ascend C算子MyInplace与PyTorch算子my\_inplace符合方式1，可以删除默认实现，使用TorchAir自动生成能力。

## 自定义算子包编译部署

1.  在自定义算子工程目录下执行如下命令，进行算子工程编译。

    ```bash
    cd ./MyInplace
    bash build.sh
    ```

    编译完成后，会在当前目录下创建build\_out目录，并在build\_out目录下生成自定义算子安装包custom\_opp\__<target os\>__\___<target architecture\>_.run。

    注意，自定义算子包的默认vendor名为customize，相同vendor名称的算子包会互相覆盖。

2.  自定义算子包安装。

    ```bash
    bash build_out/custom_opp_<target os>_<target architecture>.run
    cd ..
    ```