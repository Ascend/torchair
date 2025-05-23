# 简介
本项目开发了名为TorchAir(Torch Ascend Intermediate Representation)的扩展库，支持用户基于PyTorch框架和torch_npu插件在昇腾NPU上使用图模式进行推理。
TorchAir继承自PyTorch框架[Dynamo模式](https://pytorch.org/docs/stable/torch.compiler_dynamo_deepdive.html)，将PyTorch的[FX图](https://pytorch.org/docs/stable/fx.html)转换为GE计算图，并提供了GE计算图在昇腾NPU的编译与执行的能力。
> - **如果您想了解如何使用TorchAir，可以优先通过访问[TorchAir图模式使用指南](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/modthirdparty/torchairuseguide/torchair_0001.html)，了解关于TorchAir的更多信息。**
> - **如果您计划为TorchAir贡献代码，请参考REAMDE内容。**


# 安装与卸载
1. 安装
- 安装依赖

在安装TorchAir之前，请参考[版本配套表](#version_match)和[支持型号](#hardware_support)说明，确保您的硬件能够使用TorchAir，并安装最新昇腾软件栈。

- 编译准备

克隆TorchAir代码仓
```shell
git clone https://gitee.com/ascend/torchair.git
```

下载依赖三方库
```shell
git submodule update --init --recursive
```

配置编译环境，执行配置命令
> 配置只需要进行一次，用于获取pytorch的编译选项（如当前的torch是否开启ABI）及Ascend sdk的安装路径（如果需要在本地CPU上进行调试）。
```shell
cd ./torchair
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

如果您不需要在本地CPU模拟执行，可以直接回车跳过。否则，需要输入昇腾处理器开发套件的安装路径（默认安装路径通常为~/Ascend/ascend-toolkit/latest/）。

> 您可以通过设置 `ASCEND_SDK_PATH` 环境变量指定 SDK 目录 或 设置 `NO_ASCEND_SDK` 环境变量指定不需要 SDK 来抑制交互式窗口弹出。

键入后，等待配置完成。

- 编译

执行以下命令，编译生成TorchAir安装包：
```shell
mkdir build
cd build
cmake ..
make torchair -j8
```

- 安装

编译完成后，会在`build/dist/dist/`目录下生成名为torchair-{version}-py3-none-any.whl的安装包文件。

您可以直接使用pip安装该安装包，或者使用make命令安装至您configure时指定的python环境中。
```shell
make install_torchair
```

2. 卸载
    
torchair的卸载只需要执行命令：

```
pip3 uninstall torchair
```
如需要保存卸载日志，可在pip3 uninstall命令后面加上参数`--log <PATH>`，并对您指定的目录`<PATH>`做好权限管控。

# 快速上手

> 如果您在配置时未指定Ascend sdk的安装路径，则无法执行CPU上的调试，需要在NPU环境上进行测试。

CPU调试时，需要设置LD_LIBRARY_PATH到生成的fake so文件目录以及sdk目录

> tools/env.sh会根据配置生成对应的LD_LIBRARY_PATH（如果您在配置时指定了Ascend sdk安装路径）
```shell
source tools/env.sh
```
执行以下python脚本快速验证TorchAir基本功能
```python
import torch
import torchair

config = torchair.CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.add(x, y)

model = Model()
model = torch.compile(model, backend=npu_backend, dynamic=False)
x = torch.randn(2, 2)
y = torch.randn(2, 2)
model(x, y)
```


# 特性介绍
TorchAir常用特性介绍
| 特性功能               | 功能介绍       | 参考资料            |
| -----------           | ---------------------------------   | -----------        |
| 日志功能               |  日志功能             | [参考链接](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/modthirdparty/torchairuseguide/torchair_0012.html)       |
| graph dump功能         |  图dump功能      | [参考链接](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/modthirdparty/torchairuseguide/torchair_0014.html)        |
| dynamo export功能      | air格式图导出功能         | [参考链接](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/modthirdparty/torchairuseguide/torchair_0015.html)        |
| data dump功能          |  精度数据dump功能           | [参考链接](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/modthirdparty/torchairuseguide/torchair_0016.html)        |
| graph fusion功能       |  用户自定义关闭/开启部分融合算子功能           | [参考链接](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/modthirdparty/torchairuseguide/torchair_0017.html)        |
| experimental性能提升功能 |  试验性质功能，不同功能适用于特定场景，详见参考资料             | [参考链接](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/modthirdparty/torchairuseguide/torchair_0019.html)        |
| converter功能拓展       |  用户自行扩展模型中缺失的converter功能             | [参考链接](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/modthirdparty/torchairuseguide/torchair_0026.html)        |
| 支持的aten API清单      |  支持的aten API清单               | [参考链接](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/modthirdparty/torchairuseguide/torchair_0042.html)        |

<a id="version_match"></a>
# 版本配套表
| TorchAir版本 | PyTorch版本 | torch_npu版本 | CANN版本 | Python版本
| ----------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| master（主线） | 2.1.0 | 在研版本 | 在研版本 | Python3.8.x<br/>Python3.9.x<br/>Python3.10.x | 
| 6.0.rc3 | 2.1.0 | 6.0.rc3 | 8.0.rc3 | Python3.8.x<br/>Python3.9.x<br/>Python3.10.x | 
| 6.0.rc2 | 2.1.0 | 6.0.rc2 | 8.0.rc2 | Python3.8.x<br/>Python3.9.x<br/>Python3.10.x | 
| 6.0.rc1 | 2.1.0 | 6.0.rc1 | 8.0.rc1 | Python3.8.x<br/>Python3.9.x<br/>Python3.10.x | 

<a id="hardware_support"></a>
# 支持的型号
- Atlas A2 训练系列产品
- Atlas 推理系列产品（配置Ascend 310P AI处理器）

# 贡献
如果您计划为TorchAir做出贡献，请参考[CONTRIBUTING](https://gitee.com/ascend/torchair/tree/master/CONTRIBUTING.md)。


# 安全声明
TorchAir安全声明参考[SECURITY_README](https://gitee.com/ascend/torchair/blob/master/SECURITY_README.md)文件。


# 参考文档
有关TorchAir的更多详细信息,请参考[TorchAir图模式使用](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/modthirdparty/torchairuseguide/torchair_0001.html)。


# 许可证
TorchAir插件使用BSD许可证。详见[LICENSE](https://gitee.com/ascend/torchair/blob/master/LICENSE)文件。


---

# 免责声明

## 致TorchAir使用者

1. TorchAir提供的模型仅供您用于非商业目的。
2. 对于各模型，TorchAir平台仅提示性地向您建议可用于训练的数据集，华为不提供任何数据集，如您使用这些数据集进行训练，请您特别注意应遵守对应数据集的License，如您因使用数据集而产生侵权纠纷，华为不承担任何责任。
3. 如您在使用TorchAir模型过程中，发现任何问题（包括但不限于功能问题、合规问题），请在Gitee提交issue，我们将及时审视并解决。

## 致数据集所有者

如果您不希望您的数据集在TorchAir中的模型被提及，或希望更新TorchAir中的模型关于您的数据集的描述，请在Gitee提交issue，我们将根据您的issue要求删除或更新您的数据集描述。衷心感谢您对TorchAir的理解和贡献。