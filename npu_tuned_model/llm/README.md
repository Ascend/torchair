# 介绍

本章节旨在提供在npu上优化改造好的大模型，方便开发者直接使用已经适配好的npu大模型或者对自定义的大模型进行NPU的迁移时提供参考。

是一个快速且易于让大模型接入昇腾CANN框架的推理和服务库

性能优势体现：

- 支持Npu融合算子

易用性优势体现：

- 支持pytorch前端框架

支持的Hugging Face大模型:

- llama2、llama3
- chatglm
- qwen2

# 公告

- 2024年7月19号：qwen2优化torch_npu.npu_incre_flash_attention算子计算量和使能tiling全下沉配置
- 2024年7月5号：提供qwen2适配好的npu模型结构和前端切分分布式执行样例
- 2024年6月3号：提供chatglm3适配好的npu模型结构和前端切分分布式执行样例
- 2024年5月6号：提供llama2基于分离部署api的适配样例
- 2024年3月6号：提供llama2适配好的npu模型结构和前端切分分布式执行样例

# 环境准备

请参考[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)安装对应的驱动/固件/软件包，在安装深度学习框架时选择安装PyTorch框架，根据资料指导进行安装并设置环境变量，涉及的软件包如下：

|    软件    |                             版本                             |
| :--------: | :----------------------------------------------------------: |
|   Python   |                            3.8.0                             |
|    CANN    | [社区版本](https://www.hiascend.com/developer/download/community/result?module=pt+cann&pt=6.0.RC3.alpha002&cann=8.0.RC3.alpha002) |
|   kernel   | [社区版本](https://www.hiascend.com/developer/download/community/result?module=pt+cann&pt=6.0.RC3.alpha002&cann=8.0.RC3.alpha002) |
|   torch    |                            2.1.0                             |
| torch_npu  | [2.1.0](https://gitee.com/ascend/pytorch/releases/tag/v6.0.rc2-pytorch2.1.0) |
|    apex    |                             0.1                              |
| 第三方依赖 |                       requirement.txt                        |

# 环境搭建

```shell
# arm环境搭建示例
conda create -n test python=3.8
conda activate test

# 根据CANN安装指南安装固件/驱动/cann包

# 安装 torch 和 torch_npu
pip3 install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
pip3 install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl
pip3 install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

git clone https://gitee.com/ascend/torchair.git
cd torchair/npu_tuned_model/llm
pip3 install -r requirement.txt
```

**注意**：建议昇腾run包安装在conda创建的镜像中安装。如果遇到ASCEND_HOME_PATH环境变量未设置，设置为run包安装路径

# 模型及数据集

| 数据集      | 参数量                                                                                               |
|----------|---------------------------------------------------------------------------------------------------|
| llama2   | [70b](https://huggingface.co/TheBloke/Llama-2-70B-fp16/tree/main)                                 |
| llama3   | [70b](https://huggingface.co/meta-llama/Meta-Llama-3-70B/tree/main)                               |
| chatglm3 | [6b](https://huggingface.co/THUDM/chatglm3-6b)                                                    |
| qwen2    | [7B](https://huggingface.co/Qwen/Qwen2-7B) \| [72B](https://huggingface.co/Qwen/Qwen2-72B) |

# 快速体验

[llama](./llama/README.md#性能测试)

[chatglm3](./chatglm/README.md#性能测试)

[qwen2](./qwen/README.md#性能测试)

# 新模型改造

选择和样例模型结构近似的进行参考适配

# 使能分离部署功能

[llama](./llama/benchmark/pd_separate/README.md)