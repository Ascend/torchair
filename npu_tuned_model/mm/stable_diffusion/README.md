# 介绍

本章节旨在提供在npu上优化改造好的大模型，方便开发者直接使用已经适配好的npu大模型或者对自定义的大模型进行NPU的迁移时提供参考。

是一个快速且易于让大模型接入昇腾CANN框架的推理和服务库

性能优势体现：

- 支持Npu融合算子

易用性优势体现：

- 支持pytorch前端框架

支持的Hugging Face大模型:

- stable diffusion 2.1 base

# 公告

- 2024年6月11号：提供sd2.1适配好的npu模型结构和前端切分分布式执行样例

# 环境依赖

|    软件    |             [版本](https://www.hiascend.com/zh/)             |
| :--------: | :----------------------------------------------------------: |
|   Python   |                            3.8.0                             |
|   driver   | [2024Q1商发](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743/software/261964445?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C252764743) |
|  firmware  | [2024Q1商发](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743/software/261964445?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C252764743) |
|    CANN    | [2024Q2商发](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/262517542?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C251168373) |
|   kernel   | [2024Q2商发](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/262517542?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C251168373) |
|   torch    |                            2.1.0                             |
| torch_npu  |   [2023Q4商发](https://gitee.com/ascend/pytorch/releases)    |
|    apex    | [2023Q4商发](https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/Daily/v2.1.0/20231225.2/pytorch_v2.1.0_py38.tar.gz) |
| 第三方依赖 |                       requirement.txt                        |

# 环境搭建

```shell
# python3.8
conda create -n test python=3.8
conda activate test

# 安装 torch 和 torch_npu
pip3 install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
pip3 install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl
pip3 install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

pip3 install -r requirement.txt
```

[昇腾环境准备](https://www.hiascend.com/document/detail/zh/canncommercial/700/modeldevpt/ptmigr/AImpug_000005.html)

**注意**：建议昇腾run包安装在conda创建的镜像中安装

# 下载代码仓

首先下载torchair仓：
```shell
git clone https://gitee.com/ascend/torchair.git
```

针对昇腾NPU，我们提供了优化补丁，以便模型能够更好地进行部署（依赖patch命令）：
```shell
cd torchair/npu_tuned_model/mm/engines
bash diffusers_patch.sh
```

# 项目结构

```t
│  README.md
│  requirement.txt
│
├─engines
│          diffusers_npu.patch   # 适配npu的diffusers库patch
│          diffusers_patch.sh   # 安装diffusers库patch脚本
│          
│      
├─examples           # 模型脚本样例
│  │  
│  └─sd2.1
│          run_sd.py
│          run_sd.sh         # 分布式推理脚本
│          
└─models            
   │  __init__.py
   │  
   └─npu
           attention_processor.py     # 适配昇腾NPU的attention_processor样例

```

# 下载模型

| 模型 | 版本                                                       |
| ------ | ------------------------------------------------------------ |
| sd2.1 | [base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/tree/main) |


# 快速体验

在npu_tuned_model/mm/examples/sd2.1/run_sd.sh中首先根据提示配置：
```shell
export PYTHONPATH=$PYTHONPATH:/path/to/your/torchair/npu_tuned_model/mm/stable_diffusion:/path/to/your/torchair/npu_tuned_model/mm/stable_diffusion/engines/src/
cann_path=/usr/local/Ascend # 昇腾cann包安装目录
source ${cann_path}/latest/bin/setenv.bash
model_path=xxx/stable-diffusion-2-1-base # 下载的权重和模型信息
device_list="0"
```

若期望在Ascend 310P DUO卡上使能双芯并行推理，则改为“0,1"
```shell
device_list="0,1" 
```

执行脚本
```shell
bash run_sd.sh
```

# 使能改造好的npu models

一：使能NPU FlashAttention，可在执行命令中通过--FA控制。

```python
from models.npu.attention_processor import AttnProcessor2_0
pipe.unet.set_attn_processor(AttnProcessor2_0())
```

二：使能torch dynamo图模式，可在执行命令中通过--dynamo控制。
```python
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
config = CompilerConfig()
config.experimental_config.frozen_parameter = True
npu_backend = tng.get_npu_backend(compiler_config=config)

'''
use weight nz
'''
tng.experimental.inference.use_internal_format_weight(pipe.text_encoder)
tng.experimental.inference.use_internal_format_weight(pipe.unet)
tng.experimental.inference.use_internal_format_weight(pipe.vae.decoder)
'''
torch dynamo
'''
pipe.text_encoder = torch.compile(pipe.text_encoder, backend=npu_backend)
pipe.unet = torch.compile(pipe.unet, backend=npu_backend)
pipe.vae.decoder = torch.compile(pipe.vae.decoder, backend=npu_backend)
```

三：设置随机种子

双芯推理时，需要保证不同进程的随机保持一致，需要设置一样的随机种子。

```python
def init_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch_npu.npu.manual_seed(seed)
    torch_npu.npu.manual_seed_all(seed)

init_seed(args.seed)
```

四：使能[Token Merge](https://arxiv.org/abs/2303.17604)加速算法

可在执行命令中通过--TOME开启。出于推理精度考虑，我们仅在第一个down_blocks和最后一个up_blocks的前两个self_attention结构使能Token Merge。

```python
pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1.set_processor(AttnProcessor2_0Tome())
pipe.unet.down_blocks[0].attentions[1].transformer_blocks[0].attn1.set_processor(AttnProcessor2_0Tome())
pipe.unet.up_blocks[3].attentions[0].transformer_blocks[0].attn1.set_processor(AttnProcessor2_0Tome())
pipe.unet.up_blocks[3].attentions[1].transformer_blocks[0].attn1.set_processor(AttnProcessor2_0Tome())
```

五：使能[Deep cache](https://arxiv.org/pdf/2312.00858.pdf)加速算法

可在执行脚本中通过--DC开启。在推理过程中配置cache_interval>1，cache_layer_id、cache_block_id配置请参考论文。

```python
image = pipe(prompt, cache_interval=2, cache_layer_id=0, cache_block_id=0).images[0] 
```