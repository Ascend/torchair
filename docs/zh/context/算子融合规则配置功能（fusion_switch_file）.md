# 算子融合规则配置功能（fusion\_switch\_file）

## 功能简介

图模式场景下执行模型时会自动融合算子，以降低网络推理时间、提高整网性能。算子融合方式主要包括如下两种：

-   图融合（Graph Fusion）：指融合引擎根据融合规则进行改图的过程，该过程主要通过拆分/合并计算图中的算子来提升计算效率，以实现加速运算的目的，与硬件无关。
-   UB融合（Unified Buffer Fusion）：指对图上算子进行硬件UB相关的融合，UB是昇腾AI处理器上的Unified Buffer。例如两个算子a和b单独运行时，算子a的计算结果在UB上，需要搬移到DDR（Double Data Rate，双倍速率同步动态随机存储器）。算子b再执行时，需要将算子a的输出由DDR再搬移到UB，进行算子b的计算逻辑，计算完之后，又从UB搬移回DDR。

更多融合规则相关介绍请参见《CANN 图融合和UB融合规则参考》。目前系统内置了一些算子融合规则，**默认情况下均为开启**（如有默认关闭，会有特殊说明）。TorchAir在npu\_backend中额外提供了config配置项，允许用户自定义关闭/开启部分融合算子。

## 使用约束

本功能仅支持max-autotune模式。

## 使用方法

1.  创建算子融合规则配置文件（\*.cfg）。

    文件名自定义（例如fusion\_switch.cfg），内容示例如下，其中on表示开启融合规则，off表示关闭融合规则。

    ```
    {
        "Switch":{
            "GraphFusion":{
                "ConvToFullyConnectionFusionPass":"on",
                "SoftmaxFusionPass":"on",
                "ConvConcatFusionPass":"on",
                "MatMulBiasAddFusionPass":"on",
                "PoolingFusionPass":"on",
                "ZConcatv2dFusionPass":"on",
                "ZConcatExt2FusionPass":"on",
                "TfMergeSubFusionPass":"on"
            },
            "UBFusion":{
                "FusionVirtualOpSetSwitch":"on"
            }
        }
    }
    ```

    同时支持用户一键关闭/开启融合规则，一键关闭操作示例如下：

    ```
    {
        "Switch":{
            "GraphFusion":{
                "ALL":"off"    
            },
            "UBFusion":{
                "ALL":"off"
             }
        }
    }
    ```

    > **说明：** 
    >-   一键式关闭融合规则仅关闭系统部分融合规则，而非全部融合规则。换言之配置"ALL": "off"后，部分融合算子仍旧会生效，因为关闭部分融合规则会导致功能问题。
    >-   一键式关闭融合规则时，也可以同时开启部分融合规则，样例如下：
    >    ```
    >    {
    >        "Switch":{
    >            "GraphFusion":{
    >                "ALL":"off",
    >                "SoftmaxFusionPass":"on"        
    >            },
    >            "UBFusion":{
    >                "ALL":"off",
    >                "TbePool2dQuantFusionPass":"on"
    >             }
    >        }
    >    }
    >    ```

2.  使能算子融合规则配置文件。

    该配置文件通过[torchair.get\_npu\_backend](get_npu_backend.md)中compiler\_config参数配置生效，示例如下，参数说明如下表。

    ```python
    import torch_npu, torchair
    config = torchair.CompilerConfig()
    # 指定融合配置文件的路径
    config.fusion_config.fusion_switch_file = "/home/test/fusion_switch.cfg"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    opt_model = torch.compile(model, backend=npu_backend)
    ```

    **表 1**  参数说明

    
    | 参数名 | 说明 |
    | --- | --- |
    | fusion_switch_file | 指定算子融合规则文件。<br>**说明**： 请确保该参数指定的路径确实存在，并且运行用户具有读、写操作权限。 |

    若想确认该功能配置是否生效，在运行之前需要提前设置[日志环境变量](TorchAir-C++层日志.md#使用方法)，该环境变量会打印图初始化时的一些开关配置。

    ```bash
    export TNG_LOG_LEVEL=0
    ```

    若配置正确，在日志信息中搜索“ge.fusionSwitchFile”关键字，可以看见类似的打印信息。

    ```bash
    concrete_graph/session.cpp:28   ge.fusionSwitchFile: /home/test/fusion_switch.cfg
    ```

## 产物说明

图编译完成后默认在当前执行路径下生成fusion\_result.json。该文件用于记录图编译过程中除去fusion\_switch.cfg文件中关闭的融合规则外，仍旧使用的融合规则。其中"match\_times"字段表示过程中匹配到的融合规则次数，"effect\_times"字段表示实际生效的次数。

```
{
    "session_and_graph_id_0_0": {         
        "graph_fusion": {
            "pass1": {                    
                "effect_times": "1",      
                "match_times": "1"        
            },
            "pass2": {                     
                "effect_times": "2",
                "match_times": "2"
            }
        },
        "ub_fusion": {                  
            "pass3": {                         
                "effect_times": "3",          
                "match_times": "3",            
                "repository_hit_times": "0"    
            }
        }
    },
    "session_and_graph_id_1_1": {
        "graph_fusion": {
            "pass1": {
                "effect_times": "5",
                "match_times": "5"
            }
 
        },
        "ub_fusion": {
            "pass2": {
                "effect_times": "7",
                "match_times": "7",
                "repository_hit_times": "0"
            }
        }
    }
}
```

-   session\_and\_graph\_id\__xx\_xx_：表示融合结果所属线程和图编号。
-   graph\_fusion：表示图融合。
-   ub\_fusion：表示UB融合。
-   match\_times：表示图编译过程中匹配到的融合规则次数。
-   effect\_times：表示实际生效的次数。
-   repository\_hit\_times：优化UB融合知识库命中的次数。

