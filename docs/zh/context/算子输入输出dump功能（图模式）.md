# 算子输入输出dump功能（图模式）

## 功能简介

Graph模式下，dump Ascend IR计算图上算子执行时的输入、输出数据，用于后续问题定位和分析，如算子运行性能或精度问题。

> **说明：** 
> 本功能与[图结构dump功能](图结构dump功能.md)是不同的功能，二者可以单独使用，也可共同用于用户定位精度问题。

**图 1**  算子图模式dump功能介绍 

![](figures/算子图模式dump功能介绍.png)

## 使用约束

- 本功能仅支持max-autotune模式。
- 参数配置过程中若涉及文件路径，请确保路径确实存在，并且运行用户具有读、写操作权限。

## 使用方法

该功能通过[torchair.get\_npu\_backend](get_npu_backend.md)中compiler\_config配置，示例如下，仅供参考不支持直接拷贝运行，参数介绍参见[表1](#table1)。

```python
import torch_npu, torchair
config = torchair.CompilerConfig()
# data dump开关：[必选]
config.dump_config.enable_dump = True
# dump类型：[可选]，all代表dump所有数据
config.dump_config.dump_mode = "all"
# dump路径：[可选]，默认为当前执行目录
config.dump_config.dump_path = '/home/dump'
# 量化data dump开关：[可选]，是否采集量化前的dump数据
config.dump_config.quant_dumpable = True
# 保存dump的步数，否则每一步都会保存：[可选]
config.dump_config.dump_step = "0|1"
# 指定需要dump的算子：[可选]
config.dump_config.dump_layer = "Add_1Mul_1 Add2"
# 指定算子dump类型：[可选]，stats表示dump输出csv文件
config.dump_config.dump_data = "stats"
# dump配置文件的路径：[可选]，通过配置文件使能data dump。不建议与其他options一起使用，否则此配置将失效。
config.dump_config.dump_config_path = "/home/dump_config.json"
npu_backend= torchair.get_npu_backend(compiler_config=config)
opt_model = torch.compile(model, backend=npu_backend)
```

**表 1**  参数说明 <a name="table1"></a>


| 参数名 | 说明 |
| --- | --- |
| enable_dump | 是否开启数据dump功能，bool类型。<br>- False（默认值）：不开启数据dump。<br>- True：开启数据dump。 |
| dump_mode | dump数据模式，用于指定dump算子的输入还是输出数据，字符串类型。<br>- input：仅dump算子输入数据。<br>- output：仅dump算子输出数据。<br>- all（默认值）：同时dump算子输入和输出数据。 |
| dump_path | dump数据的存放路径，字符串类型，默认值为当前执行路径。支持配置绝对路径或相对路径（相对执行命令行时的当前路径）。<br>- 绝对路径配置以“/”开头，例如：/home/HwHiAiUser/output。<br>- 相对路径配置直接以目录名开始，例如：output。 |
| quant_dumpable | 如果是量化后的网络，可通过此参数控制是否采集量化前的dump数据，bool类型。<br>- False（默认值）：不采集量化前的dump数据。因为图编译过程中可能优化量化前的输入/输出，此时无法获取量化前的dump数据。<br>- True：开启此配置后，可确保能够采集量化前的dump数据。 |
| dump_step | 指定采集哪些迭代的dump数据。<br>字符串类型，默认值None，表示所有迭代都会产生dump数据。<br>多个迭代用“\|”分割，例如："0\|5\|10"；也可以用"\-"指定迭代范围，例如："0\|3\-5\|10"。 |
| dump_layer | 指定需要dump的算子名，多个算子名之间使用空格分隔，形如"Add1_in_0 Add2 Mul2"。算子名获取方法参见[dump_layer配置项说明](#dump_layer配置项说明)。<br>**说明**： 若指定的算子其输入涉及data算子，会同时将data算子信息dump出来。 |
| dump_data | 指定算子dump内容类型，字符串类型。<br>- tensor（默认值）：dump算子数据。<br>- stats：dump算子统计数据，保存结果为csv格式，文件中包含算子名称、输入/输出的数据类型、最大值、最小值等。<br>**说明**：通常dump数据量太大并且耗时长，可以先dump算子统计数据，根据统计数据识别可能异常的算子，再dump算子数据。 |
| dump_config_path（**推荐**） | 指定dump配置文件路径（json格式），字符串类型，无默认值。支持绝对/相对路径（即相对执行命令时的当前路径）。<br>上述dump options（除了enable_dump）均能通过json文件配置，功能模式支持模型Dump/单算子Dump、溢出算子Dump、算子Dump Watch模式等，具体使用方法和约束参见[dump_config_path配置项说明](#dump_config_path配置项说明)。 |

### dump_layer配置项说明

通过表中“dump\_layer”参数dump指定算子信息，算子名获取方法如下：

1. 通过DUMP_GE_GRAPH环境变量dump整个流程各阶段的图描述信息，建议取3（精简版dump，即只打印节点关系），详细介绍参见[《CANN 环境变量参考》](https://hiascend.com/document/redirect/CannCommunityEnvRef)中的“DUMP_GE_GRAPH”章节。

    ```bash
    export DUMP_GE_GRAPH=3
    ```
2. 设置环境变量后，在当前执行路径下生成ge\_proto\*.txt，示例如下，op中name字段为算子名。

    ```
    graph {
      name: "online_0"
      input: "Add1_in_0:0"
      input: "Add1_in_1:0"
      op {
        name: "Add1_in_0" 
        type: "Data"
        input: ""
        attr {
          key: "OUTPUT_IS_VAR"
          value {
            list {
              b: false
              val_type: VT_LIST_BOOL
            }
          }
        }
        ......
      }
      op{
        name: "Add2"
        type: "Data"
        ......
      }
    }
    ```

### dump\_config\_path配置项说明

通过表中“dump\_config\_path”参数指定dump配置json文件路径，基于json里的配置使能各种场景dump功能。

-   **使用说明**：   
    - 推荐本方式dump，上述dump options（除了enable_dump）均能通过json文件配置，并且dump options后续不再演进。
    - 当dump_config_path和上述dump options一起配置时，dump options优先级更高。一般不建议同时使用。
    - 注意：torch_npu默认已开启exception类信息的dump功能（即dump_scene参数，异常算子Dump配置） ，通过本功能配置的exception dump不会生效。
    - 对于大模型场景，通常dump数据量太大并且耗时长，建议dump_data配置为“stats”，开启算子统计功能，根据统计数据识别可能异常的算子后，再dump可能异常的算子。


-   **模型Dump**/**单算子Dump配置**：

    用于导出模型中每一层算子/单个算子的输入和输出数据，可以指定模型Dump或算子Dump进行比对，定位精度问题。

    -   模型Dump配置示例如下：

        ```
        {                                                                                            
        	"dump":{
        		"dump_list":[                                                                        
        			{	"model_name":"ResNet-101"
        			},
        			{                                                                                
        				"model_name":"ResNet-50",
        				"layer":[
        				      "conv1conv1_relu",
        				      "res2a_branch2ares2a_branch2a_relu",
        				      "res2a_branch1",
        				      "pool1"
        				] 
        			}  
        		],  
        		"dump_path":"/home/output",
                        "dump_mode":"output",
        		"dump_op_switch":"off",
                        "dump_data":"tensor"
        	}                                                                                        
        }
        ```

    -   单算子Dump配置示例如下：

        ```
        {
            "dump":{
                "dump_path":"/home/output",
                "dump_list":[], 
        	"dump_op_switch":"on",
                "dump_data":"tensor"
            }
        }
        ```

    配置说明如下：

    -   json文件支持配置的参数介绍请参考[《CANN 精度调试工具用户指南》](https://hiascend.com/document/redirect/CannCommunityToolAccucacy)中“准备离线模型dump数据文件”章节下**acl.json文件格式说明表**。
    -   若开启模型Dump/单算子Dump配置，则dump\_path必须配置，表示导出dump文件的存储路径。关于dump结果文件介绍请参考[《CANN 精度调试工具用户指南》](https://hiascend.com/document/redirect/CannCommunityToolAccucacy)中“准备离线模型dump数据文件”章节下**dump数据文件路径说明表**。

-   **溢出算子Dump配置**：

    用于导出模型中溢出算子的输入和输出数据，可用于分析溢出原因，定位精度问题。

    通过dump\_debug参数置为on表示开启溢出算子配置，示例如下：

    ```
    {
        "dump":{
            "dump_path":"output",
            "dump_debug":"on"
        }
    }
    ```

    配置说明如下：

    -   不配置dump\_debug或将dump\_debug配置为off表示不开启溢出算子配置。
    -   若开启溢出算子配置，则dump\_path必须配置，表示导出dump文件的存储路径。

        获取导出的数据文件后，文件的解析请参见[《CANN 精度调试工具用户指南》](https://hiascend.com/document/redirect/CannCommunityToolAccucacy)中“扩展功能>溢出算子数据采集与解析”章节。

        dump\_path支持配置绝对路径或相对路径：

        -   绝对路径配置以“/“开头，例如：/home。
        -   相对路径配置直接以目录名开始，例如：output。

    -   溢出算子Dump配置，不能与模型Dump配置或单算子Dump配置同时开启，否则会返回报错。
    -   仅支持采集AI Core算子的溢出数据。

-   **算子Dump Watch模式配置**：

    用于开启指定算子输出数据的观察模式，在定位部分算子精度问题且已排除算子本身的计算问题后，若怀疑被其它算子踩踏内存导致精度问题，可开启Dump Watch模式。

    将dump\_scene参数设置为watcher，开启算子Dump Watch模式，配置文件中的示例内容如下，配置效果为：

    （1）当执行完A算子、B算子时，会把C算子和D算子的输出Dump出来；

    （2）当执行完C算子、D算子时，也会把C算子和D算子的输出Dump出来。

    将（1）、（2）中的C算子、D算子的Dump文件进行比较，排查A算子、B算子是否会踩踏C算子、D算子的输出内存。

    ```
    {
        "dump":{
            "dump_list":[
                {
                    "layer":["A", "B"],
                    "watcher_nodes":["C", "D"]
                }
            ],
            "dump_path":"/home/",
            "dump_mode":"output",
            "dump_scene":"watcher"
        }
    }
    ```

    配置说明：

    -   若开启算子Dump Watch模式，则不支持同时开启溢出算子Dump（配置dump\_debug参数）或开启单算子/模型Dump（配置dump\_op\_switch参数），否则报错。
    -   在dump\_list中，通过layer参数配置可能踩踏其它算子内存的算子名称，通过watcher\_nodes参数配置可能被其它算子踩踏输出内存导致精度有问题的算子名称。
        -   若不指定layer，则模型内所有支持Dump的算子在执行后，都会将watcher\_nodes中配置的算子的输出Dump出来。
        -   layer和watcher\_nodes处配置的算子都必须是静态图、静态子图中的算子，否则不生效。
        -   若layer和watcher\_nodes处配置的算子名称相同，或者layer处配置的是集合通信类算子（算子类型以Hcom开头，例如HcomAllReduce），则只导出watcher\_nodes中所配置算子的dump文件。
        -   对于融合算子，watcher\_nodes处配置的算子名称必须是融合后的算子名称，若配置融合前的算子名称，则不导出dump文件。
        -   dump\_list内暂不支持配置model\_name。

    -   开启算子Dump Watch模式，则dump\_path必须配置，表示导出dump文件的存储路径。

        此处收集的dump文件无法通过文本工具直接查看其内容，若需查看dump文件内容，先将dump文件转换为numpy格式文件后，再通过Python查看numpy格式文件，详细转换步骤请参见[《CANN 精度调试工具用户指南》](https://hiascend.com/document/redirect/CannCommunityToolAccucacy)中“扩展功能>查看dump数据文件”章节。

        dump\_path支持配置绝对路径或相对路径：

        -   绝对路径配置以“/“开头，例如：/home。
        -   相对路径配置直接以目录名开始，例如：output。

    -   通过dump\_mode参数控制导出watcher\_nodes中所配置算子的哪部分数据，当前仅支持配置为output。

### 非dump\_config\_path配置项说明

通过表中“非dump\_config\_path”参数使能各种场景dump功能。dump结果文件存储在dump\_path参数指定的目录、\$\{dump\_path\}/\$\(worldsize\_global\_rank\)/\$\{time\}/\$\{device\_id\}/\$\{model\_name\}/\$\{model\_id\}/\$\{data\_index\}。若\$\{dump\_path\}配置为/home/dump，结果目录样例为“/home/dump/worldsize1\_global\_rank0/2024112145738/0/ge\_default\_20200808163719\_121/1/0”。

-   \$\{dump\_path\}：由dump\_path参数指定，默认为脚本所在路径。
-   \$\{worldsize\_global\_rank\}：表示集合通信相关的world\_size以及global\_rank信息，若只涉及单卡则表示为“worldsize1\_global\_rank0”。
-   \$\{time\}：dump数据文件保存的时间，格式为YYYYMMDDHHMMSS。
-   \$\{device\_id\}：设备ID。
-   \$\{model\_name\}：子图名称。可能存在多个文件夹，dump数据为计算图名称对应目录下的数据。如果\$\{model\_name\}出现了“.“、“/“、“\\“、空格时，转换为下划线表示。
-   \$\{model\_id\}：子图ID号。
-   \$\{data\_index\}：迭代数，用于保存对应迭代的dump数据。如果指定了dump\_step，则data\_index和dump\_step一致；如果不指定dump\_step，则data\_index一般从0开始计数，每dump一个迭代的数据，序号递增1。

### 解析dump数据文件

dump文件无法通过文本工具直接查看其内容，建议先将dump文件转换为numpy格式文件，再通过numpy官方提供的能力转为txt文档进行查看。详细操作指导请参考[《CANN 精度调试工具用户指南》](https://hiascend.com/document/redirect/CannCommunityToolAccucacy)中“扩展功能>查看dump数据文件”章节。

## 更多功能
对于上述[使用方法](#使用方法)提供的dump options，TorchAir提供了更灵活的dump算子范围。通过torchair.scope.data_dump接口实现，支持与上述所有dump options配套使用。

> **说明**：
> - 使用本接口时必须以with语句块形式调用，语句块内的算子信息均能被dump，具体参见下方调用示例。
> - 本接口与dump layer配置项指定的算子范围均能生效，dump算子范围为两者并集，产物目录与dump layer一致。
> - 本接口支持与上述所有dump配置项配合使用，产物目录基本一致。

```python
import torch
import torchair
import logging
from torchair import logger
logger.setLevel(logging.DEBUG)

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data0, data1):
        add_01 = torch.add(data0, data1)
        with torchair.scope.data_dump():
            sub_01 = torch.sub(data0, data1)
        return add_01, sub_01

input0 = torch.randn(2, 2, dtype=torch.float16).npu()
input1 = torch.randn(2, 2, dtype=torch.float16).npu()
config = torchair.CompilerConfig()
config.dump_config.enable_dump = True
config.dump_config.dump_layer = " Add "
npu_backend = torchair.get_npu_backend(compiler_config=config)
npu_mode = Network().npu()
npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
npu_out = npu_mode(input0, input1)
```
