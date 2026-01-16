# Dynamo导图功能

## 功能简介

离线推理场景下可通过[dynamo\_export](dynamo_export.md)接口，导出TorchAir生成的离线图（air格式）。

导出的推理模型不再依赖PyTorch框架，可直接由CANN软件栈加载执行，减少了框架调度带来的性能损耗，方便在不同的部署环境上移植。

## 使用约束

-   本功能仅支持max-autotune模式，暂不支持同时配置[固定权重类输入地址功能（Ascend IR）](固定权重类输入地址功能（Ascend-IR）.md)。
-   导出时需要保证被导出部分能构成一张图。
-   支持单卡和多卡场景下导出图，且支持导出后带AllReduce等通信类算子。
-   导出的air文件大小不允许超过2G（依赖的第三方库Protobuf存在限制导致）。
-   受Dynamo功能约束，不支持动态控制流if/else。

## 使用方法

dynamo\_export接口原型定义如下，详细的参数说明参见[dynamo\_export](dynamo_export.md)。

```python
def dynamo_export(*args, model: torch.nn.Module, export_path: str = "export_file", export_name: str = "export", dynamic: bool = False, config=CompilerConfig(), **kwargs)
```

关键参数说明如下：

-   **export\_path取值**：支持相对路径或绝对路径。

    > **说明：** 
    >请确保该参数指定的路径确实存在，并且运行用户具有读、写操作权限。

    -   若采用相对路径：在ATC（Ascend Tensor Compiler，昇腾张量编译器）编译、执行离线模型时，需要在相对路径的父路径中执行。
    -   若采用绝对路径：在ATC编译、执行离线模型时无路径限制，但是当编译好的模型拷贝至其他服务器环境时需要保证绝对路径相同，否则会找不到权重文件。

-   **config取值**：导离线图时支持配置config，具体参见下表。

    **表 1**  导图功能配置说明

    
    | 支持的功能 | 功能说明 | 配置说明 |
    | --- | --- | --- |
    | auto_atc_config_generated | 前端切分场景下（PyTorch模型导出时包含集合通信逻辑），是否开启自动生成ATC的json配置文件样例模板。 | - False（默认值）：不开启自动生成json模板，用户自行手动配置通信域信息。<br>- True：开启自动生成json模板。 |
    | enable_record_nn_module_stack | 导出的图是否携带nn_module_stack信息，方便后端切分（PyTorch模型导出时不含集合通信逻辑，而由GE添加集合通信逻辑）运用模板。<br> **说明**： <br>  - 前端脚本定义layer时，需要以数组的形式，即类似layer[0] = xxx，layer[1] = xxx。若不以数组形式表示变量名，相同模型结构被重复执行，从栈信息中将无法看出模型的layer结构，后端也无法切分。<br>  - record_nn_module_stack只有在model结构深度两层及以上才能获取到。 | - False（默认值）：导出图不带nn_module_stack信息。<br>  - True：导出图带nn_module_stack信息。 |

    config的配置示例如下：

    ```python
    import torch_npu, torchair
    config = torchair.CompilerConfig()
    # 开启自动生成ATC的json配置文件模板
    config.export.experimental.auto_atc_config_generated = True 
    # 携带nn_module_stack信息
    config.export.experimental.enable_record_nn_module_stack = True 
    ```

导图功能支持在单卡和多卡场景下导图，且支持导出的计算图携带AllReduce等通信类算子，导图结果参见[产物说明](#产物说明)，dynamo\_export接口的调用示例参见[使用示例](#使用示例)。

## 产物说明

dynamo\_export导图结果文件名默认为"export\_file"，支持用户自定义，产物目录如下：

```bash
└── export_file                         // 导出的文件夹，可自定义
    ├── dynamo.pbtxt                    // 导出的模型信息（可读格式）
    ├── export.air                      // 导出的模型文件（不可读），文件名默认值为“export”
    ├── weight_xx                       // 导出的权重文件
    ├── ......                          
    ├── model_relation_config.json      // 使能auto_atc_config_generated生成的文件
    └── numa_config.json                // 使能auto_atc_config_generated生成的文件
```

> **说明：** 
>对于导出的权重文件，其可能直接存入export.air，也可能单独存在weight\_xx文件中。
>-   当模型权重参数量不超过（2G-200MB）时，直接保存在export.air文件中，而权重存储路径、dtype等信息会被记录在dynamo.pbtxt的Const节点中。
>-   当模型权重参数量超过（2G-200MB）时，不存入export.air文件，会在export\_path路径中自动生成权重文件，（如[多卡场景下dynamo export示例](#demo2)中p1、p2文件，文件个数取决于网络中权重的定义）。同时权重存储路径、dtype等信息会被记录在dynamo.pbtxt的FileConstant节点中。

-   dynamo.pbtxt ：导出的图文件，文件名固定，支持用户直接查看。该文件记录了图节点、参数数据类型/数据维度等信息。
-   export.air：导出的图文件，不支持用户直接查看。该文件记录了图节点、参数数据类型/数据维度等信息，某些场景下还包含模型权重信息。
-   weight\_xx：导出的权重文件，可选产物。
-   model\_relation\_config.json ：可选产物，使能auto\_atc\_config\_generated生成的文件，表示多个切片模型间数据关联和分布式通信组关系的配置文件。
-   numa\_config.json：可选产物，使能auto\_atc\_config\_generated生成的文件，用于指定目标部署环境逻辑拓扑关系的配置文件。

对于export.air文件，ATC工具支持转换为可执行的\*.om模型文件（CANN能加载运行的格式），用户按需使用。命令形如：

```bash
atc --model=./export_file/export.air --framework=1 --output=./export_file/offline_module --soc_version=<soc_version>
```

-   --model：待转换的原始模型文件。
-   --framework：原始网络框架类型。本场景只能取1，代表原始模型是通过本功能导出的\*.air文件。
-   --output：生成的离线模型路径。
-   --soc\_version：昇腾AI处理器的型号，例如“Ascend_xxxyy_”。

详细工具指导请参见[《CANN ATC离线模型编译工具》](https://hiascend.com/document/redirect/CannCommunityAtc)。

## 使用示例

-   **单卡场景下dynamo export示例**：<a name="demo1"></a>

    ```python
    import torch, torch_npu, torchair
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(2, 2)
            self.linear2 = torch.nn.Linear(2, 2)
            for param in self.parameters():
                torch.nn.init.ones_(param)
        def forward(self, x, y):
            return self.linear1(x) + self.linear2(y)
    model = Model()
    x = torch.randn(2, 2)
    y = torch.randn(2, 2)
    torchair.dynamo_export(x, y, model=model, dynamic=False)
    ```

    执行如下命令查看导出结果：

    ```bash
    [root@localhost example_export]# tree
    ├── example1.py
    └── export_file             // 指定导出的文件夹，当文件夹不存在时会自动创建
        ├── dynamo.pbtxt       // 导出可读的图信息，用于debug
        ├── export.air        // 导出的模型文件，ATC编译时的输入。其中通过fileconst节点记录了权重所在的路径与文件名
    1 directory, 3 files
    ```

    由于权重文件小于2G，因此权重并未外置而是转为Const节点记录在图中。可以打开dynamo.pbtxt文件查看图接口信息。

-   **多卡场景下dynamo export示例**：<a name="demo2"></a>

    ```python
    import torch, os, torch_npu, torchair
    from torchair import CompilerConfig
    
    class AllReduceSingleGroup(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p1 = torch.nn.Parameter(torch.tensor([[1.1, 1.1], [1.1, 1.1]]))
            self.p2 = torch.nn.Parameter(torch.tensor([[2.2, 2.2], [3.3, 3.3]]))
    
        def forward(self, x, y):
            x = x + y + self.p1 + self.p2
            torch.distributed.all_reduce(x)
            return x
    
    def example(rank, world_size):
           torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
           x = torch.ones([2, 2], dtype=torch.int32)
           y = torch.ones([2, 2], dtype=torch.int32)
           mod = AllReduceSingleGroup()
           config = CompilerConfig()
           config.export.experimental.auto_atc_config_generated = True
           config.export.experimental.enable_record_nn_module_stack = True
           torchair.dynamo_export(x, y, model=mod, dynamic=True, export_path="./mp", export_name="mp_rank", config=config)
    
    def mp():
        world_size = 2
        torch.multiprocessing.spawn(example, args=(world_size, ), nprocs=world_size, join=True)
    
    if __name__ == '__main__':
         os.environ["MASTER_ADDR"] = "localhost"
         os.environ["MASTER_PORT"] = "29505"
         mp()
    ```

    执行如下命令查看导出结果：

    ```bash
    [root@localhost example_export]# tree
    ├── example1.py
    ├── example2.py
    └── mp                                       
        ├── model_relation_config.json          
        ├── mp_rank0.air                  // 第一张卡导出的模型文件
        ├── mp_rank1.air                  // 第二张卡导出的模型文件
        ├── numa_config.json          
        ├── rank_0                    // 第一张卡子目录
        │   ├── dynamo.pbtxt        // 导出可读的图信息，用于debug
        │   ├── p1                 // 导出的权重文件
        │   └── p2                 // 导出的权重文件
        └── rank_1
            ├── dynamo.pbtxt
            ├── p1
            └── p2
    3 directories, 12 files
    ```

    -   mp：指定导出的文件夹，即export\_path，当文件夹不存在时会自动创建。
    -   mp\_rank0/1：由指定的export\_name加上rank id拼接而成。
    -   model\_relation\_config.json、numa\_config.json：前端切分场景下，自动生成ATC编译的json配置文件模板。

        json中相关字段用户需根据实际情况修改，字段参数含义请参考[《CANN ATC离线模型编译工具》](https://hiascend.com/document/redirect/CannCommunityAtc)中的“--model\_relation\_config”章节与[《CANN ATC离线模型编译工具》](https://hiascend.com/document/redirect/CannCommunityAtc)中的“--cluster\_config”章节。针对多卡场景，item节点被生成在node\_id为0的表中，需要用户根据自己的需求手动划分至不同的node下。

    -   mp/rank\_0和mp/rank\_1：生成的子目录，里面存放着每张卡的dynamo.pbtxt图信息、权重文件（若权重没有被保存在air文件中）。
