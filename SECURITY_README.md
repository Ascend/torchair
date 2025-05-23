# 安全声明
## 系统安全加固
1. 建议用户在系统中配置开启ASLR（级别2），又称**全随机地址空间布局随机化**，可参考以下方式进行配置：
    ```
    echo 2 > /proc/sys/kernel/randomize_va_space
    ```

## 运行用户建议
出于安全性及权限最小化角度考虑，不建议使用root等管理员类型账户使用TorchAir。

## 文件权限控制
1. 建议用户在主机（包括宿主机）及容器中设置运行系统umask值为0027及以上，保障新增文件夹默认最高权限为750，新增文件默认最高权限为640。
2. 建议用户对个人数据、商业资产、源文件、推理过程中保存的各类文件等敏感内容做好权限管控。涉及场景如TorchAir安装目录权限管控、多用户使用共享数据集权限管控、GE图dump权限管控等场景，管控权限可参考表1进行设置。
3. TorchAir中集成GE图dump工具，使用时会在本地生成GE图，文件权限默认600，用户可根据实际需求对生成文件权限进行进阶管控。

**表1 文件（夹）各场景权限管控推荐最大值**
| 类型          | linux权限参考最大值 |
| --------------- | --------------------|
| 用户主目录                          |    750（rwxr-x---）                |
| 程序文件（含脚本文件、库文件等）      |    550（r-xr-x---）                |
| 程序文件目录                        |    550（r-xr-x---）                |
| 配置文件                            |    640（rw-r-----）                |
| 配置文件目录                        |    750（rwxr-x---）                |
| 日志文件（记录完毕或者已经归档）      |    440（r--r-----）                |
| 日志文件（正在记录）                 |    640（rw-r-----）                |
| 日志文件记录                        |    750（rwxr-x---）                |
| Debug文件                          |    640（rw-r-----）                |
| Debug文件目录                      |    750 (rwxr-x---)                 |
| 临时文件目录                       |     750（rwxr-x---）                |
| 维护升级文件目录                    |    770（rwxrwx---）                |
| 业务数据文件                       |     640（rw-r-----）                |
| 业务数据文件目录                   |     750（rwxr-x---）                |
| 密钥组件、私钥、证书、密文文件目录   |     700（rwx------）                |
| 密钥组件、私钥、证书、加密密文      |     600（rw-------）                |
| 加解密接口、加解密脚本             |     500（r-x------）                |


## 调试工具声明

1. TorchAir内集成生成GE图dump数据调试工具：
    - 集成原因：对标pytorch原生支持能力，提供NPU编译后端图dump能力，加速图模式开发调试过程。
    - 使用场景：默认不开启，如用户使用pytorch图模式且需要进行图分析时，可在模型推理脚本中调用生成GE图dump数据接口生成dump数据。
    - 风险提示：使用该功能会在本地生成GE图，用户需加强对相关dump数据的保护，请在图模式调试阶段使用该能力，调试完毕后及时关闭。

2. TorchAir内集成生成数据dump调试工具：
    - 集成原因：提供NPU图模式中间算子数据dump功能，用于比对精度，加速图模式开发调试过程。
    - 使用场景：默认不开启，如用户使用pytorch图模式且需要进行精度分析时，可在模型启动脚本中调用数据dump接口生成dump数据。
    - 风险提示：使用该功能会在本地生成计算数据，用户需加强对相关dump数据的保护，请在图模式调试阶段使用该能力，调试完毕后及时关闭。


## 构建安全声明

1. TorchAir在源码编译安装过程中，会下载依赖三方库并执行构建shell脚本，编译过程中会产生临时编译目录和程序文件。用户可根据需要，对源代码目录中的文件及文件夹进行权限管控，降低安全风险。


## 运行安全声明

1. 建议用户结合运行资源状况编写对应推理脚本。若推理脚本与资源状况不匹配，如数据集加载内存大小超出内存容量限制、推理脚本在本地生成数据超过磁盘空间大小等情况，可能引发错误并导致进程意外退出。
2. PyTorch、torch_npu和TorchAir在运行异常时会退出进程并打印报错信息，属于正常现象。建议用户根据报错提示定位具体错误原因，包括通过设定算子同步执行、查看CANN日志、解析生成的Core Dump文件等方式。


## 公网地址声明

代码涉及公网地址参考[public_address_statement.md](https://gitee.com/ascend/torchair/blob/master/public_address_statement.md)


## 公开接口声明

AscendPyTorch是PyTorch适配插件，TorchAir是为AscendPyTorch提供图模式能力的扩展库，支持用户使用PyTorch在昇腾设备上进行图模式推理。

参考[PyTorch社区公开接口规范](https://github.com/pytorch/pytorch/wiki/Public-API-definition-and-documentation)，  TorchairAir提供了对外的自定义接口。TorchAir提供在昇腾设备上的编译后端以对接Pytorch的原生torch.compile接口，因此，TorchAir对外提供了接口来实现此功能，具体接口可参考本文《torchair常用类和公开接口介绍》章节。如果一个函数看起来符合公开接口的标准且在文档中有展示，则该接口是公开接口。否则，使用该功能前可以在社区询问该功能是否确实是公开的或意外暴露的接口，因为这些未暴露接口将来可能会被修改或者删除。

TorchAir项目采用C++和Python联合开发，当前正式接口只提供Python接口，在TorchAir的二进制包中动态库不直接提供服务，暴露的接口为内部使用，不建议用户使用。

### TorchAir常用类和公开接口介绍
    
1. **TORCHAIR.GET_NPU_BACKEND**
    
    `torchair.get_npu_backend(*, compiler_config=None, custom_decompositions={})`
    
    获取能够在NPU上运行的图编译后端npu_backend，可以作为backend参数传入torch.compile，参考[图模式使能示例](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha002/devguide/moddevg/torchair/torchair_01_0007.html)。
    
    - Keyword Arguments
    - **compiler_config**(*CompilerConfig, optional*)- 配置项，具体可见torchair.CompilerConfig条目。
    - **custom_decomposition**(*Dict, optional*)- 手动指定模型运行时用到的decomposition。

2. **TORCHAIR.GET_COMPILER**
    
    `torchair.get_compiler(compiler_config=None)`
    
    获取能够在NPU上运行的图编译器。torchair.get_npu_backend()获取的图编译后端默认使用由本接口获取的图编译器。用户也可将获取的图编译器传入自定义的后端中。
    
    - Parameters
    - **compiler_config**(*CompilerConfig*)- 配置项，具体可见torchair.CompilerConfig条目。

3. **TORCHAIR C++层日志级别控制**
    1. **功能简介**
        
        环境变量TNG_LOG_LEVEL开启TorchAir C++的日志系统：
        - TNG_LOG_LEVEL：0, 日志级别DEBUG, 开启后输出DEBUG, INFO, WARNING, ERROR日志。
        - TNG_LOG_LEVEL：1, 日志级别INFO, 开启后输出INFO, WARNING, ERROR日志。
        - TNG_LOG_LEVEL：2, 日志级别WARNING, 开启后输出WARNING, ERROR日志。
        - TNG_LOG_LEVEL：3, 日志级别ERROR, 开启后输出ERROR日志。
        - TNG_LOG_LEVEL：4, 日志级别EVENT, 开启后输出ERROR, EVENT日志。
        - 默认为ERROR级别日志

    2. **使用方法**
        
        可以通过以下两种方式设置环境变量TNG_LOG_LEVEL：
        1. 在shell环境或shell脚本中设置环境变量
            ```shell
            export TNG_LOG_LEVEL=0
            ```
        2. python脚本中设置环境变量(注意：此方式设置环境变量需要早于import torchair)
            ```python
            import os
            os.environ['TNG_LOG_LEVEL'] = '0'
            ```

3. **TORCHAIR.LOGGER**
    
    `torchair.logger`
    
    TorchAir python层日志借助python的logging模块来实现，存在DEBUG、INFO、WARNING、ERROR四个级别。用法参照[TorchAir python层日志级别控制](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha002/devguide/moddevg/torchair/torchair_01_0013.html)。
    
    logger用到的函数继承了logging模块的函数，下面仅列出常用的函数和TorchAir新增的函数，其他函数用户可参考[logging模块文档](https://docs.python.org/3.8/library/logging.html)。
    1. **Functions**
        - **setLevel** -配置日志级别。
        - **warning** -打印WARNING级别信息
        - **warning_once** -打印WARNING级别信息一次，不重复打印。


4. **TORCHAIR.COMPILERCONFIG**
    
    `torchair.CompilerConfig`
    
    配置类。用户可以通过CompilerConfig配置以下功能：
    - **debug** 
        
        `graph_dump` : 用于配置是否导出图，以及导出图时图的格式导出等选项。

        - *dtype* -指定生成图的格式。默认为None，不导出图，可选参数为["txt", "pbtxt", "py"]。

        `data_dump` : 用于配置是否导出FX图Eager执行时每个aten IR的输出。

        - *filter* -自定义导出输出的过滤规则。默认不配置。
        - *type* -指定输出导出的数据类型。默认为None，不导出输出。可选参数为["npy"]，导出numpy格式的数据。

        `fx_summary` : 用于配置是否导出图中的aten算子信息。

        - *type* -指定导出的文件类型。默认为None，不导出。可选参数为["csv"]。
        - *skip_compile* -是否跳过GE的编译，以FX图Eager方式执行。默认为True，以FX图Egaer方式执行。可选参数为[True, False].

    - **aoe_config** 用于配置自动调优工具AOE。

        - *aoe_mode* -指定AOE模式。默认为None，不开启。可选参数为["2"]，"2"代表算子调优模式。
        - *work_path* -AOE调优工作目录，默认为当前目录。
        - *aoe_config_file* -AOE配置文件路径。

    - **dump_config** 用于配置数据dump接口。可参考[data dump功能](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha002/devguide/moddevg/torchair/torchair_01_0016.html)。

        - *enable_dump* 是否开启数据dump功能。默认为False，不开启。可选参数为[False, True]。
        - *dump_path* 数据dump路径。默认为当前路径。
        - *dump_mode* 数据dump类型。可选参数为["input", "output", "all"]，分别指dump输入、输出、所有数据，默认为"all"，dump所有数据。
        - *quant_dumpable* 是否开启dump量化前的输出。默认为False，不开启。可选参数为[False, True]。

    - **export** 用于配置导出air格式离线图时的选项。建议通过dynamo_export接口来进行配置，参考[export功能](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha002/devguide/moddevg/torchair/torchair_01_0015.html)。
        - *export_mode* -是否开启导出离线图。默认为False，不开启。
        - *export_path_dir* -导出离线图路径，默认为None，不导出。
        - *export_name* -导出离线图的名字，默认为None，不导出。

        `experimental` export的实验接口。
        
        - *enable_record_nn_module_stack* -配置是否携带nn_module_stack信息。默认为False，不携带。可选参数为[False, True]。
        - *auto_atc_config_generated* -是否开启使能自动生成atc的配置文件样例。默认为False，不开启。可选参数为[False, True]。

    - **fusion_config** 用于配置融合选项。

        - *fusion_switch_file* -指定融合配置文件的路径，用于指定融合规则的开启和关闭。

    - **ge_config** 用于配置ge option。

        - *enable_single_stream* -是否开启图单流执行。默认为False，不开启。可选参数为[True, False]。

    - **experimental_config** 用于配置实验功能。
        - *cc_parallel_enable* -是否开启计算与通信并行功能。默认为False，不开启。可选参数为[False, True]。
        - *keep_inference_input_mutation* -是否开启inplace场景入图性能优化。默认为True，开启。可选参数为[True, False]。
        - *memory_efficiency* -是否开启内存优化。默认为False，不开启。可选参数为[Fasle, True]。
        - *separate_atomic_clean* -是否集中清理网络中所有atomic算子（含有atomic属性的算子都是atomic算子）占用的内存。默认为True，开启。可选参数为[True, False]。
        - *frozen_parameter* -推理场景是否固定权重的内存地址，以降低图执行时输入地址刷新耗时。默认为False，不固定。可选参数为[True, False]。
        - *static_model_ops_lower_limit* -动静子图拆分场景性能优化。参数指定静态子图中包含节点的个数上限。默认为None，不进行优化。参数范围为[-1, 9223372036854775807)。
        - *jit_compile* -配置编译模式。默认为"auto"，在静态shape时调用二进制kernel函数，在动态shape时自动编译。可选参数为["auto"]。
        - *topology_sorting_strategy* -图模式编译节点遍历方式。默认为"DFS"，深度优先搜索。可选参数为["BFS", "DFS", "RDFS", "StableRDFS"]，"RDFS"为Reverse DFS。
        - *enable_ref_data* -如果存在ref类算子会改写输入内存的情况，构GE图时将Data类型修改为RefData类型，例如Assign、ScatterUpdate等算子。离线dynamo_export接口默认开启，在线时通过开关开启，默认为False，不开启。可选参数为[True, False]。
        - *tiling_schedule_optimize* -是否开启tiling下沉功能。可选参数为[True, False]，默认为False，不开启。

5. **TORCHAIR.DYNAMO_EXPORT**
    
    `torchair.dynamo_export(*args, model, export_path="export_file", export_name="export", dynamic=False, config=CompilerConfig(), **kwargs)`
    
    导出由TorchAir生成的离线图。

    1. **Parameters**
        
        - **model**(*torch.nn.Module*) -需要导出的model。
        - **export_path**(*str*) -离线图导出的文件存放路径，默认为当前路径下的"export_file"。
        - **export_name**(*str*) -离线图导出的名字，默认为"export"。
        - **dynamic**(*bool*) -设置导出静态模型还是动态模型。默认为False，静态模型。
        - **config**(*CompilerConfig*) -配置项，具体可见torchair.CompilerConfig条目。
        - **\*args,\*\*kwargs** -导出model时的样例输入，不同的输入可能导致model走入不同的分支，进而导致trace的图不同。应当选取执行推理时的典型值。

6. **TORCHAIR.USE_INTERNAL_FORMAT_WEIGHT**
    
    `torchair.use_internal_format_weight(model)`
    
    将模型的权重转换为NPU的私有格式。
    
    1. **Parameters**
        
        - **model**(*torch.nn.Module*) -用户的自定义模型。

## 通信安全加固

TorchAir在运行时依赖PyTorch及torch_npu，用户需关注通信安全加固，具体方式可参考torch_npu[通信安全加固](https://gitee.com/ascend/pytorch/blob/master/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E5%AE%89%E5%85%A8%E5%8A%A0%E5%9B%BA)。


## 通信矩阵

TorchAir在运行时依赖PyTorch及torch_npu，涉及通信矩阵，具体方式可参考torch_npu[通信矩阵](https://gitee.com/ascend/pytorch/blob/master/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E7%9F%A9%E9%98%B5)。

### 特殊场景

| 场景 | 使用方法 | 端口 | 是否有风险|
| --------------- | --------------- | --------------- | --------------- |
| 使用from_pretrained下载模型代码 | 调用from_pretrained函数 | 随机端口 | 无风险 |