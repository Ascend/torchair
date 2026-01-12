# cache\_compile

## 功能说明

开启“**模型编译缓存功能**”时需要调用该接口实现编译缓存。

## 函数原型

```python
cache_compile(func, *, config: Optional[CompilerConfig] = None, backend: Optional[Any] = None, dynamic: bool = True, cache_dir: Optional[str] = None, global_rank: Optional[int] = None, tp_rank: Optional[int] = None, pp_rank: Optional[int] = None, custom_decompositions: Optional[dict] = None, ge_cache: bool = False, **kwargs) -> Callable
```

## 参数说明


| 参数 | 输入/输出 | 说明 | 是否必选 |
| --- | --- | --- | --- |
| func | 输入 | 模型编译缓存的函数。 | 是 |
| config | 输入 | 图编译配置，[CompilerConfig类](CompilerConfig类.md)的实例化，默认情况下采用TorchAir自动生成的配置。<br> **说明**： <br>  -  本功能仅支持max-autotune模式，暂不支持同时配置[Dynamo导图功能](Dynamo导图功能.md)、[RefData类型转换功能](RefData类型转换功能.md)。<br>  -  reduce-overhead模式下，可与其他功能同时配置。 | 否 |
| backend | 输入 | 后端选择，默认值为"None"，通过torchair.get_npu_backend()获取。<br>  -  当同时传入config和backend时，会校验config和backend里的config是否一致，不一致报错。<br>  -  若只传入backend，则会将backend里的config取出赋值给入参config，然后传递下去，后续流程不变。 | 否 |
| dynamic | 输入 | 是否按照输入动态trace，bool类型。<br>该参数继承了PyTorch原有特性，详细介绍请参考[LINK](https://pytorch.org/docs/stable/generated/torch.compile.html#torch-compile)。<br>默认True，进行动态trace。 | 否 |
| cache_dir | 输入 | 缓存文件保存的根目录，支持绝对路径和相对路径。<br>  -  若cache_dir指定路径，且为绝对路径，则缓存文件路径为\${cache\_dir}/\${model\_info}/\${func}。<br>  -  若cache_dir指定路径，且为相对路径，则缓存文件路径为\${work\_dir}/\${cache\_dir}/\${model\_info}/\${func}。<br>  **说明**：\${cache\_dir}默认为“.torchair\_cache”（若无会新建），\${work\_dir}为当前工作目录，\${model\_info}为模型信息，\${func}为封装的func函数。 | 否 |
| global_rank | 输入 | 分布式训练时的rank，int类型。取值范围为[0, world\_size-1]，其中world\_size是参与分布式训练的总进程数。<br>一般情况下TorchAir会自动通过torch.distributed.get_rank()获取默认值。 | 否 |
| tp_rank | 输入 | 指张量模型并行rank，int类型，取值是global\_rank中划分为TP域的rank id。 | 否 |
| pp_rank | 输入 | 指流水线并行rank，int类型，取值是global\_rank中划分为PP域的rank id。 | 否 |
| custom_decompositions | 输入 | 手动指定模型运行时使用的decomposition（将较大算子操作分解为小算子实现）。<br>用户根据实际情况配置，具体请参见[Add算子示例](#调用示例)。 | 否 |
| ge_cache | 输入 | 是否缓存Ascend IR图编译结果，bool类型。除了优化Dynamo编译耗时，还支持优化Ascend IR图编译耗时。<br>-  True：开启缓存Ascend IR编译结果。生成的缓存路径是cache\_dir指定的目录。如/home/workspace/.torchair\_cache/\${model\_info}/prompt/ge\_cache\_\${时间戳}.om。<br>-  False（默认值）：关闭缓存Ascend IR图编译结果。该功能受CANN包版本变更影响，用户根据实际情况手动开启。 | 否 |
| * | 输入 | 预留参数项，用于后续功能扩展。 | 否 |

## 返回值说明

返回一个Callable对象。

## 约束说明

- 如果图中包含依赖随机数生成器（RNG）的算子（例如randn、bernoulli、dropout等），不支持使用本功能。
- 本功能与torch.compile原始方案相比多了如下限制：
    - 缓存要与执行计算图一一对应，若重编译则缓存失效。
    - Guards阶段被跳过且不会触发JIT编译，要求生成模型的脚本和加载缓存的脚本一致。
    - CANN包跨版本缓存无法保证兼容性，如果版本升级，需要清理缓存目录并重新进行Ascend IR计算图编译生成缓存。

- reduce-overhead模式下，本接口仅支持如下产品：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>

- cache_dir参数使用约束：
   - 请确保该参数指定的路径确实存在，并且运行用户具有读、写操作权限。
   - 若编译缓存的模型涉及多机多卡，缓存路径包含集合通信相关的world_size以及global_rank信息，缓存文件路径为\$\{work\_dir}/\$\{cache\_dir\}/\$\{model\_info\}/world\$\{world\_size\}global\_rank\$\{global\_rank\}/\$\{func\}/。
   - reduce-overhead模式下，\${model\_info}里会自动增加"aclgraphcache"关键词。

- ge_cache参数使用约束：
   - ge_cache仅支持max-autotune模式，reduce-overhead模式配置ge_cache=True不会生效。
   - 默认情况下，ge_cache=False（功能不开启），因受CANN包版本变更影响，用户需根据实际情况手动开启该功能。 
   - CANN包跨版本的缓存无法保证兼容性，如果版本升级，需要清理缓存目录并重新GE编译生成缓存。
   - 在单算子和图混跑场景下，开启该功能会增加通信域资源开销，有额外显存消耗。

## 调用示例

- [reduce-overhead模型编译缓存示例](模型编译缓存功能（aclgraph）.md#Dynamo编译缓存)

- [max-autotune模型编译缓存示例](模型编译缓存功能（Ascend-IR）.md#Dynamo编译缓存)

- Add算子custom_decompositions示例：

    ```python
    # 注册算子分解函数
    import torch, torch_npu, torchair
    from torch._decomp import get_decompositions, register_decomposition
    @register_decomposition(torch.ops.aten.add.default)
    def test_add_decomp(t1, t2):
        return t1 + t2
    
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # 将被分解算子的列表通过custom_decompositions传入
            self.cached = torchair.inference.cache_compile(self.inner_forward,
                custom_decompositions=get_decompositions([torch.ops.aten.add.default]))
    
        def inner_forward(self, t1, t2):
            return torch.ops.aten.add(t1, t2)
    
        def forward(self, t1, t2):
            return self.cached(t1, t2)
            ......
    ```