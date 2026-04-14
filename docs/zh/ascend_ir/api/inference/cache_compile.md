# cache\_compile

## 功能说明

开启[模型编译缓存功能](../../features/advanced/compile_cache.md)能时需调用该接口实现编译缓存，降低成图编译耗时。

## 函数原型

```python
cache_compile(func, *, config: Optional[CompilerConfig] = None, backend: Optional[Any] = None, dynamic: bool = True, cache_dir: Optional[str] = None, global_rank: Optional[int] = None, tp_rank: Optional[int] = None, pp_rank: Optional[int] = None, ge_cache: bool = False, **kwargs) -> Callable
```

## 参数说明

|参数|输入/输出|说明|
|--|--|--|
|func|输入|模型编译缓存的函数。|
|config|输入|图编译配置，CompilerConfig类的实例化，默认情况下采用TorchAir自动生成的配置。|
|backend|输入|后端选择，默认值为"None"，通过torchair.get_npu_backend()获取。当同时传入config和backend时，会校验config和backend里的config是否一致，不一致报错。若只传入backend，则会将backend里的config取出赋值给入参config，然后传递下去，后续流程不变。|
|dynamic|输入|是否按照输入动态trace，bool类型。该参数继承了PyTorch原有特性，详细介绍请参考LINK。默认True，进行动态trace。|
|cache_dir|输入|缓存文件保存的根目录，支持绝对路径和相对路径。若cache_dir指定路径，且为绝对路径，则缓存文件路径为${cache_dir}/${model_info}/${func}。若cache_dir指定路径，且为相对路径，则缓存文件路径为${work_dir}/${cache_dir}/${model_info}/${func}。${cache_dir}默认为“.torchair_cache”（若无会新建），${work_dir}为当前工作目录，${model_info}为模型信息，${func}为封装的func函数。|
|global_rank|输入|分布式训练时的rank，int类型。取值范围为[0, world_size-1]，其中world_size是参与分布式训练的总进程数。一般情况下TorchAir会自动通过torch.distributed.get_rank()获取默认值。|
|tp_rank|输入|指张量模型并行rank，int类型，取值是global_rank中划分为TP域的rank id。|
|pp_rank|输入|指流水线并行rank，int类型，取值是global_rank中划分为PP域的rank id。|
|ge_cache|输入|是否缓存Ascend IR图编译结果，bool类型。除了优化Dynamo编译耗时，还支持优化Ascend IR图编译耗时。<br>True：开启缓存Ascend IR编译结果。生成的缓存路径是cache_dir指定的目录。如`/home/workspace/.torchair_cache/${model_info}/prompt/ge_cache_${时间戳}.om`。<br>False（默认值）：关闭缓存Ascend IR图编译结果。该功能受CANN包版本变更影响，用户根据实际情况手动开启。|否|
|**kwargs|输入|预留参数项，用于后续功能扩展。当前版本支持指定模型运行时使用的decomposition（将较大算子操作分解为小算子实现），通过custom_decompositions配置项实现。您可以参考调用示例的Add算子分解样例。|

## 返回值说明

返回一个Callable对象。

## 约束说明

- 如果图中包含依赖随机数生成器（RNG）的算子（例如randn、bernoulli、dropout等），不支持使用本功能。
- 本功能与torch.compile原始方案相比多了如下限制：
    - 缓存要与执行计算图一一对应，若重编译则缓存失效。
    - Guards阶段被跳过且不会触发JIT编译，要求生成模型的脚本和加载缓存的脚本一致。
    - CANN包跨版本缓存无法保证兼容性，如果版本升级，需要清理缓存目录并重新进行Ascend IR计算图编译生成缓存。

- cache_dir参数使用约束：
    - 请确保该参数指定的路径确实存在，并且运行用户具有读、写操作权限。
    - 若编译缓存的模型涉及多机多卡，缓存路径包含集合通信相关的world\_size以及global\_rank信息，缓存文件路径为`{work_dir}/{cache_dir}/{model_info}/world{world_size}global_rank{global_rank}/{func}/`。
    - `{model_info}`里会自动增加"aclgraphcache"关键词。

- ge\_cache参数使用约束：
    - 仅GE图模式场景（max-autotune）支持ge\_cache参数。
    - 默认情况下，ge\_cache=False（功能不开启），因受CANN包版本变更影响，用户需根据实际情况手动开启该功能。
    - CANN包跨版本的缓存无法保证兼容性，如果版本升级，需要清理缓存目录并重新GE编译生成缓存。
    - 在单算子和图混跑场景下，开启该功能会增加通信域资源开销，有额外显存消耗。

## 调用示例

- [Ascend IR编译缓存](../../features/advanced/compile_cache.md#ascend-ir编译缓存)
- Add算子custom\_decompositions简单示例：

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

    # ...
    ```
