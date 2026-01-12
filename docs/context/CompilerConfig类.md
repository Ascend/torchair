# CompilerConfig类

该类用于构造传入torch.compiler backend的config参数，具体定义如下：

```python
class CompilerConfig(NpuBaseConfig):
    """Set CompilerConfig configuration"""

    def __init__(self):
        self.debug = _DebugConfig()            # 配置debug调试类功能
        self.export = _ExportConfig()          # 配置离线导图相关功能
        self.dump_config = _DataDumpConfig()   # 配置图模式下数据dump功能
        self.fusion_config = _FusionConfig()   # 配置图融合相关功能
        self.experimental_config = _ExperimentalConfig()    # 配置各种试验功能
        self.inference_config = _InferenceConfig()          # 配置推理场景相关功能
        self.ge_config = _GEConfig()                        # 配置Ascend IR图相关功能
        self.aclgraph_config = _AclGraphConfig()            # 配置aclgraph相关功能
        self.mode = OptionValue("max-autotune", ["max-autotune", "reduce-overhead"])  # 配置图模式执行器
      
        super(CompilerConfig, self).__init__()
```
