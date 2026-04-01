# CompilerConfig类

该类用于构造传入torch.compiler backend的config参数，具体定义如下：

```python
class CompilerConfig(NpuBaseConfig):
    """Set CompilerConfig configuration"""

    def __init__(self):
        self.debug = _DebugConfig()
        self.export = _ExportConfig()
        self.dump_config = _DataDumpConfig()
        self.fusion_config = _FusionConfig()
        self.experimental_config = _ExperimentalConfig()
        self.inference_config = _InferenceConfig()
        self.ge_config = _GEConfig()
        self.mode = OptionValue("max-autotune", ["max-autotune"])
      
        super(CompilerConfig, self).__init__()
```
