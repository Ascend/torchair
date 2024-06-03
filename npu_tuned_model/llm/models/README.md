## 模型迁移

[模型迁移指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha003/devguide/moddevg/torchair/torchair_01_0001.html)

**注意**：模型迁移，先把eager模式跑通，然后再进行下面的图模式修改

此适配点主要是加入走pytorch图模式分支。
对于chatglm3模型部分如图，无需对下上述文件进行修改。
```python
# transformers/generation/utils.py中greedy_search函数
exe_mode = os.getenv("EXE_MODE", "dynamo")
dynamic_compile = eval(os.getenv("DYNAMIC_COMPILE", "False"))

if exe_mode == "dynamo":
    logging.info("Start to run model in dynamo mode, dynamic=%s, fullgraph=%s, backend=npu" % (dynamic_compile,
                                                                                               True))
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    npu_backend = tng.get_npu_backend(compiler_config=config)
    self = torch.compile(self, dynamic=dynamic_compile, fullgraph=True, backend=npu_backend)
else:
    logging.info("Start to run model in eager(HOST API) mode")

# 在模型执行前后添加torch.npu.synchronize()
torch.npu.synchronize()
outputs = self(
    **model_inputs,
    return_dict=True,
    output_attentions=output_attentions,
    output_hidden_states=output_hidden_states,
)
torch.npu.synchronize()
```

- torchair提供了NPU的图构造/图编译/图执行能力。相关能力全部集成到NPU图的后端，在使用torch.compile接口时，指定NPU图后端来使能。同时提供了开关和config控制图的编译和执行流程。
- 在使用NPU图后端的时候，torchair提供了静态图和动态图两种图执行的能力。根据dynamic参数决定是否走动态图。

