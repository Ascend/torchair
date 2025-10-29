# Torchair Roadmap 2025 H2

我们在2025年H2阶段的目标是提升Torchair的易用性，提供基于torch.compile+Aclgraph的最佳实践，并推动优秀开源框架更顺畅地适配TorchAir生态，从而繁荣整个技术生态。具体的研发Roadmap如下：

---

## Aclgraph

- [x] [***August***]Torchair-Aclgraph支持编译缓存。
- [x] [***August***]Torchair-Aclgraph支持静态Kernel；针对纯静态 shape 网络或 shape 变化较少的动态shape网络，可在部署阶段使用算子编译工具（op_compiler）预先编译生成静态 Kernel 包，从而显著提升算子调用性能。
- [x] [***August***]Torchair-Aclgraph支持多级内存复用。
- [x] [***September***]Torchair-Aclgraph支持输入地址变化时进行 recapture。
- [ ] [***October***]Torchair-Aclgraph Pass功能补齐；完成包括冗余copy消除、多原地算子functionalize优化、多原地算子re-inplace优化、冗余slice消除等关键特性集成。
- [ ] [***October***]Torchair-Aclgraph提供dump能力。
- [ ] [***December***]Torchair-Aclgraph支持Super Kernel，将多个算子的task任务融合成一个task任务下发，减少task调度耗时。
- [ ] [***December***]Torchair-Aclgraph支持训练。

## Ascend IR Converter

- [x] [***September***]Ascend IR支持结构化自定义算子的自动converter实现，自动生成converter及Ascend IR的python代码，显著增加torch入图易用性。

## Pass增强

- [ ] [***October***]提供Stream间Kernel执行时序控制能力&支持自定义注册fx pass[#ICW9JJ](https://gitee.com/ascend/Torchair/issues/ICW9JJ?from=project-issue)。

## DFX

- [ ] [***October***]优化并增强当前Torchair的dump功能易用性。

## 特性文档

- [ ] [***November***]撰写并补充Torchair-Aclgraph关键特性文档，包括技术原理详解与性能开销分析。

## 框架对接

- [ ] SGLang，SGLang框架支持Torchair-Aclgraph。
- [ ] Vllm，Vllm框架支持 torch.compile。
- [ ] CANN Recipe实现0 Day支持 torch.compile。

---

# Torchair Roadmap 2025 H1

---

## 性能优化

- [x] [***March***]Ascend IR支持Super Kernel，提供识别Super Kernel范围的能力。
- [x] [***April***]Ascend IR图下发头开销优化。

## Aclgraph

- [x] [***March***]支持Torchair-Aclgraph。
- [x] [***April***]Torchair-Aclgraph支持tiling下沉场景。
- [x] [***May***]Torchair-Aclgraph支持集合通信。
- [x] [***May***]Torchair-Aclgraph支持aot_autograd流程配置keep_inference_input_mutations=True。
- [x] [***May***]Torchair-Aclgraph支持micro-batch；对于一些可以并行的场景，支持在图模式下做分流并行操作。
- [x] [***June***]Torchair-Aclgraph支持配置torch.compile中dynamic为True，提供支持泛化shape的能力。

## 集合通信能力增强

- [x] [***January***]集合通信zero-copy，消除集合通信过程中的卡内拷贝。
- [x] [***April***]AllGatherV和ReduceScatterV支持图模式，支持多卡间输入tensor第0维不等长的场景。

## Ascend IR Converter

- [x] [***March***]Torchair支持通过lite注册的converter优先级高于默认converter优先级。

## 编译缓存

- [x] [***May***]cache_compile支持传入backend，来指定compile config。
- [x] [***May***]cache_compile增加对执行时输入信息的校验。

## DFX

- [x] [***May***]精度比对时支持执行eager-mode，在fx图的基础上先支持fx图执行eager-mode，可以方便问题定界。
- [x] [***May***]Torchair支持图dump功能指定dump路径，支持fx dump功能指定dump路径并对dump文件命名优化。

## 私有格式

- [x] [***June***] AscendIR图模式支持私有格式与分档功能。

---