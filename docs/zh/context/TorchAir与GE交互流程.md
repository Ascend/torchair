# TorchAir与GE交互流程

编译和执行阶段TorchAir和GE的衔接时序如下图：

![](figures/zh-cn_image_0000002512422333.png)

1.  Converter前FX图优化：如特殊inplace\_pattern优化、sym\_input优化、view\_to\_reshape优化。
2.  Converter：实现Aten IR转换为Ascend IR。
3.  Converter后FX图优化：如死节点消除、符号输入转换为ge.Data等。
4.  优化后的图，经过反序列化加载得到GE Model对象。
5.  首次执行，触发向GE Session添加graph的动作。
6.  首次执行，触发向GE Session的graph的编译动作。
7.  TorchAir根据编译结果，生成对应的Executor执行器。
8.  调用GE图执行接口，如ExecuteGraphWithStreamAsync。

