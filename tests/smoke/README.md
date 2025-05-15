# Smoke test介绍

本章节提供torchair冒烟测试的介绍与demo，旨在帮助开发人员自测converter的功能及精度问题。

---
## 1. converter_st.py

在converter实现的函数上方，通过定义converter支持的场景，可以自动生成对应的测试用例。

> 图模式用例torch.compie参数设置：dynamic=None（静态图场景）, fullgraph=False。

以torch.ops.aten.add.Tensor为例，以下代码生成了7个对应用例，支持不同场景的测试。

```python
    @declare_supported([
        Support(F32(2, 2), F32(2, 2)), # 支持基础的torch.add()
        Support(F32(2, 2), F32(2, 1)), # 支持f32类型间的广播
        Support(F32(2, 2), F16(2, 1)), # 支持f32与f16类型的广播
        Support(F32(2, 2), F16(2, 2), alpha=2), # 支持带alpha入参场景
        Support(F32(2, 2), 2.0), # 支持与浮点常量的加法
        Support(F32(2, 2), 2), # 支持f32类型与整型常量的加法
        Support(F32(2, 2), 2, alpha=2.0), # 支持带alpha入参场景
    ])
```
执行测试用例的方式有2种：
1. 批量运行所有注册了declare_supported的算子测试用例
```shell
python3 smoke/converter_test.py
```
> 使用约束： npu_define的集合通信算子不会主动导入converter，因此无法自动生成用例，在此方式下无法生效，可通过本章节的hcom_st.py测试。
2. 测试满足某个前缀的算子用例
```shell
python3 smoke/converter_test.py aten.add.Tensor
```
> 使用约束：当传参指定了具体算子时，会尝试加载自定义算子converter，如果未实现对应的@declare_supported场景注册则会报错RuntimeError：Cannot find testcase match prefix。
> npu_define的集合通信算子限制仍然存在，无法直接测试对应convereter，可通过本章节的hcom_st.py测试。
 
## 2. hcom_st.py

本脚本测试集合通信算子在图模式下是否能正确入图，可通过两种方式验证：
1. 验证所有的集合通信算子patch到自定义算子如torch.ops.npu_define.all_gather.default之后，能正确入图且精度与eager mode一致。
```shell
python3 smoke/hcom_st.py
```
> 此方式会执行所有的用例，耗时较长。当测试某个算子时开销较大，此时推荐使用第2种方式。
2. 验证某个通信算子，如torch.distributed.all_gather patch到自定义算子torch.ops.npu_define.all_gather.default之后入图功能OK且精度正常。
```shell
cd smoke
python3 -m unittest -v hcom_st.HcomTest.test_allgather 
```