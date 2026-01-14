# TorchAir C++层日志

## 功能简介

TorchAir的C++层日志（图执行过程中的日志信息）主要通过环境变量TNG\_LOG\_LEVEL开启，实现C++层功能调试和问题定位。

支持的日志级别如下：

-   TNG\_LOG\_LEVEL：0，日志级别DEBUG，开启后输出DEBUG、INFO、WARNING、ERROR日志。
-   TNG\_LOG\_LEVEL：1，日志级别INFO，开启后输出INFO、WARNING、ERROR日志。
-   TNG\_LOG\_LEVEL：2，日志级别WARNING，开启后输出WARNING、ERROR日志。
-   TNG\_LOG\_LEVEL：3，日志级别ERROR，开启后输出ERROR日志。
-   TNG\_LOG\_LEVEL：4，日志级别EVENT，开启后输出ERROR、EVENT日志。

环境变量TNG\_LOG\_LEVEL的默认值为“3“。

## 使用方法

-   方式1：在安装完软件包后，以运行用户身份登录环境，并设置环境变量TNG\_LOG\_LEVEL，以DEBUG级别为例。

    ```bash
    export TNG_LOG_LEVEL=0
    ```

-   方式2：通过python脚本设置环境变量，以DEBUG级别为例。

    > **说明：** 
    >该方式设置环境变量时，需早于import torchair，否则影响日志打印。

    ```python
    import os
    os.environ['TNG_LOG_LEVEL'] = '0'
    ```

C++侧Debug日志样例如下：

```text
[DEBUG] TORCHAIR(2250956,python):2025-02-06-15:44:53.084.205 [static_npu_graph_executor.cpp:46]2250956 Assemble aten device input 0 at::Tensor(shape=[1, 1, 2, 8], dtype='float', device=npu:0, addr=0x12c041200000) to ge::Tensor(storage shape=[1, 1, 2, 8], origin shape=[1, 1, 2, 8], storage format=ND, origin format=ND, dtype=DT_FLOAT, device=NPU, addr=0x12c041200000)
[DEBUG] TORCHAIR(2250956,python):2025-02-06-15:44:53.084.323 [static_npu_graph_executor.cpp:46]2250956 Assemble aten device input 1 at::Tensor(shape=[1], dtype='long int', device=npu:0, addr=0x12c041200200) to ge::Tensor(storage shape=[1], origin shape=[1], storage format=ND, origin format=ND, dtype=DT_INT64, device=NPU, addr=0x12c041200200)
[DEBUG] TORCHAIR(2250956,python):2025-02-06-15:44:53.084.379 [static_npu_graph_executor.cpp:46]2250956 Assemble aten device input 2 at::Tensor(shape=[1, 1, 1, 8], dtype='float', device=npu:0, addr=0x12c041200400) to ge::Tensor(storage shape=[1, 1, 1, 8], origin shape=[1, 1, 1, 8], storage format=ND, origin format=ND, dtype=DT_FLOAT, device=NPU, addr=0x12c041200400)
[DEBUG] TORCHAIR(2250956,python):2025-02-06-15:44:53.084.487 [static_npu_graph_executor.cpp:130]2250956 Create empty output 0 at::Tensor(shape=[1, 1, 2, 8], dtype='float', device=npu:0, addr=0x12c041201000)
[DEBUG] TORCHAIR(2250956,python):2025-02-06-15:44:53.084.527 [static_npu_graph_executor.cpp:138]2250956 Assemble torch output 0 at::Tensor(shape=[1, 1, 2, 8], dtype='float', device=npu:0, addr=0x12c041201000) to ge::Tensor(storage shape=[1, 1, 2, 8], origin shape=[1, 1, 2, 8], storage format=ND, origin format=ND, dtype=DT_FLOAT, device=NPU, addr=0x12c041201000)
[DEBUG] TORCHAIR(2250956,python):2025-02-06-15:44:53.084.591 [concrete_graph/session.cpp:238]2250956 Start to session load graph 0
[DEBUG] TORCHAIR(2250956,python):2025-02-06-15:44:53.090.305 [concrete_graph/session.cpp:250]2250956 Start to session execute graph 0
[INFO] TORCHAIR(2250956,python):2025-02-06-15:44:53.090.459 [static_npu_graph_executor.cpp:256]2250956 Static npu graph executor run graph 0 on stream 0x3345cca0 successfully.
```

