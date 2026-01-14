# 自定义算子入图Meta注册报错“tensor's device must be 'meta', got xxx instead”

## 问题现象描述

图模式场景下执行带有自定义算子的推理脚本时，出现如下的报错日志：

```bash
torch._dynamo.exc.TorchRuntimeError: Failed running call_function custom_define.npu_custom_batch_matmul_cce(*(FakeTensor(..., device='npu:4', size=(3072, 2048), dtype=torch.int8), FakeTensor(..., device='npu:4', size=(2048, 4096), dtype=torch.int8), FakeTensor(..., device='npu:4', size=(4096,), dtype=torch.int64)), **{}): tensor's device must be `meta`, got cpu instead
```

## 可能的原因

Meta注册时构造的Tensor类型不符合要求。

## 处理步骤

1.  此类报错通常问题出现在Dynamo编译阶段，该阶段自定义算子主要的代码实现就是Meta注册。
2.  根据报错提示，先检查Meta注册代码，代码形如下方，可以发现确实返回了CPU Tensor。

    ```python
    @impl(m, "npu_custom_batch_matmul_cce", "Meta") 
    def npu_custom_batch_matmul_cce_meta(a, b, scale):   
        return torch.zeros(a.shape[0], b.shape[1])
    ```

3.  将返回的Tensor指定device为"meta"，问题即可解决。

    ```python
    @impl(m, "npu_custom_batch_matmul_cce", "Meta") 
    def npu_custom_batch_matmul_cce_meta(a, b, scale): 
        return torch.zeros(a.shape[0], b.shape[1], device="meta")
    ```

