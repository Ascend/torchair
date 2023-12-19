import os

os.environ["NPU_CORE_TYPE"] = "ai_core-ascend910B1"  # 要和stub实现、执行环境匹配
os.environ["ASCIR_NOT_READY"] = "1"  # 禁用ascir和pyautofuser，使用stub实现
import npu_extension_for_inductor
import torch


@torch.compile(dynamic=True)
def test_abs(x):
    return torch.abs(x)


x = torch.full((16, 512), -1.0, dtype=torch.float16)

print("Input:", x, flush=True)
y = test_abs(x)
print("Output:", y, flush=True)

assert torch.allclose(y, torch.abs(x))
print("Result check succeed!", flush=True)
