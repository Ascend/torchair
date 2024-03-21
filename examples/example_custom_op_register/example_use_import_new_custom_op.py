import torch
import torch_npu
import new_custom_op
import torchair


###############################test########################################
class PlugInAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, input2):
        return torch.ops.npu_define.custom_op(input1, input2)


def test_add():
    torch.npu.set_device(0)
    input1 = torch.arange(4).npu()
    input2 = torch.arange(4).npu()

    model = PlugInAdd().npu()

    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    model = torch.compile(model, backend=npu_backend, dynamic=True)

    with torch.no_grad():
        output = model(input1, input2)

    assert output.equal(input1 + input2)


if __name__ == '__main__':
    test_add()
