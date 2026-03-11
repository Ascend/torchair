import sys
import unittest
import torch

import npugraph_ex
from torchair_st_stub_aclgraph_utils import (
    StubNpu,
    patch_ops_npu_module,
    patch_torch_point_npu_module,
    patch_torch_npu_module,
    register_custom_ops,
)
from torchair_st_utils import generate_faked_module, register_is_npu

torch.utils.rename_privateuse1_backend("npu")
torch._register_device_module('npu', generate_faked_module())


def get_npugraph_ex_backend():
    def _exec(*args, **kwargs):       
        config = npugraph_ex.CompilerConfig()
        config.mode = "npugraph_ex"
        return npugraph_ex.get_npu_backend(compiler_config=config)(*args, **kwargs)
    return _exec

class NpugraphExSt(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.original_npu_module = None
        self.original_torch_point_npu_module = None
        self.original_torch_npu_module = None
        self.stub_module = StubNpu()
        register_custom_ops()
        register_is_npu()

    def setUp(self) -> None:
        self.original_npu_module = patch_ops_npu_module(self.stub_module)
        self.original_torch_point_npu_module = patch_torch_point_npu_module(self.stub_module)
        self.original_torch_npu_module = patch_torch_npu_module(self.stub_module)

        from npugraph_ex._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        self.call_bak = AclConcreteGraph.__call__
        from npugraph_ex.inference._cache_compiler import CacheBackend
        self.cachebackend_fw_compiler = CacheBackend.fw_compiler
        from npugraph_ex._acl_concrete_graph import cat_optimization
        self.optimize_cat_with_out_tensor = cat_optimization.optimize_cat_with_out_tensor

        from torch._dynamo import register_backend as _register_npu_backend
        npugraph_ex_backend = get_npugraph_ex_backend()
        _register_npu_backend(npugraph_ex_backend, "npugraph_ex")

        return super().setUp()

    def tearDown(self) -> None:
        torch.ops.npu = self.original_npu_module
        torch.npu = self.original_torch_point_npu_module
        sys.modules['torch_npu'] = self.original_torch_npu_module
        from npugraph_ex._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        AclConcreteGraph.__call__ = self.call_bak
        from npugraph_ex.inference._cache_compiler import CacheBackend
        CacheBackend.fw_compiler = self.cachebackend_fw_compiler
        from npugraph_ex._acl_concrete_graph import cat_optimization
        cat_optimization.optimize_cat_with_out_tensor = self.optimize_cat_with_out_tensor
        return super().tearDown()

    def test_aclgraph_capture_and_replay(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 2)
                self.linear2 = torch.nn.Linear(2, 2)

            def forward(self, x):
                ln1 = self.linear1(x)
                ln2 = self.linear2(x)
                return ln1 + ln2
        model = torch.compile(Model(), backend="npugraph_ex", options={"clone_input": False}, dynamic=False)
        x = torch.randn([3, 2])
        for i in range(2):
            model(x)
   

if __name__ == '__main__':
    unittest.main()
