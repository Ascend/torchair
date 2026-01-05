import sys
import logging
import unittest

import torch
import torch._inductor.config as inductor_config
from functorch import make_fx
import _privateuse1_backend

import torchair
from torchair.configs.compiler_config import CompilerConfig
from torchair.core.utils import logger
from torchair._acl_concrete_graph.graph_pass import _mutated_input_reinplace
from torchair._utils.graph_utils import add_stream_label_to_node_meta
from torchair_st_stub_aclgraph_utils import (
    StubNpu,
    patch_ops_npu_module,
    patch_torch_point_npu_module,
    patch_torch_npu_module,
)
from torchair_st_utils import generate_faked_module

logger.setLevel(logging.DEBUG)

try:
    from torch._dynamo.utils import ReinplaceCounters
except ImportError:
    ReinplaceCounters = None
    logger.debug("function[ReinplaceCounters] is not support on torch < 2.6")

try:
    from torch._higher_order_ops.auto_functionalize import auto_functionalized
except ImportError:
    auto_functionalized = None
    logger.debug("function[auto_functionalized] is not support on torch < 2.6")
    
try:
    from torch._higher_order_ops.auto_functionalize import auto_functionalized_v2
except ImportError:
    auto_functionalized_v2 = None
    logger.debug("function[auto_functionalized_v2] is not support on torch < 2.6")

try:
    from torch._inductor.fx_passes.post_grad import decompose_auto_functionalized
except ImportError:
    decompose_auto_functionalized = None
    logger.debug("function[decompose_auto_functionalized] is not support on torch < 2.6")


npu_device = _privateuse1_backend.npu_device()
torch.utils.rename_privateuse1_backend("npu")
torch._register_device_module('npu', generate_faked_module())


def num_reinplacing_failures():
    return ReinplaceCounters.get_total_missed()


def miss_inplaced_bytes():
    return ReinplaceCounters.get_total_missed_bytes()


lib = torch.library.Library("custom", "FRAGMENT")
lib.define("sin(Tensor x, Tensor(a!) result) -> None")


@torch.library.impl(lib, "sin", "CPU")
def sin(x, result):
    result.copy_(x.sin())


@torch.library.impl(lib, "sin", "Meta")
def sin(x, result):
    pass


lib.define("sin_cos(Tensor x, Tensor(a!) out_sin, Tensor(b!) out_cos) -> None")


@torch.library.impl(lib, "sin_cos", "CPU")
def sin_cos(x, out_sin, out_cos):
    out_sin.copy_(x.sin())
    out_cos.copy_(x.cos())


@torch.library.impl(lib, "sin_cos", "Meta")
def sin_cos(x, out_sin, out_cos):
    pass


lib.define("boo(Tensor(a!) x) -> None")


@torch.library.impl(lib, "boo", "CPU")
def boo(x):
    x.sin_()


@torch.library.impl(lib, "boo", "Meta")
def boo(x):
    pass


class AclGraphSt(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

        self.original_npu_module = None
        self.original_torch_point_npu_module = None
        self.original_torch_npu_module = None
        self.stub_module = StubNpu()

    def setUp(self) -> None:
        self.original_npu_module = patch_ops_npu_module(self.stub_module)
        self.original_torch_point_npu_module = patch_torch_point_npu_module(self.stub_module)
        self.original_torch_npu_module = patch_torch_npu_module(self.stub_module)
        from torchair._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        self.call_bak = AclConcreteGraph.__call__

        return super().setUp()

    def tearDown(self) -> None:
        torch.ops.npu = self.original_npu_module
        torch.npu = self.original_torch_point_npu_module
        sys.modules['torch_npu'] = self.original_torch_npu_module
        from torchair._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        AclConcreteGraph.__call__ = self.call_bak

        return super().tearDown()
    
    @unittest.skipIf(torch.__version__ < "2.6", "torch._dynamo.utils.ReinplaceCounters and torch._dynamo.utils.ReInplaceTrigger is unsupported when torch < 2.6")
    def test_counters_functionalize_old(self):
        ReinplaceCounters.clear()

        def f(x):
            out = torch.empty_like(x)
            _, new_out = auto_functionalized(torch.ops.custom.sin.default, x=x, result=out)
            y = out * new_out
            return new_out, y

        x = torch.randn(3)
        gm = make_fx(f, tracing_mode="fake")(x)
        add_stream_label_to_node_meta(gm)
        _mutated_input_reinplace(gm)

        # We shouldn't have been able to reinplace `out` because it was used after
        # auto_functionalized. Note that this usually doesn't happen in practice;
        # we're artificially creating this example to test the counter.
        # IF THIS NUMBER GOES TO ZERO, PLEASE FIND ANOTHER EXAMPLE
        self.assertEqual(num_reinplacing_failures(), 1)
        self.assertEqual(miss_inplaced_bytes(), 12)
    
    @unittest.skipIf(torch.__version__ < "2.6", "torch.ops.higher_order.auto_functionalized_v2 is unsupported when torch < 2.6")
    def test_counters_functionalize_v2(self):
        ReinplaceCounters.clear()

        def f(x):
            out = torch.empty_like(x)
            _, new_out = auto_functionalized_v2(
                torch.ops.custom.sin.default,
                x=x,
                _result_base_index=0,
                _result_size=(3,),
                _result_stride=(1,),
                _result_storage_offset=0,
                _all_bases=[out],
            )
            y = out * new_out
            return new_out, y

        x = torch.randn(3)
        gm = make_fx(f, tracing_mode="fake")(x)
        add_stream_label_to_node_meta(gm)
        _mutated_input_reinplace(gm)

        # We shouldn't have been able to reinplace `out` because it was used after
        # auto_functionalized. Note that this usually doesn't happen in practice;
        # we're artificially creating this example to test the counter.
        # IF THIS NUMBER GOES TO ZERO, PLEASE FIND ANOTHER EXAMPLE
        self.assertEqual(num_reinplacing_failures(), 1)
    
    def get_not_inplaced_count(self, graph):
        counter = 0
        auto_functionalized_found = False
        for node in graph.nodes:
            if (node.target == torch.ops.higher_order.auto_functionalized) or (
                node.target == torch.ops.higher_order.auto_functionalized_v2
            ):
                auto_functionalized_found = True
                counter += len(node.meta["only_clone_these_tensors"])
        if not auto_functionalized_found:
            raise RuntimeError(f"Expected auto_functionalized to be True, but got {auto_functionalized_found}")
        return counter

    @unittest.skipIf(torch.__version__ < "2.6", "torch.ops.higher_order.auto_functionalized_v2 is unsupported when torch < 2.6")
    def test_view_inplaced_functionalize_v2(self):
        def f(arg0_1):
            torch.ops.aten.select.int(arg0_1, 0, 0)
            _auto_functionalized = auto_functionalized_v2(
                torch.ops.custom.boo.default,
                _x_base_index=0,
                _x_size=(3,),
                _x_stride=(1,),
                _x_storage_offset=0,
                _all_bases=[arg0_1],
            )
            getitem_1 = _auto_functionalized[1]
            torch.ops.aten.copy_.default(arg0_1, getitem_1)
            return ()

        x1 = torch.randn(3)
        gm = make_fx(f, tracing_mode="fake")(x1)
        add_stream_label_to_node_meta(gm)
        _mutated_input_reinplace(gm)
        self.assertEqual(self.get_not_inplaced_count(gm.graph), 0)

        actual = gm.print_readable(print_output=False)
        expected = """class f(torch.nn.Module):
    def forward(self, arg0_1_1: "f32[3]"):
        # No stacktrace found for following nodes
        select: "f32[]" = torch.ops.aten.select.int(arg0_1_1, 0, 0);  select = None
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.custom.boo.default, _x_base_index = 0, _x_size = (3,), _x_stride = (1,), _x_storage_offset = 0, _all_bases = [arg0_1_1])
        getitem = auto_functionalized_v2[0];  getitem = None
        getitem_1: "f32[3]" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        copy_: "f32[3]" = torch.ops.aten.copy_.default(arg0_1_1, getitem_1);  arg0_1_1 = getitem_1 = copy_ = None
        return ()

        """
        self.assertMultiLineEqual(actual.strip(), expected.strip())

        decompose_auto_functionalized(gm.graph)
        actual = gm.print_readable(print_output=False)
        expected = """class f(torch.nn.Module):
    def forward(self, arg0_1_1: "f32[3]"):
        # No stacktrace found for following nodes
        select: "f32[]" = torch.ops.aten.select.int(arg0_1_1, 0, 0);  select = None
        as_strided_default: "f32[3]" = torch.ops.aten.as_strided.default(arg0_1_1, [3], [1], 0)
        boo_default = torch.ops.custom.boo.default(as_strided_default);  as_strided_default = boo_default = None
        copy_: "f32[3]" = torch.ops.aten.copy_.default(arg0_1_1, arg0_1_1);  arg0_1_1 = copy_ = None
        return ()

        """
        self.assertMultiLineEqual(actual.strip(), expected.strip())
    
    @unittest.skipIf(torch.__version__ < "2.6", "torch.ops.higher_order.auto_functionalized_v2 is unsupported when torch < 2.6")
    def test_view_inplaced2_functionalize_v2(self):
        def f(arg0_1):
            _select = torch.ops.aten.select.int(arg0_1, 0, 0)
            another_view = arg0_1[2]
            _auto_functionalized = auto_functionalized_v2(
                torch.ops.custom.boo.default,
                _x_base_index=0,
                _x_size=(3,),
                _x_stride=(1,),
                _x_storage_offset=0,
                _all_bases=[arg0_1],
            )
            getitem_1 = _auto_functionalized[1]
            _copy = torch.ops.aten.copy_.default(arg0_1, getitem_1)
            return another_view

        x1 = torch.randn(3)
        gm = make_fx(f, tracing_mode="fake")(x1)
        add_stream_label_to_node_meta(gm)
        _mutated_input_reinplace(gm)
        self.assertEqual(self.get_not_inplaced_count(gm.graph), 0)

        actual = gm.print_readable(print_output=False)
        expected = """class f(torch.nn.Module):
    def forward(self, arg0_1_1: "f32[3]"):
        # No stacktrace found for following nodes
        select: "f32[]" = torch.ops.aten.select.int(arg0_1_1, 0, 0);  select = None
        select_1: "f32[]" = torch.ops.aten.select.int(arg0_1_1, 0, 2)
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.custom.boo.default, _x_base_index = 0, _x_size = (3,), _x_stride = (1,), _x_storage_offset = 0, _all_bases = [arg0_1_1])
        getitem = auto_functionalized_v2[0];  getitem = None
        getitem_1: "f32[3]" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        copy_: "f32[3]" = torch.ops.aten.copy_.default(arg0_1_1, getitem_1);  arg0_1_1 = getitem_1 = copy_ = None
        return select_1

        """
        self.assertMultiLineEqual(actual.strip(), expected.strip())

        decompose_auto_functionalized(gm.graph)
        actual = gm.print_readable(print_output=False)
        expected = """class f(torch.nn.Module):
    def forward(self, arg0_1_1: "f32[3]"):
        # No stacktrace found for following nodes
        select: "f32[]" = torch.ops.aten.select.int(arg0_1_1, 0, 0);  select = None
        select_1: "f32[]" = torch.ops.aten.select.int(arg0_1_1, 0, 2)
        as_strided_default: "f32[3]" = torch.ops.aten.as_strided.default(arg0_1_1, [3], [1], 0)
        boo_default = torch.ops.custom.boo.default(as_strided_default);  as_strided_default = boo_default = None
        copy_: "f32[3]" = torch.ops.aten.copy_.default(arg0_1_1, arg0_1_1);  arg0_1_1 = copy_ = None
        return select_1

        """
        self.assertMultiLineEqual(actual.strip(), expected.strip())
    
    @unittest.skipIf(torch.__version__ < "2.6", "torch.ops.higher_order.auto_functionalized_v2 is unsupported when torch < 2.6")
    def test_views_not_inplaced_functionalize_v2(self):
        def f(arg0_1):
            _select = torch.ops.aten.select.int(arg0_1, 0, 0)
            another_view = arg0_1[2]
            _auto_functionalized = auto_functionalized_v2(
                torch.ops.custom.boo.default,
                _x_base_index=0,
                _x_size=(3,),
                _x_stride=(1,),
                _x_storage_offset=0,
                _all_bases=[arg0_1],
            )
            getitem_1 = _auto_functionalized[1]
            use_another_view = another_view * 10
            _copy = torch.ops.aten.copy_.default(arg0_1, getitem_1)
            return use_another_view

        x1 = torch.randn(3)
        gm = make_fx(f, tracing_mode="fake")(x1)
        add_stream_label_to_node_meta(gm)
        _mutated_input_reinplace(gm)
        self.assertEqual(self.get_not_inplaced_count(gm.graph), 1)

        actual = gm.print_readable(print_output=False)
        expected = """class f(torch.nn.Module):
    def forward(self, arg0_1_1: "f32[3]"):
        # No stacktrace found for following nodes
        select: "f32[]" = torch.ops.aten.select.int(arg0_1_1, 0, 0);  select = None
        select_1: "f32[]" = torch.ops.aten.select.int(arg0_1_1, 0, 2)
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.custom.boo.default, _x_base_index = 0, _x_size = (3,), _x_stride = (1,), _x_storage_offset = 0, _all_bases = [arg0_1_1])
        getitem = auto_functionalized_v2[0];  getitem = None
        getitem_1: "f32[3]" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        mul: "f32[]" = torch.ops.aten.mul.Tensor(select_1, 10);  select_1 = None
        copy_: "f32[3]" = torch.ops.aten.copy_.default(arg0_1_1, getitem_1);  arg0_1_1 = getitem_1 = copy_ = None
        return mul

        """
        self.assertMultiLineEqual(actual.strip(), expected.strip())

        decompose_auto_functionalized(gm.graph)
        actual = gm.print_readable(print_output=False)
        expected = """class f(torch.nn.Module):
    def forward(self, arg0_1_1: "f32[3]"):
        # No stacktrace found for following nodes
        select: "f32[]" = torch.ops.aten.select.int(arg0_1_1, 0, 0);  select = None
        select_1: "f32[]" = torch.ops.aten.select.int(arg0_1_1, 0, 2)
        as_strided_default: "f32[3]" = torch.ops.aten.as_strided.default(arg0_1_1, [3], [1], 0)
        clone_default: "f32[3]" = torch.ops.aten.clone.default(as_strided_default);  as_strided_default = None
        as_strided_default_1: "f32[3]" = torch.ops.aten.as_strided.default(clone_default, [3], [1], 0);  clone_default = None
        as_strided_default_2: "f32[3]" = torch.ops.aten.as_strided.default(as_strided_default_1, [3], [1], 0)
        boo_default = torch.ops.custom.boo.default(as_strided_default_2);  as_strided_default_2 = boo_default = None
        mul: "f32[]" = torch.ops.aten.mul.Tensor(select_1, 10);  select_1 = None
        copy_: "f32[3]" = torch.ops.aten.copy_.default(arg0_1_1, as_strided_default_1);  arg0_1_1 = as_strided_default_1 = copy_ = None
        return mul

        """
        self.assertMultiLineEqual(actual.strip(), expected.strip())

    @unittest.skipIf(torch.__version__ < "2.6", "torch.ops.higher_order.auto_functionalized_v2 is unsupported when torch < 2.6")
    def test_views_not_inplaced2_functionalize_v2(self):
        def f(arg0_1):
            _select = torch.ops.aten.select.int(arg0_1, 0, 0)
            _another_view = arg0_1[2]
            _auto_functionalized = auto_functionalized_v2(
                torch.ops.custom.boo.default,
                _x_base_index=0,
                _x_size=(3,),
                _x_stride=(1,),
                _x_storage_offset=0,
                _all_bases=[arg0_1],
            )
            _getitem_1 = _auto_functionalized[1]
            return

        x1 = torch.randn(3)
        gm = make_fx(f, tracing_mode="fake")(x1)
        add_stream_label_to_node_meta(gm)
        _mutated_input_reinplace(gm)
        self.assertEqual(self.get_not_inplaced_count(gm.graph), 1)

        actual = gm.print_readable(print_output=False)
        expected = """class f(torch.nn.Module):
    def forward(self, arg0_1_1: "f32[3]"):
        # No stacktrace found for following nodes
        select: "f32[]" = torch.ops.aten.select.int(arg0_1_1, 0, 0);  select = None
        select_1: "f32[]" = torch.ops.aten.select.int(arg0_1_1, 0, 2);  select_1 = None
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.custom.boo.default, _x_base_index = 0, _x_size = (3,), _x_stride = (1,), _x_storage_offset = 0, _all_bases = [arg0_1_1]);  arg0_1_1 = None
        getitem = auto_functionalized_v2[0];  getitem = None
        getitem_1: "f32[3]" = auto_functionalized_v2[1];  auto_functionalized_v2 = getitem_1 = None
        return None

        """
        self.assertMultiLineEqual(actual.strip(), expected.strip())

        decompose_auto_functionalized(gm.graph)
        actual = gm.print_readable(print_output=False)
        expected = """class f(torch.nn.Module):
    def forward(self, arg0_1_1: "f32[3]"):
        # No stacktrace found for following nodes
        select: "f32[]" = torch.ops.aten.select.int(arg0_1_1, 0, 0);  select = None
        select_1: "f32[]" = torch.ops.aten.select.int(arg0_1_1, 0, 2);  select_1 = None
        as_strided_default: "f32[3]" = torch.ops.aten.as_strided.default(arg0_1_1, [3], [1], 0);  arg0_1_1 = None
        clone_default: "f32[3]" = torch.ops.aten.clone.default(as_strided_default);  as_strided_default = None
        as_strided_default_1: "f32[3]" = torch.ops.aten.as_strided.default(clone_default, [3], [1], 0);  clone_default = None
        as_strided_default_2: "f32[3]" = torch.ops.aten.as_strided.default(as_strided_default_1, [3], [1], 0);  as_strided_default_1 = None
        boo_default = torch.ops.custom.boo.default(as_strided_default_2);  as_strided_default_2 = boo_default = None
        return None

        """
        self.assertMultiLineEqual(actual.strip(), expected.strip())
    
    @unittest.skipIf(torch.__version__ < "2.6", "torch.ops.higher_order.auto_functionalized_v2 is unsupported when torch < 2.6")
    def test_views_not_inplaced3_functionalize_v2(self):
        def f(arg0_1):
            a = torch.ones(10)
            another_view = a[2]
            _auto_functionalized = auto_functionalized_v2(
                torch.ops.custom.boo.default,
                _x_base_index=0,
                _x_size=(),
                _x_stride=(),
                _x_storage_offset=0,
                _all_bases=[a],
            )
            _getitem_1 = _auto_functionalized[1]
            return another_view
         
        x1 = torch.randn(3)
        gm = make_fx(f, tracing_mode="fake")(x1)
        add_stream_label_to_node_meta(gm)
        _mutated_input_reinplace(gm)
        self.assertEqual(self.get_not_inplaced_count(gm.graph), 1)

        actual = gm.print_readable(print_output=False)
        expected = """class f(torch.nn.Module):
    def forward(self, arg0_1_1: "f32[3]"):
        # No stacktrace found for following nodes
        ones: "f32[10]" = torch.ops.aten.ones.default([10], device = device(type='cpu'), pin_memory = False)
        select: "f32[]" = torch.ops.aten.select.int(ones, 0, 2)
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.custom.boo.default, _x_base_index = 0, _x_size = (), _x_stride = (), _x_storage_offset = 0, _all_bases = [ones]);  ones = None
        getitem = auto_functionalized_v2[0];  getitem = None
        getitem_1: "f32[10]" = auto_functionalized_v2[1];  auto_functionalized_v2 = getitem_1 = None
        return select

        """
        self.assertMultiLineEqual(actual.strip(), expected.strip())

        decompose_auto_functionalized(gm.graph)
        actual = gm.print_readable(print_output=False)
        expected = """class f(torch.nn.Module):
    def forward(self, arg0_1_1: "f32[3]"):
        # No stacktrace found for following nodes
        ones: "f32[10]" = torch.ops.aten.ones.default([10], device = device(type='cpu'), pin_memory = False)
        select: "f32[]" = torch.ops.aten.select.int(ones, 0, 2)
        as_strided_default: "f32[10]" = torch.ops.aten.as_strided.default(ones, [10], [1], 0);  ones = None
        clone_default: "f32[10]" = torch.ops.aten.clone.default(as_strided_default);  as_strided_default = None
        as_strided_default_1: "f32[10]" = torch.ops.aten.as_strided.default(clone_default, [10], [1], 0);  clone_default = None
        as_strided_default_2: "f32[]" = torch.ops.aten.as_strided.default(as_strided_default_1, [], [], 0);  as_strided_default_1 = None
        boo_default = torch.ops.custom.boo.default(as_strided_default_2);  as_strided_default_2 = boo_default = None
        return select

        """
        self.assertMultiLineEqual(actual.strip(), expected.strip())

    @unittest.skipIf(torch.__version__ < "2.6", "torch.ops.higher_order.auto_functionalized_v2 is unsupported when torch < 2.6")
    def test_multiple_mutations(self):
        ReinplaceCounters.clear()

        def f(x, out):
            torch.ops.custom.sin.default(x, out)
            torch.ops.custom.sin.default(out, out)
            torch.ops.custom.sin.default(out, out)
            return out

        x = torch.randn(3)
        out = torch.randn(3)

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)
        model = torch.compile(f, backend=aclgraph_backend)

        self.assertEqual(num_reinplacing_failures(), 0)
    
    @unittest.skipIf(torch.__version__ < "2.6", "torch.ops.higher_order.auto_functionalized_v2 is unsupported when torch < 2.6")
    def test_multiple_intermediate(self):
        ReinplaceCounters.clear()

        def f(x):
            out = torch.empty_like(x)
            sin(x, out)
            sin(out, out)
            sin(out, out)
            return out
        
        x = torch.randn(3)

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)
        model = torch.compile(f, backend=aclgraph_backend)
        
        self.assertEqual(num_reinplacing_failures(), 0)
    
    @unittest.skipIf(torch.__version__ < "2.6", "torch._dynamo.utils.ReinplaceCounters and torch._dynamo.utils.ReInplaceTrigger is unsupported when torch < 2.6")
    def test_multi_output_intermeditate(self):
        ReinplaceCounters.clear()

        def f(x):
            out1 = torch.empty_like(x)
            out2 = torch.empty_like(x)
            torch.ops.custom.sin_cos.default(x, out1, out2)
            return out1, out2, x**2

        x = torch.randn(3)

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)
        model = torch.compile(f, backend=aclgraph_backend)
        
        self.assertEqual(num_reinplacing_failures(), 0)
        self.assertEqual(miss_inplaced_bytes(), 0)
    
    @unittest.skipIf(torch.__version__ < "2.6", "torch.ops.higher_order.auto_functionalized_v2 is unsupported when torch < 2.6")
    def test_lists_functionalize_v2(self):
        ReinplaceCounters.clear()
        with inductor_config.patch({"enable_auto_functionalized_v2": True}):
            
            _lib = torch.library.Library("custom", "FRAGMENT")
            _lib.define("mutate_op(Tensor(a!)[] y) -> None")

            @torch.library.impl(_lib, "mutate_op", "CPU")
            def mutate_op(y):
                y[0].add_(2)
                y[1].add_(3)

            @torch.library.impl(_lib, "mutate_op", "Meta")
            def mutate_op(y):
                pass

            def f(b):
                torch.ops.custom.mutate_op.default([b[0], b[1]])

            x1 = torch.tensor([0.3, 0.4])

            config = CompilerConfig()
            config.mode = "reduce-overhead"
            aclgraph_backend = torchair.get_npu_backend(compiler_config=config)
            model = torch.compile(f, backend=aclgraph_backend)
            
            # We can inplace the base y. no clones emitted.
            self.assertEqual(num_reinplacing_failures(), 0)
            self.assertEqual(miss_inplaced_bytes(), 0)

    @unittest.skipIf(torch.__version__ < "2.6", "torch._dynamo.utils.ReinplaceCounters and torch._dynamo.utils.ReInplaceTrigger is unsupported when torch < 2.6")
    def test_lists_old_functionalize(self):
        ReinplaceCounters.clear()
        with inductor_config.patch({"enable_auto_functionalized_v2": False}):

            _lib = torch.library.Library("custom", "FRAGMENT")
            _lib.define("mutate_op(Tensor(a!)[] y) -> None")

            @torch.library.impl(_lib, "mutate_op", "CPU")
            def mutate_op(y):
                y[0].add_(2)
                y[1].add_(3)

            @torch.library.impl(_lib, "mutate_op", "Meta")
            def mutate_op(y):
                pass

            def f(b):
                torch.ops.custom.mutate_op.default([b[0], b[1]])

            x1 = torch.tensor([0.3, 0.4])

            config = CompilerConfig()
            config.mode = "reduce-overhead"
            aclgraph_backend = torchair.get_npu_backend(compiler_config=config)
            model = torch.compile(f, backend=aclgraph_backend)
            
            self.assertEqual(num_reinplacing_failures(), 0)
            self.assertEqual(miss_inplaced_bytes(), 0)


if __name__ == '__main__':
    unittest.main()
