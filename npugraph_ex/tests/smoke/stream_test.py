from contextlib import contextmanager

import logging
import unittest
import re
import io

import torch

import npugraph_ex
from npugraph_ex._acl_concrete_graph import replace_stream_event
from npugraph_ex.core.utils import logger

torch._logging.set_logs(dynamo=logging.INFO)
torch.manual_seed(7)
torch.npu.manual_seed_all(7)
logger.setLevel(logging.DEBUG)


@contextmanager
def capture_logger():
    """
    Context manager to capture python logger output.

    Usage:
    with capture_logger() as stdout:
        # code that prints to stdout by logger
    captured_output = stdout.getvalue()
    """

    stream_io = io.StringIO()
    handler = logging.StreamHandler(stream_io)
    logging.getLogger().addHandler(handler)

    try:
        yield stream_io
    finally:
        captured_output = stream_io.getvalue()
        logging.getLogger().removeHandler(handler)

        # Optionally print the captured output if you want to see it
        print("Captured logger message:\n", captured_output)


class StreamTest(unittest.TestCase):
    def setUp(self) -> None:
        self.optimize_fx_bak = npugraph_ex.npu_fx_compiler._optimize_fx
        from npugraph_ex._acl_concrete_graph import cat_optimization
        self.optimize_cat_bak = cat_optimization.optimize_cat_with_out_tensor
        if not hasattr(torch.npu, "fake_record_stream"):
            from aclgraph_test import patch_dynamo
            patch_dynamo()
        replace_stream_event.GraphCounter.set_graph_id(-1)
        return super().setUp()

    def tearDown(self) -> None:
        if self.optimize_fx_bak is not None:
            npugraph_ex.npu_fx_compiler._optimize_fx = self.optimize_fx_bak
        if self.optimize_cat_bak is not None:
            from npugraph_ex._acl_concrete_graph import cat_optimization
            cat_optimization.optimize_cat_with_out_tensor = self.optimize_cat_bak
        return super().tearDown()
    
    def test_npu_stream_switch_with_stream_closure(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, in1, in2, in3, in4):
                s1 = torch.npu.Stream()
                s2 = torch.npu.Stream()
                add_result = torch.add(in1, in2)
                with torch.npu.stream(s1):
                    mm_result1 = torch.mm(in3, in4)
                    with torch.npu.stream(s2):
                        mm_result2 = torch.mm(in3, in4)
                return add_result, mm_result1, mm_result2

        options = {"clone_input": False, "input_inplace_pass": True}
        model = Model()
        opt_model = torch.compile(model, backend="npugraph_ex", options=options, fullgraph=True, dynamic=False)
        x = torch.randn([3, 3])
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            opt_model(x, x, x, x)

        self.assertTrue(
            any("tagged_event_record_default = torch.ops.air.tagged_event_record.default"
                in log for log in cm.output),
            f"Expected no DEBUG log 'Try to capture node names[tagged_event_record_default] "
            f"type[air.tagged_event_record.default]' in logs: {cm.output}")
        self.assertTrue(
            any("tagged_event_wait_default = torch.ops.air.tagged_event_wait.default"
                in log for log in cm.output),
            f"Expected no DEBUG log 'Try to capture node names[tagged_event_wait_default] "
            f"type[air.tagged_event_wait.default]' in logs: {cm.output}")

        self.assertTrue(
            any("tagged_event_record_on_stream_default = torch.ops.air.tagged_event_record_on_stream.default"
                in log for log in cm.output),
            f"Expected no DEBUG log 'Try to capture node names[tagged_event_record_on_stream_default] "
            f"type[air.tagged_event_wait.default]' in logs: {cm.output}")

        self.assertTrue(
            any("tagged_event_record_on_stream_default_1 = torch.ops.air.tagged_event_record_on_stream.default"
                in log for log in cm.output),
            f"Expected no DEBUG log 'Try to capture node names[tagged_event_record_on_stream_default_1] "
            f"type[air.tagged_event_wait.default]' in logs: {cm.output}")

        self.assertTrue(
            any("tagged_event_wait_on_stream_default = torch.ops.air.tagged_event_wait_on_stream.default"
                in log for log in cm.output),
            f"Expected no DEBUG log 'Try to capture node names[tagged_event_wait_on_stream_default] "
            f"type[air.tagged_event_wait.default]' in logs: {cm.output}")


        self.assertTrue(
            any("tagged_event_wait_on_stream_default_1 = torch.ops.air.tagged_event_wait_on_stream.default"
                in log for log in cm.output),
            f"Expected no DEBUG log 'Try to capture node names[tagged_event_wait_on_stream_default_1] "
            f"type[air.tagged_event_wait.default]' in logs: {cm.output}")


    def test_npu_stream_switch_with_tagged_event(self):
        from npugraph_ex._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        options = {"clone_input": False, "input_inplace_pass": True}

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, in1, in2, in3, in4):
                stream1 = torch.npu.Stream()
                stream2 = torch.npu.Stream()
                ext_event1 = torch.npu.Event()
                ext_event2 = torch.npu.Event()

                add_result = torch.add(in1, in2)
                ext_event1.record()
                ext_event2.record()
                with torch.npu.stream(stream1):
                    ext_event1.wait()
                    mm_result1 = torch.mm(in3, in4)
                    with torch.npu.stream(stream2):
                        ext_event2.wait()
                        mm_result2 = torch.mm(in3, in4)
                return add_result, mm_result1, mm_result2

        def check_graph(concrete_graph):
            event_record = 0
            for node in concrete_graph.fx_graph.graph.nodes:
                if str(node.target) == "aten.mm.default":
                    assert str(node.prev.target) == "air.tagged_event_wait.default"
                if str(node.target) == "air.tagged_event_record.default":
                    event_record += 1
            assert event_record == 5, f"expect event record count is 5, but got {event_record}"

        def decorator(call):
            def wrapper(*args, **kwargs):
                assert len(args) >= 3
                check_graph(args[0])
                return tuple([args[0], args[1], args[2]])

            return wrapper

        AclConcreteGraph.__call__ = decorator(AclConcreteGraph.__call__)

        model = Model()
        opt_model = torch.compile(model, backend="npugraph_ex", options=options, fullgraph=True, dynamic=False)
        x = torch.randn([3, 3])
        y = torch.randn([3, 3])
        z = torch.randn([3, 3])
        w = torch.randn([3, 3])
        opt_model(x, y, z, w)

    def test_record_stream(self):

        class StubTensor:
            def record_stream(self, stream):
                return

        origin = torch.Tensor.record_stream
        torch.Tensor.record_stream = StubTensor.record_stream

        def func():
            stream1 = torch.npu.Stream()
            A = torch.ones([100, 100])
            mm_input = torch.randn(3200, 32000)
            with torch.npu.stream(stream1):
                for _ in range(10):  # 延长secend stream执行时间，使得A.add(1)晚于主流C.add_(2)计算
                    out = mm_input * mm_input
                B = A.add(1)
                # A在secend_stream参与计算，同时主流对A所在内存进行释放，此时，需要插入record_stream延长A所在内存的生命周期，
                # 避免被提前释放, 导致出现A在secend stream计算时数据错误改写的问题
                A.record_stream(stream1)
            del A # 在主流上释放A内存，如果在second_stream流上没有插入record_stream，则可能导致A内存被提前释放，
                  # 而正好C又恰好申请到了A相同的内存地址，如果C.add_(2)在B = A.Add(1)之前执行完，则B的结果将是错误的
            C = torch.ones([100, 100])
            C.add_(2)
            return B, C

        options = {"inplace_pass": True}
        model = torch.compile(func, backend="npugraph_ex", options=options, dynamic=False)

        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            model()

        self.assertTrue(
            any("call_function[target=torch.ops.air.record_tagged_stream.default]" in log for log in cm.output),
            f"Expected DEBUG log 'call_function[target=torch.ops.air.record_tagged_stream.default]' in logs: {cm.output}"
        )
        torch.Tensor.record_stream = origin

    @unittest.skipIf(torch.__version__ < "2.5", "reinplace is unsupported when torch < 2.5")
    def test_reinplace_pass_disblabled_with_multi_stream(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                a = x.clone()
                b = a.add(1)
                stream1 = torch.npu.Stream()
                with torch.npu.stream(stream1):
                    y.mul_(2)
                return b, y

        model = Model()
        options = {"clone_input": False}
        model = torch.compile(model, backend="npugraph_ex", options=options, dynamic=False)
        x = torch.randn([8, 8])
        y = torch.randn([8, 8])
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            model(x, y)

        self.assertFalse(
            any("call_function[target=torch.ops.aten.add_.Tensor]" in log for log in cm.output),
            f"Expected DEBUG log 'call_function[target=torch.ops.aten.add_.Tensor]' in logs: {cm.output}"
        )

        self.assertTrue(
            any("call_function[target=torch.ops.aten.mul_.Tensor]" in log for log in cm.output),
            f"Expected DEBUG log 'call_function[target=torch.ops.aten.mul_.Tensor]' in logs: {cm.output}"
        )

    def test_npu_multi_stream_with_multi_graph(self):
        from npugraph_ex._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        options = {"clone_input": False, "input_inplace_pass": True}
        
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, in1, in2, in3, in4, is_pr):
                event1 = torch.npu.Event()
                event2 = torch.npu.Event()
                stream1 = torch.npu.Stream()
                stream2 = torch.npu.Stream()

                add_result = torch.add(in1, in2)

                event1.record()
                event2.record()
                mm_result2 = add_result
                with torch.npu.stream(stream1):
                    event1.wait()
                    mm_result1 = torch.mm(in3, in4)
                    if is_pr:
                        with torch.npu.stream(stream2):
                            event2.wait()
                            mm_result2 = torch.mm(in3, in4)
                return add_result, mm_result1, mm_result2

        model = Model()
        opt_model = torch.compile(model, backend="npugraph_ex", options=options, fullgraph=True, dynamic=False)
        x = torch.randn([3, 3])
        y = torch.randn([3, 3])
        z = torch.randn([3, 3])
        w = torch.randn([3, 3])
        from npugraph_ex._acl_concrete_graph.graph_pass import _GLOBAL_SCOPE_TAG_TO_EVENT
        from npugraph_ex.scope._scope_attr import _GLOBAL_TAG_TO_STREAM
        opt_model(x, y, z, w, True)
        len_of_tagged_event_1 = len(_GLOBAL_SCOPE_TAG_TO_EVENT)
        len_of_stream_1 = len(_GLOBAL_TAG_TO_STREAM)
        opt_model(x, y, z, w, False)
        len_of_tagged_event_2 = len(_GLOBAL_SCOPE_TAG_TO_EVENT)
        len_of_stream_2 = len(_GLOBAL_TAG_TO_STREAM)
        opt_model(x, y, z, w, True)
        len_of_tagged_event_3 = len(_GLOBAL_SCOPE_TAG_TO_EVENT)
        len_of_stream_3 = len(_GLOBAL_TAG_TO_STREAM)
        assert len_of_tagged_event_2 == len_of_tagged_event_3
        assert len_of_stream_2 == len_of_stream_3

    def test_force_eager_with_multi_stream_and_static_kernel(self):

        class StubTensor:
            def record_stream(self, stream):
                return 

        origin = torch.Tensor.record_stream
        torch.Tensor.record_stream = StubTensor.record_stream

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, in1, in2, in3, in4):
                stream1 = torch.npu.Stream()
                stream2 = torch.npu.Stream()
                event1 = torch.npu.Event()
                event2 = torch.npu.Event()

                add_result = torch.add(in1, in2)
                event1.record()
                with torch.npu.stream(stream1):
                    event1.wait()
                    mm_result = torch.mm(in3, in4)
                    event2.record()
                    B = in3 + in4
                    B.record_stream(stream1)
                mm1 = torch.mm(in3, in4)
                del B
                C = torch.ones(1000, 1000, dtype=torch.float16)
                C.add_(2)
                with torch.npu.stream(stream2):
                    event2.wait()
                    add2 = torch.add(in3, in4)
                return add_result, mm_result, mm1, add2, C
        
        model = Model()
        options = {
            "static_kernel_compile": True,
            "force_eager": True
        }
        model = torch.compile(model, backend="npugraph_ex", dynamic=False, fullgraph=True, options=options)
        in1 = torch.randn(1000, 1000, dtype=torch.float16)
        in2 = torch.randn(1000, 1000, dtype=torch.float16)
        in3 = torch.randn(1000, 1000, dtype=torch.float16)
        in4 = torch.randn(1000, 1000, dtype=torch.float16)
        with capture_logger() as stdout:
            result = model(in1, in2, in3, in4)

        expected_pattern = re.compile(
            r'def forward\(\*args,\s*node_info=\[\],\s*is_capturing:\s*bool\s*=\s*False\):'
            r'.*?'
            r'arg0_1,\s*arg1_1,\s*arg2_1,\s*arg3_1\s*=\s*args'
            r'.*?'
            r'global\s+_GLOBAL_USER_TAG_TO_STREAM'
            r'.*?'
            r'with torch\.npu\.stream\(_GLOBAL_USER_TAG_TO_STREAM\[\'graph_0_stream\'\]\):'
            r'.*?'
            r'ones = torch\.ops\.aten\.ones.*?\(.*?device\(type=\'cpu\'\).*?\)'
            r'.*?'
            r'with torch\.npu\.stream\(_GLOBAL_USER_TAG_TO_STREAM\[\'graph_0_stream_1\'\]\):'
            r'.*?'
            r'return\s+\(add,\s+mm,\s+mm_1,\s+add_3,\s+add_2\)',
            re.DOTALL
        )
        self.assertIsNotNone(expected_pattern.search(stdout.getvalue()))

    def test_eager_with_multi_stream_event(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.stream1 = torch.npu.Stream()
                self.stream2 = torch.npu.Stream()
                self.ext_event = torch.npu.Event()

            def forward(self, x):
                def branch1(xx):
                    y = xx + 1
                    y = y * y
                    y = y @ y
                    self.ext_event.record()
                    return y

                def branch2(xx):
                    self.ext_event.wait()
                    y = xx - 1
                    y = y @ y
                    return y

                with torch.npu.stream(self.stream1):
                    out1 = branch1(x)
                with torch.npu.stream(self.stream2):
                    out2 = branch2(x)
                return out1, out2

        model = Model()
        x = torch.randn([3, 3])
        for _ in range(2):
            model(x)

if __name__ == '__main__':
    unittest.main()
