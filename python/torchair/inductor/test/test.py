import contextlib
import os
import logging
import sys

base_dir = os.path.join(os.path.dirname(__file__), '_debug_kernel')


def enable_torch_logger(key='all', *, exclude=None):
    assert 'torch' not in sys.modules

    _getLogger = logging.getLogger

    def wrapper(name=None):
        logger = _getLogger(name)
        if hasattr(logger, 'determined'):
            return logger

        logger.determined = True
        if exclude and name and exclude in name:
            logger.setLevel(logging.ERROR)
            return logger

        if key == 'all' or name is None or (key and key in name):
            logger.setLevel(logging.DEBUG)
            if not os.path.exists(base_dir):
                os.mkdir(base_dir)
            file_handler = logging.FileHandler(os.path.join(base_dir, 'dynamo.log'))
            formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    logging.getLogger = wrapper


def debug_npu_inductor():
    os.environ['TORCH_SHOW_DISPATCH_TRACE'] = '1'
    os.environ['TORCH_COMPILE_DEBUG'] = '1'
    os.environ['INDUCTOR_POST_FUSION_SVG'] = '1'
    os.environ['INDUCTOR_ORIG_FX_SVG'] = '1'
    os.environ['INDUCTOR_WRITE_SCHEDULER_GRAPH'] = '1'
    enable_torch_logger(exclude='fake_tensor')

    from torch._inductor import config

    config.trace.debug_dir = base_dir


os.environ["NPU_CORE_TYPE"] = "ai_core-ascend910B1"  # 要和stub实现、执行环境匹配
os.environ["ASCIR_NOT_READY"] = "1"  # 禁用ascir和pyautofuser，使用stub实现

import npu_extension_for_inductor
import torch
import unittest


@contextlib.contextmanager
def disable_npu_fallback(disable=True):
    old = os.getenv("DISABLE_NPU_FALLBACK", "0")
    try:
        os.environ["DISABLE_NPU_FALLBACK"] = "1" if disable else "0"
        yield
    finally:
        os.environ["DISABLE_NPU_FALLBACK"] = old


class InductorNpuBackendTest(unittest.TestCase):
    def test_abs(self):
        """
        基础测试，测试abs算子
        """

        @torch.compile(dynamic=True)
        def test_abs(x):
            return torch.abs(x)

        x = torch.full((16, 512), -1.0, dtype=torch.float16)
        y = test_abs(x)

        self.assertTrue(torch.allclose(y, torch.abs(x)))

    def test_abs_sqrt(self):
        """
        测试不支持的算子自动fallback到单算子，当前支持abs，不支持sqrt
        """

        @torch.compile(dynamic=True)
        def test_abs_sqrt(x):
            x = torch.abs(x)
            x = torch.sqrt(x)
            return x

        x = torch.full((32, 8, 16), -1.0, dtype=torch.float16)
        y = test_abs_sqrt(x)

        self.assertTrue(torch.allclose(y, torch.sqrt(torch.abs(x))))

    def test_softmax(self):
        """
        测试softmax复杂算子，包含reduce/cast等特殊操作
        """

        @torch.compile(dynamic=True)
        def test_softmax(x):
            x = torch.softmax(x, dim=3)
            return x

        x = torch.ones(1, 96, 2048, 128, dtype=torch.float16)
        y = test_softmax(x)

        self.assertTrue(torch.allclose(y, torch.softmax(x, dim=3)))

    def test_constant(self):
        """
        测试带有constant算子的图
        """

        @torch.compile(dynamic=False)
        def test_constant(x):
            x = torch.add(x, 2.0)
            return x

        x = torch.ones(2, dtype=torch.float16)
        y = test_constant(x)

        self.assertTrue(torch.allclose(y, torch.add(x, 3.0)))

    def test_embeding(self):
        """
        测试带有embeding算子的图
        """

        @torch.compile(dynamic=True)
        def test_embeding(x, w):
            x = torch.nn.functional.embedding(x, w)
            return x

        x = torch.ones(2, dtype=torch.int64)
        w = torch.arange(0, 200, dtype=torch.float16).view(10, 20)
        y = test_embeding(x, w)

        self.assertTrue(torch.allclose(y, torch.nn.functional.embedding(x, w)))


if __name__ == '__main__':
    unittest.main()