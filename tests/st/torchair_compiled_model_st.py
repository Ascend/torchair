from typing import List
import unittest
import time
import os
import shutil
import logging
import torch
import torch.distributed._functional_collectives as funcol
import torchair
import torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce
from torchair.core.utils import logger

os.environ['TNG_LOG_LEVEL'] = '0'
logger.setLevel(logging.DEBUG)


def get_dumped_file_list(dir_path, file_extension='.pbtxt'):
    return [i for i in os.listdir(dir_path) if i.startswith('dynamo') and i.endswith(f'{file_extension}')]


class TorchairSt(unittest.TestCase):
    def setUp(self) -> None:
        self.clean_env()
        return super().setUp()

    def tearDown(self) -> None:
        self.clean_env()
        return super().tearDown()

    def clean_env(self):
        for export_path in ["export_file", "false_export_path2", "true_export_path2", \
                            "true_export_path3", "test_export_file_path"]:
            if os.path.exists(export_path):
                shutil.rmtree(export_path)

    def test_basic_save_and_load(self):
        class Model(torch.nn.Module):

            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x + y

        model = Model()
        x = (torch.ones(2, 4), torch.ones(2, 4),)

        export_path = "test_export_file_path/"
        if os.path.exists(export_path):
            shutil.rmtree(export_path)

        torchair.compiled_model.save_graph(model, args=x, save_path=export_path)

        compiled_model = torchair.compiled_model.load_graph(export_path)
        assert compiled_model.graph.name == "graph_1"
        ret = compiled_model(x[0], x[1])

    def test_load_file_not_exist(self):
        load_path = "false_export_path2"
        try:
            torchair.compiled_model.load_graph(load_path)
        except FileNotFoundError as e:
            assert type(e).__name__ == "FileNotFoundError"

    def test_saved_graph_with_compile_option(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x.add_(y)

        model = Model()
        x = (torch.ones(2, 4), torch.ones(2, 4),)

        export_path = "test_export_file_path/"
        if os.path.exists(export_path):
            shutil.rmtree(export_path)

        from torchair.configs.compiler_config import CompilerConfig
        compile_config = CompilerConfig()
        compile_config.experimental_config.static_model_ops_lower_limit = 5

        torchair.compiled_model.save_graph(model, args=x, config=compile_config, save_path=export_path)

        compiled_model = torchair.compiled_model.load_graph(export_path)

        assert compiled_model._fx_inputs_mapping == {0: 0, 1: 1}
        assert compiled_model._fx_outputs_mapping == {0: 0}
        assert compiled_model._local_compile_options == {'export_path_dir': 'test_export_file_path/',
                                                         'export_name': 'compiled_graph',
                                                         'ge.featureBaseRefreshable': '0', 'ge.topoSortingMode': '1',
                                                         'ge.jit_compile': '2',
                                                         'ge.exec.static_model_ops_lower_limit': '5',
                                                         'ge.deterministic': '0', 'ge.exec.atomicCleanPolicy': '1'}
        assert compiled_model._global_compile_options == {'ge.exec.enableEngineParallel': '0',
                                                          'ge.enableSingleStream': 'false',
                                                          'ge.exec.staticMemoryPolicy': '2'}
        assert compiled_model._graph_output_ref_input == {0: 0}

    def test_saved_graph_uncontiguous_input_with_ref_unsupported(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return y.add_(x)
        x0 = torch.randn(2, 4)
        x1 = torch.randn(4, 2).t()
        save_path = "test_export_file_path/"
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        try:
            torchair.compiled_model.save_graph(Model(), (x0, x1), save_path=save_path)
        except NotImplementedError as e:
            assert type(e).__name__ == "NotImplementedError"

    def test_saved_graph_input_with_1list_supported(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x: torch.Tensor, y: List):
                x = x + x
                return torch.as_strided(x, y, [1])

        save_path = "test_export_file_path/"
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        x0 = torch.randn(10, 5)
        x1 = [50]
        torchair.compiled_model.save_graph(Model(), (x0, x1), save_path=save_path)

    def test_saved_graph_input_with_2list_unsupport(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x: torch.Tensor, size1: List, y:torch.Tensor, size2: List):
                x = x + x
                y = y + y
                return torch.as_strided(x, size1, [1]), torch.as_strided(y, size2, [1])

        save_path = "test_export_file_path/"
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        x1 = torch.randn(10, 2, 3, 40)
        size1 = [2400]
        x2 = torch.randn(2, 3, 40)
        size2 = [240]
        try:
            torchair.compiled_model.save_graph(Model(), (x1, size1, x2, size2), dynamic=True, save_path=save_path)
        except NotImplementedError as e:
            assert type(e).__name__ == "NotImplementedError"

    def test_saved_graph_unsupport3(self):
        def call_sub(self, *args, **kwargs):
            self._check_support_for_save_graph()
            return args

        from torchair.ge_concrete_graph.fx2ge_converter import GeConcreteGraph
        src_call = GeConcreteGraph.__call__
        GeConcreteGraph.__call__ = call_sub
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dp = torch.nn.Dropout(0.3)

            def forward(self, x):
                y = self.dp(x)
                b1 = torch.ops.aten.bernoulli.p(x, 0.8)
                y = y * b1
                return y

        x = torch.randn(10, 10)
        save_path = "test_export_file_path/"
        if os.path.exists(save_path):
            shutil.rmtree(save_path)

        try:
            torchair.compiled_model.save_graph(Model(), (x,), dynamic=False, save_path=save_path)
        except NotImplementedError as e:
            assert type(e).__name__ == "NotImplementedError"

        GeConcreteGraph.__call__ = src_call

    def test_saved_graph_output_is_view_of_input_unsupported(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x: torch.Tensor, size: List):
                return x.view(size)
        save_path = "test_export_file_path/"
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        x0 = torch.randn(10, 5)
        x1 = [50]
        try:
            torchair.compiled_model.save_graph(Model(), (x0, x1), save_path=save_path)
        except NotImplementedError as e:
            assert type(e).__name__ == "NotImplementedError"