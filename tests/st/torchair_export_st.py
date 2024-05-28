import unittest
import time
import os
import shutil
import logging
import torch
import torch.distributed._functional_collectives as funcol
import torchair
import torchair.ge_concrete_graph.ge_converter.experimental.hcom_allreduce
from torchair.ge_concrete_graph.ge_ir_pb2 import ModelDef
from torchair.core.utils import logger

os.environ['TNG_LOG_LEVEL'] = '0'
logger.setLevel(logging.DEBUG)


def get_dumped_file_list(dir_path, file_extension='.pbtxt'):
    return [i for i in os.listdir(dir_path) if i.startswith('dynamo') and i.endswith(f'{file_extension}')]


def check_tensor_desc_list_same(tensor_desc_list1, tensor_desc_list2):
    assert not (len(tensor_desc_list1) != len(tensor_desc_list2))
    # 这个list只看第一个就够了
    if len(tensor_desc_list1) == 0:
        return True
    assert not (tensor_desc_list1[0].dtype != tensor_desc_list2[0].dtype)
    assert not (tensor_desc_list1[0].shape != tensor_desc_list2[0].shape)
    assert not (tensor_desc_list1[0].layout != tensor_desc_list2[0].layout)
    assert not (set(tensor_desc_list1[0].attr) != set(tensor_desc_list2[0].attr))
    assert not (tensor_desc_list1[0].device_type != tensor_desc_list2[0].device_type)
    return True


def check_op_list_same(op_list_1, op_list_2):
    assert not (len(op_list_1) != len(op_list_2))
    op_list_1.sort(key=lambda item: item.name)
    op_list_2.sort(key=lambda item: item.name)
    for i in range(len(op_list_1)):
        # 不能校验 op_list_1[i].name，name与进程中add算子数量相关
        assert not (op_list_1[i].type != op_list_2[i].type)
        # input中根据name索引，不能强校验input
        assert not (set(op_list_1[i].attr) != set(op_list_2[i].attr))

        assert check_tensor_desc_list_same(op_list_1[i].input_desc, op_list_2[i].input_desc)
        assert check_tensor_desc_list_same(op_list_1[i].output_desc, op_list_2[i].output_desc)
    return True


def protobuf_equal(msg1, msg2):
    assert not (type(msg1) != type(msg2))
    # 检查消息的所有字段是否相等
    assert not (msg1.ListFields()[1][1] != msg2.ListFields()[1][1])
    # 不能校验 msg1.ListFields()[0][1][0].name，graph name与进程中图数量相关
    assert not (set(msg1.ListFields()[0][1][0].input) != set(msg2.ListFields()[0][1][0].input))
    assert check_op_list_same(msg1.ListFields()[0][1][0].op, msg2.ListFields()[0][1][0].op)
    return True


def compare_with_base_line(file_path, str):
    model = ModelDef()
    baseline_model = ModelDef()
    baseline_buffer = str
    baseline_model.ParseFromString(baseline_buffer)
    with open(file_path, 'rb') as f:
        buffer = f.read()
        model.ParseFromString(buffer)
    return protobuf_equal(baseline_model, model)


class TorchairSt(unittest.TestCase):
    def setUp(self) -> None:
        self.clean_env()
        return super().setUp()

    def tearDown(self) -> None:
        self.clean_env()
        return super().tearDown()

    def clean_env(self):
        for export_path in ["export_file", "false_export_path2", "true_export_path2", \
                            "true_export_path3", "test_export_file_path", "test_export_file_path2"]:
            if os.path.exists(export_path):
                shutil.rmtree(export_path)

    def test_export(self):
        class Model(torch.nn.Module):

            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x = x + y
                return x

        model = Model()
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)

        export_path1 = "test_export_file_path"
        if os.path.exists(export_path1):
            shutil.rmtree(export_path1)
        torchair.dynamo_export(x, y, model=model, export_path=export_path1, dynamic=False)

        dumped_file_list = get_dumped_file_list(export_path1)
        dumped_file_list.sort(key=lambda file_name: os.path.getmtime(os.path.join(export_path1, file_name)))
        assert dumped_file_list.__len__() > 0
        file_name = os.path.join(export_path1, dumped_file_list[-1])

        with open(file_name, 'r') as f:
            src = f.read()

        assert src.count("op: \"Data\"") == 2
        assert src.count("op: \"Shape\"") == 0

        baseline_air = (b':\x99\x10\n\x07graph_1"\x08arg0_1:0"\x08arg1_1:02\xc9\x03\n\x06arg0_1\x12\x04Data*\x00R'
        b'\x19\n\x10_output_name_key\x12\x05\n\x03\x12\x01yR\x18\n\x0f_input_name_key\x12\x05\n\x03\x12\x01xR'
        b'\x1b\n\x12_output_name_value\x12\x05\n\x03\x1a\x01\x00R\x1a\n\x11_input_name_value\x12\x05\n\x03\x1a'
        b'\x01\x00R\x0b\n\x05index\x12\x02\x18\x00\x8a\x02{\x10\x01\x1a\x04\n\x02\x02\x04"\x02ND*\x15\n\r'
        b'origin_format\x12\x04\x12\x02ND*\x1e\n\x18origin_shape_initialized\x12\x02(\x00*\x14\n\x0eformat_for_int'
        b'\x12\x02\x18\x02*\x1b\n\x15origin_format_for_int\x12\x02\x18\x02r\x03CPU\x92\x02\xbc\x01\x10\x01\x1a\x04'
        b'\n\x02\x02\x04"\x02ND*\x1e\n\x18origin_shape_initialized\x12\x02(\x00*\x14\n\x0eformat_for_int\x12\x02'
        b'\x18\x02*\x1b\n\x15origin_format_for_int\x12\x02\x18\x02*\x15\n\rorigin_format\x12\x04\x12\x02ND*?\n\x05'
        b'_meta\x126\x124Tensor(dtype=torch.float32, shape=torch.Size([2, 4])r\x03CPU2\xc9\x03\n\x06arg1_1\x12\x04'
        b'Data*\x00R\x0b\n\x05index\x12\x02\x18\x01R\x18\n\x0f_input_name_key\x12\x05\n\x03\x12\x01xR\x1b\n\x12'
        b'_output_name_value\x12\x05\n\x03\x1a\x01\x00R\x1a\n\x11_input_name_value\x12\x05\n\x03\x1a\x01\x00R\x19\n'
        b'\x10_output_name_key\x12\x05\n\x03\x12\x01y\x8a\x02{\x10\x01\x1a\x04\n\x02\x02\x04"\x02ND*\x1b\n\x15'
        b'origin_format_for_int\x12\x02\x18\x02*\x15\n\rorigin_format\x12\x04\x12\x02ND*\x1e\n\x18'
        b'origin_shape_initialized\x12\x02(\x00*\x14\n\x0eformat_for_int\x12\x02\x18\x02r\x03CPU\x92\x02\xbc\x01\x10'
        b'\x01\x1a\x04\n\x02\x02\x04"\x02ND*\x14\n\x0eformat_for_int\x12\x02\x18\x02*\x1e\n\x18'
        b'origin_shape_initialized\x12\x02(\x00*\x15\n\rorigin_format\x12\x04\x12\x02ND*\x1b\n\x15'
        b'origin_format_for_int\x12\x02\x18\x02*?\n\x05_meta\x126\x124Tensor(dtype=torch.float32, '
        b'shape=torch.Size([2, 4])r\x03CPU2\x81\x06\n\x03Add\x12\x03Add*\x08arg0_1:0*\x08arg1_1:0R\x1b\n\x12'
        b'_output_name_value\x12\x05\n\x03\x1a\x01\x00R\x19\n\x10_output_name_key\x12\x05\n\x03\x12\x01yR\x1b\n\x11'
        b'_input_name_value\x12\x06\n\x04\x1a\x02\x00\x01R\x1d\n\x0f_input_name_key\x12\n\n\x08\x12\x02x1\x12\x02x2'
        b'\x8a\x02\xbc\x01\x10\x01\x1a\x04\n\x02\x02\x04"\x02ND*\x14\n\x0eformat_for_int\x12\x02\x18\x02*\x15\n\r'
        b'origin_format\x12\x04\x12\x02ND*\x1b\n\x15origin_format_for_int\x12\x02\x18\x02*?\n\x05_meta\x126\x124'
        b'Tensor(dtype=torch.float32, shape=torch.Size([2, 4])*\x1e\n\x18origin_shape_initialized\x12\x02(\x00r\x03'
        b'CPU\x8a\x02\xbc\x01\x10\x01\x1a\x04\n\x02\x02\x04"\x02ND*\x1b\n\x15origin_format_for_int\x12\x02\x18\x02*'
        b'\x14\n\x0eformat_for_int\x12\x02\x18\x02*\x15\n\rorigin_format\x12\x04\x12\x02ND*?\n\x05_meta\x126\x124'
        b'Tensor(dtype=torch.float32, shape=torch.Size([2, 4])*\x1e\n\x18origin_shape_initialized\x12\x02(\x00r\x03'
        b'CPU\x92\x02\xeb\x01\x10\x01\x1a\x00"\x02ND*\x14\n\x0eformat_for_int\x12\x02\x18\x02*1\n\x0f_fx_tensor_name'
        b'\x12\x1e\x12\x1cadd-aten.add.Tensor.OUTPUT.0*\x1b\n\x15origin_format_for_int\x12\x02\x18\x02*\x1e\n\x18'
        b'origin_shape_initialized\x12\x02(\x00*?\n\x05_meta\x126\x124Tensor(dtype=torch.float32, shape=torch.Size'
        b'([2, 4])*\x15\n\rorigin_format\x12\x04\x12\x02NDr\x03NPU2\xc7\x02\n\tNetOutput\x12\tNetOutput*\x05Add:0R'
        b'\x1a\n\x11_input_name_value\x12\x05\n\x03\x1a\x01\x00R\x1d\n\x0f_input_name_key\x12\n\n\x08\x12\x06input0'
        b'\x8a\x02\xeb\x01\x10\x01\x1a\x00"\x02ND*\x15\n\rorigin_format\x12\x04\x12\x02ND*\x1e\n\x18'
        b'origin_shape_initialized\x12\x02(\x00*?\n\x05_meta\x126\x124Tensor(dtype=torch.float32, shape=torch.Size'
        b'([2, 4])*\x1b\n\x15origin_format_for_int\x12\x02\x18\x02*\x14\n\x0eformat_for_int\x12\x02\x18\x02*1\n\x0f'
        b'_fx_tensor_name\x12\x1e\x12\x1cadd-aten.add.Tensor.OUTPUT.0r\x03NPUZ\x14\n\x0e_executor_type\x12\x02\x18'
        b'\x00Z\x15\n\x0btarget_type\x12\x06\x12\x04MINIZ\x10\n\nstream_num\x12\x02\x18\x00Z\x0f\n\tevent_num\x12'
        b'\x02\x18\x00Z\x11\n\x0bweight_size\x12\x02\x18\x00Z\x15\n\x0fp2p_memory_size\x12\x02\x18\x00Z\x11\n\x0b'
        b'memory_size\x12\x02\x18\x00Z\x0f\n\tlabel_num\x12\x02\x18\x00')
        assert compare_with_base_line('./test_export_file_path/export.air', baseline_air)

    def test_export_with_sym(self):
        def get_inputnum_in_node(strgraph, opname):
            start_str = opname
            end_str = "attr {"
            start_index = strgraph.find(start_str)
            sub_str = strgraph[start_index: len(strgraph) - 1]
            end_index = sub_str.find(end_str)
            result = sub_str[0: end_index]
            return result.count("input: ")

        class Model(torch.nn.Module):

            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x = x + y
                z = torch.cat((x, y), 0)
                return z.size()[1], x

        model = Model()
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)

        export_path1 = "test_export_file_path"
        if os.path.exists(export_path1):
            shutil.rmtree(export_path1)

        torchair.dynamo_export(x, y, model=model, export_path=export_path1, dynamic=True)

        dumped_file_list = get_dumped_file_list(export_path1)
        dumped_file_list.sort(key=lambda file_name: os.path.getmtime(os.path.join(export_path1, file_name)))
        assert dumped_file_list.__len__() > 0
        file_name = os.path.join(export_path1, dumped_file_list[-1])

        with open(file_name, 'r') as f:
            src = f.read()

        assert src.count("op: \"Shape\"") == 1
        assert src.count("op: \"Data\"") == 2
        assert get_inputnum_in_node(src, "op: \"NetOutput\"")

    def test_export_with_allreduce(self):
        def get_sub_path_dynamo_pbtxt(export_path, rankid):
            return export_path + "/rank_" + str(rankid) + "/dynamo.pbtxt"

        def get_model_relation_config(export_path):
            return export_path + "/model_relation_config.json"

        def get_numa_config(export_path):
            return export_path + "/numa_config.json"

        def mp():
            world_size = 2
            torch.multiprocessing.spawn(example, args=(world_size,), nprocs=world_size, join=True)

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29505"

        mp()

        file_name = get_sub_path_dynamo_pbtxt("export_file", 0)
        with open(file_name, 'r') as f:
            src = f.read()
        assert src.count("op: \"Const\"") == 2
        assert src.count("op: \"Data\"") == 2
        assert src.count("op: \"HcomAllReduce\"") == 1
        assert src.count("key: \"ranklist\"") == 1

        file_name = get_sub_path_dynamo_pbtxt("false_export_path2", 0)
        with open(file_name, 'r') as f:
            src = f.read()
        assert src.count("op: \"HcomAllReduce\"") == 4  # 多group场景

        file_name = get_sub_path_dynamo_pbtxt("true_export_path2", 0)
        with open(file_name, 'r') as f:
            src = f.read()
        assert src.count(" dim: -1") == 3  # 动态图存在-1

        file_name = get_model_relation_config("true_export_path2")
        # mutil group case, can not create atc config file
        assert os.path.exists(file_name) == False
        file_name = get_numa_config("true_export_path2")
        assert os.path.exists(file_name) == False

        file_name = get_sub_path_dynamo_pbtxt("true_export_path3", 0)
        with open(file_name, 'r') as f:
            src = f.read()
        assert src.count("HcomReduceScatter") == 3  # dist reduce_scatter_tensor入图

    def test_export_weight_externalized(self):
        class Model(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.p1 = torch.nn.Parameter(torch.randn([1024, 1024, 1024], dtype=torch.float16))  # 2G weight

            def forward(self, x, y):
                x = x + y
                w = self.p1 * 2
                return x, w

        model = Model()
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)

        export_path1 = "test_export_file_path"

        torchair.dynamo_export(x, y, model=model, export_path=export_path1, dynamic=False)

        dumped_file_list = get_dumped_file_list(export_path1)
        dumped_file_list.sort(key=lambda file_name: os.path.getmtime(os.path.join(export_path1, file_name)))
        assert dumped_file_list.__len__() > 0
        file_name = os.path.join(export_path1, dumped_file_list[-1])

        with open(file_name, 'r') as f:
            src = f.read()

        assert src.count("op: \"FileConstant\"") == 1
        assert src.count("op: \"Data\"") == 2
        assert src.count("op: \"Shape\"") == 0
        baseline_fileconst = (
            b':\xa8#\n\x07graph_1"\x08arg1_1:0"\x08arg2_1:02\xa5\x02\n\x06arg0_1\x12\x0cFileConstantR\x0b\n\x05dtype'
            b'\x12\x02x\x01R\x1b\n\x12_output_name_value\x12\x05\n\x03\x1a\x01\x00R\x16\n\x05shape\x12\r\n\x0b\x1a\x06'
            b'\x80\x08\x80\x08\x80\x08\xa0\x01\x02R\r\n\x07file_id\x12\x02\x12\x00R\x19\n\x10_output_name_key\x12\x05\n'
            b'\x03\x12\x01yR\'\n\tfile_path\x12\x1a\x12\x18test_export_file_path/p1\x92\x02w\x10\x01\x1a\x00"\x02ND*'
            b'\x1b\n\x15origin_format_for_int\x12\x02\x18\x02*\x15\n\rorigin_format\x12\x04\x12\x02ND*\x14\n\x0e'
            b'format_for_int\x12\x02\x18\x02*\x1e\n\x18origin_shape_initialized\x12\x02(\x00r\x03NPU2\xc9\x03\n\x06'
            b'arg1_1\x12\x04Data*\x00R\x1a\n\x11_input_name_value\x12\x05\n\x03\x1a\x01\x00R\x0b\n\x05index\x12\x02'
            b'\x18\x00R\x19\n\x10_output_name_key\x12\x05\n\x03\x12\x01yR\x18\n\x0f_input_name_key\x12\x05\n\x03\x12'
            b'\x01xR\x1b\n\x12_output_name_value\x12\x05\n\x03\x1a\x01\x00\x8a\x02{\x10\x01\x1a\x04\n\x02\x02\x04"'
            b'\x02ND*\x15\n\rorigin_format\x12\x04\x12\x02ND*\x1e\n\x18origin_shape_initialized\x12\x02(\x00*\x1b\n'
            b'\x15origin_format_for_int\x12\x02\x18\x02*\x14\n\x0eformat_for_int\x12\x02\x18\x02r\x03CPU\x92\x02\xbc'
            b'\x01\x10\x01\x1a\x04\n\x02\x02\x04"\x02ND*\x1b\n\x15origin_format_for_int\x12\x02\x18\x02*\x15\n\r'
            b'origin_format\x12\x04\x12\x02ND*\x1e\n\x18origin_shape_initialized\x12\x02(\x00*\x14\n\x0eformat_for_int'
            b'\x12\x02\x18\x02*?\n\x05_meta\x126\x124Tensor(dtype=torch.float32, shape=torch.Size([2, 4])r\x03CPU2\xc9'
            b'\x03\n\x06arg2_1\x12\x04Data*\x00R\x18\n\x0f_input_name_key\x12\x05\n\x03\x12\x01xR\x0b\n\x05index\x12'
            b'\x02\x18\x01R\x1a\n\x11_input_name_value\x12\x05\n\x03\x1a\x01\x00R\x19\n\x10_output_name_key\x12\x05\n'
            b'\x03\x12\x01yR\x1b\n\x12_output_name_value\x12\x05\n\x03\x1a\x01\x00\x8a\x02{\x10\x01\x1a\x04\n\x02\x02'
            b'\x04"\x02ND*\x14\n\x0eformat_for_int\x12\x02\x18\x02*\x1b\n\x15origin_format_for_int\x12\x02\x18\x02*'
            b'\x15\n\rorigin_format\x12\x04\x12\x02ND*\x1e\n\x18origin_shape_initialized\x12\x02(\x00r\x03CPU\x92\x02'
            b'\xbc\x01\x10\x01\x1a\x04\n\x02\x02\x04"\x02ND*\x1b\n\x15origin_format_for_int\x12\x02\x18\x02*\x1e\n'
            b'\x18origin_shape_initialized\x12\x02(\x00*?\n\x05_meta\x126\x124Tensor(dtype=torch.float32, '
            b'shape=torch.Size([2, 4])*\x14\n\x0eformat_for_int\x12\x02\x18\x02*\x15\n\rorigin_format\x12\x04\x12'
            b'\x02NDr\x03CPU2\x81\x06\n\x03Add\x12\x03Add*\x08arg1_1:0*\x08arg2_1:0R\x1b\n\x12_output_name_value'
            b'\x12\x05\n\x03\x1a\x01\x00R\x1b\n\x11_input_name_value\x12\x06\n\x04\x1a\x02\x00\x01R\x19\n\x10'
            b'_output_name_key\x12\x05\n\x03\x12\x01yR\x1d\n\x0f_input_name_key\x12\n\n\x08\x12\x02x1\x12\x02x2'
            b'\x8a\x02\xbc\x01\x10\x01\x1a\x04\n\x02\x02\x04"\x02ND*\x1b\n\x15origin_format_for_int\x12\x02\x18\x02*'
            b'\x1e\n\x18origin_shape_initialized\x12\x02(\x00*\x15\n\rorigin_format\x12\x04\x12\x02ND*?\n\x05_meta'
            b'\x126\x124Tensor(dtype=torch.float32, shape=torch.Size([2, 4])*\x14\n\x0eformat_for_int\x12\x02\x18\x02r'
            b'\x03CPU\x8a\x02\xbc\x01\x10\x01\x1a\x04\n\x02\x02\x04"\x02ND*\x1b\n\x15origin_format_for_int\x12\x02\x18'
            b'\x02*\x14\n\x0eformat_for_int\x12\x02\x18\x02*\x1e\n\x18origin_shape_initialized\x12\x02(\x00*\x15\n\r'
            b'origin_format\x12\x04\x12\x02ND*?\n\x05_meta\x126\x124Tensor(dtype=torch.float32, shape=torch.'
            b'Size([2, 4])r\x03CPU\x92\x02\xeb\x01\x10\x01\x1a\x00"\x02ND*1\n\x0f_fx_tensor_name\x12\x1e\x12\x1c'
            b'add-aten.add.Tensor.OUTPUT.0*\x14\n\x0eformat_for_int\x12\x02\x18\x02*\x1b\n\x15origin_format_for_int'
            b'\x12\x02\x18\x02*\x15\n\rorigin_format\x12\x04\x12\x02ND*\x1e\n\x18origin_shape_initialized\x12\x02'
            b'(\x00*?\n\x05_meta\x126\x124Tensor(dtype=torch.float32, shape=torch.Size([2, 4])r\x03NPU2\xea\x02\n'
            b'\x05Const\x12\x05ConstR\x90\x01\n\x05value\x12\x86\x01b\x83\x01\nw\x10\x08\x1a\x00"\x02ND*\x1b\n\x15'
            b'origin_format_for_int\x12\x02\x18\x02*\x14\n\x0eformat_for_int\x12\x02\x18\x02*\x1e\n\x18'
            b'origin_shape_initialized\x12\x02(\x00*\x15\n\rorigin_format\x12\x04\x12\x02NDr\x03NPU\x12\x08\x02\x00'
            b'\x00\x00\x00\x00\x00\x00R\x16\n\x0f_readable_value\x12\x03\x12\x012R\x1b\n\x12_output_name_value\x12'
            b'\x05\n\x03\x1a\x01\x00R\x18\n\x10_output_name_key\x12\x04\n\x02\x12\x00\x92\x02w\x10\x08\x1a\x00"\x02'
            b'ND*\x15\n\rorigin_format\x12\x04\x12\x02ND*\x14\n\x0eformat_for_int\x12\x02\x18\x02*\x1b\n\x15'
            b'origin_format_for_int\x12\x02\x18\x02*\x1e\n\x18origin_shape_initialized\x12\x02(\x00r\x03NPU2\x87\x03'
            b'\n\x04Cast\x12\x04Cast*\x07Const:0R\x18\n\x0f_input_name_key\x12\x05\n\x03\x12\x01xR\x0e\n\x08dst_type'
            b'\x12\x02\x18\x01R\x1b\n\x12_output_name_value\x12\x05\n\x03\x1a\x01\x00R\x19\n\x10_output_name_key\x12'
            b'\x05\n\x03\x12\x01yR\x1a\n\x11_input_name_value\x12\x05\n\x03\x1a\x01\x00\x8a\x02w\x10\x08\x1a\x00"\x02'
            b'ND*\x1e\n\x18origin_shape_initialized\x12\x02(\x00*\x15\n\rorigin_format\x12\x04\x12\x02ND*\x14\n\x0e'
            b'format_for_int\x12\x02\x18\x02*\x1b\n\x15origin_format_for_int\x12\x02\x18\x02r\x03NPU\x92\x02w\x10\x01'
            b'\x1a\x00"\x02ND*\x1b\n\x15origin_format_for_int\x12\x02\x18\x02*\x1e\n\x18origin_shape_initialized\x12'
            b'\x02(\x00*\x15\n\rorigin_format\x12\x04\x12\x02ND*\x14\n\x0eformat_for_int\x12\x02\x18\x02r\x03NPU2\xd4'
            b'\x04\n\x03Mul\x12\x03Mul*\x08arg0_1:0*\x06Cast:0R\x19\n\x10_output_name_key\x12\x05\n\x03\x12\x01yR\x1d'
            b'\n\x0f_input_name_key\x12\n\n\x08\x12\x02x1\x12\x02x2R\x1b\n\x11_input_name_value\x12\x06\n\x04\x1a\x02'
            b'\x00\x01R\x1b\n\x12_output_name_value\x12\x05\n\x03\x1a\x01\x00\x8a\x02\xcc\x01\x10\x02\x1a\x08\n\x06'
            b'\x80\x08\x80\x08\x80\x08"\x02ND*\x14\n\x0eformat_for_int\x12\x02\x18\x02*K\n\x05_meta\x12B\x12@Tensor'
            b'(dtype=torch.float16, shape=torch.Size([1024, 1024, 1024])*\x15\n\rorigin_format\x12\x04\x12\x02ND*\x1e'
            b'\n\x18origin_shape_initialized\x12\x02(\x00*\x1b\n\x15origin_format_for_int\x12\x02\x18\x02r\x03CPU\x8a'
            b'\x02w\x10\x01\x1a\x00"\x02ND*\x1e\n\x18origin_shape_initialized\x12\x02(\x00*\x1b\n\x15'
            b'origin_format_for_int\x12\x02\x18\x02*\x15\n\rorigin_format\x12\x04\x12\x02ND*\x14\n\x0eformat_for_int'
            b'\x12\x02\x18\x02r\x03NPU\x92\x02w\x10\x01\x1a\x00"\x02ND*\x14\n\x0eformat_for_int\x12\x02\x18\x02*\x1e'
            b'\n\x18origin_shape_initialized\x12\x02(\x00*\x1b\n\x15origin_format_for_int\x12\x02\x18\x02*\x15\n\r'
            b'origin_format\x12\x04\x12\x02NDr\x03NPU2\x88\x04\n\x06Cast_1\x12\x04Cast*\x05Mul:0R\x1b\n\x12'
            b'_output_name_value\x12\x05\n\x03\x1a\x01\x00R\x18\n\x0f_input_name_key\x12\x05\n\x03\x12\x01xR\x19\n'
            b'\x10_output_name_key\x12\x05\n\x03\x12\x01yR\x0e\n\x08dst_type\x12\x02\x18\x01R\x1a\n\x11'
            b'_input_name_value\x12\x05\n\x03\x1a\x01\x00\x8a\x02w\x10\x01\x1a\x00"\x02ND*\x1b\n\x15'
            b'origin_format_for_int\x12\x02\x18\x02*\x1e\n\x18origin_shape_initialized\x12\x02(\x00*\x14\n\x0e'
            b'format_for_int\x12\x02\x18\x02*\x15\n\rorigin_format\x12\x04\x12\x02NDr\x03NPU\x92\x02\xf7\x01\x10'
            b'\x02\x1a\x00"\x02ND*\x1e\n\x18origin_shape_initialized\x12\x02(\x00*\x1b\n\x15origin_format_for_int'
            b'\x12\x02\x18\x02*\x14\n\x0eformat_for_int\x12\x02\x18\x02*\x15\n\rorigin_format\x12\x04\x12\x02ND*1'
            b'\n\x0f_fx_tensor_name\x12\x1e\x12\x1cmul-aten.mul.Tensor.OUTPUT.0*K\n\x05_meta\x12B\x12@Tensor(dtype'
            b'=torch.float16, shape=torch.Size([1024, 1024, 1024])r\x03NPU2\xd5\x04\n\tNetOutput\x12\tNetOutput*'
            b'\x05Add:0*\x08Cast_1:0R%\n\x0f_input_name_key\x12\x12\n\x10\x12\x06input0\x12\x06input1R\x1b\n\x11'
            b'_input_name_value\x12\x06\n\x04\x1a\x02\x00\x01\x8a\x02\xeb\x01\x10\x01\x1a\x00"\x02ND*\x14\n\x0e'
            b'format_for_int\x12\x02\x18\x02*\x1b\n\x15origin_format_for_int\x12\x02\x18\x02*\x1e\n\x18'
            b'origin_shape_initialized\x12\x02(\x00*\x15\n\rorigin_format\x12\x04\x12\x02ND*1\n\x0f_fx_tensor_name'
            b'\x12\x1e\x12\x1cadd-aten.add.Tensor.OUTPUT.0*?\n\x05_meta\x126\x124Tensor(dtype=torch.float32, '
            b'shape=torch.Size([2, 4])r\x03NPU\x8a\x02\xf7\x01\x10\x02\x1a\x00"\x02ND*\x14\n\x0eformat_for_int'
            b'\x12\x02\x18\x02*K\n\x05_meta\x12B\x12@Tensor(dtype=torch.float16, shape=torch.Size([1024, 1024, 1024])'
            b'*\x1e\n\x18origin_shape_initialized\x12\x02(\x00*1\n\x0f_fx_tensor_name\x12\x1e\x12\x1cmul-aten.mul.'
            b'Tensor.OUTPUT.0*\x15\n\rorigin_format\x12\x04\x12\x02ND*\x1b\n\x15origin_format_for_int\x12\x02\x18'
            b'\x02r\x03NPUZ\x14\n\x0e_executor_type\x12\x02\x18\x00Z\x11\n\x0bmemory_size\x12\x02\x18\x00Z\x11\n'
            b'\x0bweight_size\x12\x02\x18\x00Z\x10\n\nstream_num\x12\x02\x18\x00Z\x15\n\x0btarget_type\x12\x06\x12'
            b'\x04MINIZ\x15\n\x0fp2p_memory_size\x12\x02\x18\x00Z\x0f\n\tevent_num\x12\x02\x18\x00Z\x0f\n\t'
            b'label_num\x12\x02\x18\x00')
        assert compare_with_base_line('./test_export_file_path/export.air', baseline_fileconst)

    def test_export_with_atc_config_generated(self):
        def get_sub_path_dynamo_pbtxt(export_path, rankid):
            return export_path + "/rank_" + str(rankid) + "/dynamo.pbtxt"

        def get_model_relation_config(export_path):
            return export_path + "/model_relation_config.json"

        def get_numa_config(export_path):
            return export_path + "/numa_config.json"

        def mp():
            world_size = 2
            torch.multiprocessing.spawn(example_atc_config_generated, args=(world_size,), nprocs=world_size, join=True)

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29505"

        mp()

        file_name = get_sub_path_dynamo_pbtxt("export_file", 0)
        with open(file_name, 'r') as f:
            src = f.read()
        assert src.count("op: \"Const\"") == 2
        assert src.count("op: \"Data\"") == 2
        assert src.count("op: \"HcomAllReduce\"") == 1
        assert src.count("key: \"ranklist\"") == 1

        file_name = get_model_relation_config("export_file")
        with open(file_name, 'r') as f:
            src = f.read()
        assert src.count("\"submodel_name\": \"export0.air\"") == 2
        assert src.count("\"group_rank_list\": \"[0, 1]\"") == 1
        assert src.count("model_instance_id") == 4
        assert src.count("0:0:0") == 1
        assert src.count("0:0:1") == 1

        file_name = get_numa_config("export_file")
        with open(file_name, 'r') as f:
            src = f.read()
        assert src.count("\"item_id\": 0") == 1
        assert src.count("\"item_id\": 1") == 1

    def test_export_bf16(self):
        class Model(torch.nn.Module):

            def __init__(self):
                super().__init__()
                size = 10
                self.p1 = torch.nn.Parameter(torch.randn([size], dtype=torch.bfloat16))

            def forward(self, x, y):
                x = x + y + torch.ones([2, 4], dtype=torch.float16)
                w = self.p1 * 2
                return x, w

        model = Model()
        x = torch.randn([2, 4], dtype=torch.bfloat16)
        y = torch.randn([2, 4], dtype=torch.bfloat16)

        export_path1 = "test_export_file_path"

        torchair.dynamo_export(x, y, model=model, export_path=export_path1, dynamic=False)

        dumped_file_list = get_dumped_file_list(export_path1)
        dumped_file_list.sort(key=lambda file_name: os.path.getmtime(os.path.join(export_path1, file_name)))
        assert dumped_file_list.__len__() > 0
        file_name = os.path.join(export_path1, dumped_file_list[-1])

        with open(file_name, 'r') as f:
            src = f.read()

        assert src.count("op: \"Const\"") == 4
        assert src.count("op: \"Data\"") == 2
        assert src.count("op: \"Shape\"") == 0
        assert src.count("dtype: DT_BF16") == 13
        assert src.count("  dim: 10") == 3

    def test_export_enable_record_nn_module_stack(self):
        class Model2(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                z = x + y
                return torch.chunk(z, 2, dim=0)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.submodel = Model2()

            def forward(self, x, y):
                x = x * 2
                return self.submodel(x, y)

        model = Model()
        x = torch.randn([2, 4], dtype=torch.bfloat16)
        y = torch.randn([2, 4], dtype=torch.float16)
        from torchair.configs.compiler_config import CompilerConfig
        export_path1 = "test_export_file_path"
        config = CompilerConfig()
        config.export.experimental.enable_record_nn_module_stack = True
        torchair.dynamo_export(x, y, model=model, export_path=export_path1, dynamic=False, config=config)

        dumped_file_list = get_dumped_file_list(export_path1)
        dumped_file_list.sort(key=lambda file_name: os.path.getmtime(os.path.join(export_path1, file_name)))
        assert dumped_file_list.__len__() > 0
        file_name = os.path.join(export_path1, dumped_file_list[-1])

        with open(file_name, 'r') as f:
            src = f.read()
        assert src.count("\"nn_module_stack\"") == 6  # 插入了2个cast,有两个输出
        assert src.count("Model2") == 6

    def test_export_with_sym_pack(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y, z, p, m, n):
                b = p.view([x]) + z.view([x]) + z.view([x])
                c = m.view([x, x]).sum()
                a = torch.stack([n, n, n, n])
                d = (m.view([4, y, y]) + a).sum()
                return b + c - d

        model = Model()
        z = torch.randn([3, 2])
        p = torch.randn([2, 3])
        m = torch.randn([36])
        n = torch.randn([3, 3])

        export_path1 = "test_export_file_path"
        if os.path.exists(export_path1):
            shutil.rmtree(export_path1)
        torchair.dynamo_export(6, 3, z, p, m, n, model=model, export_path=export_path1, dynamic=True)

        dumped_file_list = get_dumped_file_list(export_path1)
        dumped_file_list.sort(key=lambda file_name: os.path.getmtime(os.path.join(export_path1, file_name)))
        assert dumped_file_list.__len__() > 0
        file_name = os.path.join(export_path1, dumped_file_list[-1])

        with open(file_name, 'r') as f:
            src = f.read()

        assert src.count("op: \"Data\"") == 6
        assert src.count("op: \"Pack\"") == 2

    def test_export_with_ref_data(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                a = x * x
                return y.add_(a)

        model = Model()
        export_path = "test_export_file_path"
        if os.path.exists(export_path):
            shutil.rmtree(export_path)
        x = torch.randn([2, 4], dtype=torch.float16)
        y = torch.randn([2, 4], dtype=torch.float16)
        torchair.dynamo_export(x, y, model=model, export_path=export_path, dynamic=False)

    def test_one_graph_with_same_ref_data(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x.mul_(2)
                return x, y.add_(1)

        model = Model()
        export_path = "test_export_file_path"
        if os.path.exists(export_path):
            shutil.rmtree(export_path)
        x = torch.randn([2, 4], dtype=torch.float16)
        torchair.dynamo_export(x, x, model=model, export_path=export_path, dynamic=False)

        dumped_file_list = get_dumped_file_list(export_path)
        dumped_file_list.sort(key=lambda file_name: os.path.getmtime(os.path.join(export_path, file_name)))
        assert dumped_file_list.__len__() > 0
        file_name = os.path.join(export_path, dumped_file_list[-1])

        with open(file_name, 'r') as f:
            src1 = f.read()

        shape_str = "_".join(str(sh) for sh in x.shape)
        stride_str = "_".join(str(std) for std in x.stride())
        offset_str = str(x.storage_offset())
        new_refdata_name = "RefData_" + shape_str + "_" + stride_str + "_" + offset_str + "_" + str(
            id(x))

        name_str = f'name: \"{new_refdata_name}\"'
        assert src1.count(name_str) == 1
        op_str = f'op: "RefData"'
        assert src1.count(op_str) == 1



    def test_two_graph_with_same_ref_data(self):
        class Model1(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                a = x * x
                return y.add_(a)
        class Model2(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                a = torch.mm(x, x)
                b = a + x
                return y.mul_(b)

        model = Model1()
        export_path1 = "test_export_file_path"
        export_path2 = "test_export_file_path2"

        if os.path.exists(export_path1):
            shutil.rmtree(export_path1)
        input_tensor1 = torch.randn([2, 4], dtype=torch.float16)
        input_tensor2 = torch.randn([2, 4], dtype=torch.float16)
        print(id(input_tensor1))
        print(id(input_tensor2))

        torchair.dynamo_export(input_tensor1, input_tensor2, model=model, export_path=export_path1, dynamic=False)

        torchair.dynamo_export(input_tensor1, input_tensor2, model=model, export_path=export_path2, dynamic=False)

        dumped_file_list = get_dumped_file_list(export_path1)
        dumped_file_list.sort(key=lambda file_name: os.path.getmtime(os.path.join(export_path1, file_name)))
        assert dumped_file_list.__len__() > 0
        file_name = os.path.join(export_path1, dumped_file_list[-1])

        with open(file_name, 'r') as f:
            src1 = f.read()

        shape_str = "_".join(str(x) for x in input_tensor2.shape)
        stride_str = "_".join(str(x) for x in input_tensor2.stride())
        offset_str = str(input_tensor2.storage_offset())
        new_refdata_name = "RefData_" + shape_str + "_" + stride_str + "_" + offset_str + "_" + str(
            id(input_tensor2))

        sub_str = f'name: \"{new_refdata_name}\"'
        assert src1.count(sub_str) == 1

        dumped_file_list = get_dumped_file_list(export_path2)
        dumped_file_list.sort(key=lambda file_name: os.path.getmtime(os.path.join(export_path2, file_name)))
        assert dumped_file_list.__len__() > 0
        file_name = os.path.join(export_path2, dumped_file_list[-1])

        with open(file_name, 'r') as f:
            src2 = f.read()
        assert src2.count(sub_str) == 1


class AllReduceSingeGroup(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.tensor([[1.1, 1.1], [1.1, 1.1]]))
        self.p2 = torch.nn.Parameter(torch.tensor([[2.2, 2.2], [3.3, 3.3]]))

    def forward(self, x, y):
        x = x + y + self.p + self.p2
        torch.distributed.all_reduce(x)
        return x


class AllReduceMultiGroup(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = funcol.all_reduce(x, reduceOp='SUM', group=[0, 1], tag='test1')
        x = funcol.all_reduce(x, reduceOp='SUM', group=[0, 1], tag='test2')
        x = funcol.all_reduce(x, reduceOp='SUM', group=[0, 1], tag='test3')
        x = funcol.all_reduce(x, reduceOp='SUM', group=[0, 1], tag='test1')  # 重复的group case
        return x


class DistReduceScatterTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, output):
        from torch.distributed.distributed_c10d import _world
        # 必须要带group参数
        torch.distributed.reduce_scatter_tensor(output, x, group=_world.default_pg)
        return x


class FuncolReduceScatterTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        from torch.distributed.distributed_c10d import _world
        out = funcol.reduce_scatter_tensor(x, "sum", scatter_dim=-1, group=_world.default_pg)
        return out


def example(rank, world_size):
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    x = torch.ones([2, 2], dtype=torch.int32)
    y = torch.ones([2, 2], dtype=torch.int32)
    mod = AllReduceSingeGroup()
    torchair.dynamo_export(x, y, model=mod)

    mod2 = AllReduceMultiGroup()
    xx2 = torch.ones([3], dtype=torch.int32)
    torchair.dynamo_export(xx2, model=mod2, dynamic=False, export_path="false_export_path2")
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    config.export.experimental.auto_atc_config_generated = True
    torchair.dynamo_export(xx2, model=mod2, dynamic=True, export_path="true_export_path2",
                           config=config)

    mod3 = DistReduceScatterTensor()
    xx3 = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    output = torch.empty([2], dtype=torch.int32)
    torchair.dynamo_export(xx3, output, model=mod3, dynamic=True, export_path="true_export_path3")


def example_atc_config_generated(rank, world_size):
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    x = torch.ones([2, 2], dtype=torch.int32)
    y = torch.ones([2, 2], dtype=torch.int32)
    mod = AllReduceSingeGroup()
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    config.export.experimental.auto_atc_config_generated = True
    torchair.dynamo_export(x, y, model=mod, config=config)


if __name__ == '__main__':
    unittest.main()
