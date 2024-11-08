import importlib
from torchair._ge_concrete_graph.ge_converter.prim import *
from torchair._ge_concrete_graph.ge_converter.higher_order import *
from torchair._ge_concrete_graph.ge_converter.quantized import *
from torchair._ge_concrete_graph.ge_converter.rngprims import *
from torchair._ge_concrete_graph.ge_converter.c10d_functional import *
import torchair._ge_concrete_graph.ge_converter.builtin_converters


def try_import_submodule(submodule_name):
    module_name = f"torchair._ge_concrete_graph.ge_converter.{submodule_name}"
    submodule = importlib.import_module(module_name)
    for name in submodule.__all__:
        file_name = module_name + "." + name
        try:
            importlib.import_module(file_name)
        except AttributeError:
            pass


try_import_submodule("prims")
try_import_submodule("aten")