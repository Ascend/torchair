import importlib
from torchair.ge_concrete_graph.ge_converter.prim import *
from torchair.ge_concrete_graph.ge_converter.prims import *
from torchair.ge_concrete_graph.ge_converter.higher_order import *
from torchair.ge_concrete_graph.ge_converter.quantized import *
from torchair.ge_concrete_graph.ge_converter.rngprims import *
from torchair.ge_concrete_graph.ge_converter.c10d_functional import *
import torchair.ge_concrete_graph.ge_converter.builtin_converters
import torchair.ge_concrete_graph.ge_converter.aten as aten
for name in aten.__all__:
    module_name = "torchair.ge_concrete_graph.ge_converter.aten." + name
    try:
        importlib.import_module(module_name)
    except AttributeError:
        pass