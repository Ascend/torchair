import os
import pkgutil
import importlib


__all__ = list(module for _, module, _ in pkgutil.iter_modules([os.path.dirname(__file__)]))

for name in __all__:
    module_name = "torchair._ge_concrete_graph.ge_converter.custom." + name
    try:
        importlib.import_module(module_name)
    except AttributeError:
        pass