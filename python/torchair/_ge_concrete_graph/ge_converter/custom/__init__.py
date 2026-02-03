import os
import pkgutil
import importlib
import warnings


__all__ = list(module for _, module, _ in pkgutil.iter_modules([os.path.dirname(__file__)]))

for name in __all__:
    module_name = "torchair._ge_concrete_graph.ge_converter.custom." + name
    try:
        importlib.import_module(module_name)
    except AttributeError as e:
        warnings.warn(f"Failed to import {module_name} due to AttributeError: {e}")