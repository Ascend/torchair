import pkgutil
import importlib
from typing import Callable
import inspect
import os
import json
import unittest
from pathlib import Path

import torchair


def _discover_path_importables(pkg_pth, pkg_name):
    """Yield all importables under a given path and package.

    This is like pkgutil.walk_packages, but does *not* skip over namespace
    packages.
    """
    for dir_path, _d, file_names in os.walk(pkg_pth):
        pkg_dir_path = Path(dir_path)

        if pkg_dir_path.parts[-1] == '__pycache__':
            continue

        if all(Path(_).suffix != '.py' for _ in file_names):
            continue

        rel_pt = pkg_dir_path.relative_to(pkg_pth)
        pkg_pref = '.'.join((pkg_name,) + rel_pt.parts)
        yield from (
            pkg_path
            for _, pkg_path, _ in pkgutil.walk_packages(
            (str(pkg_dir_path),), prefix=f'{pkg_pref}.', onerror=lambda x: print(f'Failed import {x}')
        )
        )


def _read_allow_api_json():
    with open(os.path.join(os.path.dirname(__file__), 'allowlist_for_publicAPI.json')) as allow_file_json:
        allow_api_dict = json.load(allow_file_json)
        allow_api = []
        for key, values in allow_api_dict.items():
            allow_api.extend([f'{key}.{value}' for value in values])
    return allow_api


def _is_alias(public_api_fun_name, public_api):

    if public_api.split('.')[-1] in public_api_fun_name:
        return True
    return False


SKIP_CHECK_MODULES = [
    "torchair.ge_concrete_graph",
]


class TestPublicBindings(unittest.TestCase):
    @staticmethod
    def _is_mod_public(modname):
        split_strs = modname.split('.')
        for elem in split_strs:
            if elem.startswith("_"):
                return False
        return True

    @staticmethod
    def _is_legacy_public(modname):
        for mod in SKIP_CHECK_MODULES:
            if modname.startswith(mod):
                return True
        return False

    def test_correct_module_names(self):
        '''
        An API is considered public, if  its  `__module__` starts with `torchair.`
        and there is no name in `__module__` or the object itself that starts with “_”.
        Each public package should either:
        - (preferred) Define `__all__` and all callables and classes in there must have their
         `__module__` start with the current submodule's path. Things not in `__all__` should
          NOT have their `__module__` start with the current submodule.
        - (for simple python-only modules) Not define `__all__` and all the elements in `dir(submod)` must have their
          `__module__` that start with the current submodule.
        '''
        failure_list = []
        allow_api = _read_allow_api_json()
        public_api_fun_name = set()
        for api in allow_api:
            public_api_fun_name.add(api.split('.')[-1])

        def test_module(modname):
            try:
                if "__main__" in modname:
                    return
                mod = importlib.import_module(modname)
            except Exception:
                # It is ok to ignore here as we have a test above that ensures
                # this should never happen
                return

            if not self._is_mod_public(modname):
                return

                # verifies that each public API has the correct module name and naming semantics

            def check_one_element(elem, modname, mod, *, is_public, is_all):
                if self._is_legacy_public(f'{modname}.{elem}'):
                    return
                obj = getattr(mod, elem)
                if not (isinstance(obj, Callable) or inspect.isclass(obj)):
                    return
                elem_module = getattr(obj, '__module__', None)
                # Only used for nice error message below
                why_not_looks_public = ""
                if elem_module is None:
                    why_not_looks_public = "because it does not have a `__module__` attribute"
                elem_modname_starts_with_mod = elem_module is not None and \
                                               elem_module.startswith(modname) and \
                                               '._' not in elem_module
                if not why_not_looks_public and not elem_modname_starts_with_mod:
                    why_not_looks_public = f"because its `__module__` attribute (`{elem_module}`) is not within the " \
                                           f"torch library or does not start with the submodule where it is " \
                                           f"defined (`{modname}`)"
                # elem's name must NOT begin with an `_` and it's module name
                # SHOULD start with it's current module since it's a public API
                looks_public = not elem.startswith('_') and elem_modname_starts_with_mod
                if not why_not_looks_public and not looks_public:
                    why_not_looks_public = f"because it starts with `_` (`{elem}`)"

                if is_public != looks_public:
                    if is_public:
                        why_is_public = f"it is inside the module's (`{modname}`) `__all__`" if is_all else \
                            "it is an attribute that does not start with `_` on a module that " \
                            "does not have `__all__` defined"
                        fix_is_public = f"remove it from the modules's (`{modname}`) `__all__`" if is_all else \
                            f"either define a `__all__` for `{modname}` or add a `_` at the beginning of the name"
                    else:
                        assert is_all
                        why_is_public = f"it is not inside the module's (`{modname}`) `__all__`"
                        fix_is_public = f"add it from the modules's (`{modname}`) `__all__`"

                    if looks_public:
                        why_looks_public = "it does look public because it follows the rules from the doc above " \
                                           "(does not start with `_` and has a proper `__module__`)."
                        fix_looks_public = "make its name start with `_`"
                    else:
                        why_looks_public = why_not_looks_public
                        if not elem_modname_starts_with_mod:
                            fix_looks_public = "make sure the `__module__` is properly set and points to a submodule " \
                                               f"of `{modname}`"
                        else:
                            fix_looks_public = "remove the `_` at the beginning of the name"

                    failure_list.append(f"# {modname}.{elem}:")
                    is_public_str = "" if is_public else " NOT"
                    failure_list.append(f"  - Is{is_public_str} public: {why_is_public}")
                    looks_public_str = "" if looks_public else " NOT"
                    failure_list.append(f"  - Does{looks_public_str} look public: {why_looks_public}")
                    # Swap the str below to avoid having to create the NOT again
                    failure_list.append("  - You can do either of these two things to fix this problem:")
                    failure_list.append(f"    - To make it{looks_public_str} public: {fix_is_public}")
                    failure_list.append(f"    - To make it{is_public_str} look public: {fix_looks_public}")

                if is_public and looks_public:
                    public_api = f"{modname}.{elem}"
                    if public_api not in allow_api and not _is_alias(public_api_fun_name, public_api):
                        failure_list.append(f"# {public_api} is public api, "
                                            f"please add it to allowlist_for_publicAPI.json.")

            if hasattr(mod, '__all__'):
                public_api = mod.__all__
                all_api = dir(mod)
                for elem in all_api:
                    check_one_element(elem, modname, mod, is_public=elem in public_api, is_all=True)
            else:
                all_api = dir(mod)
                for elem in all_api:
                    if not elem.startswith('_'):
                        check_one_element(elem, modname, mod, is_public=True, is_all=False)

        for modname in _discover_path_importables(str(torchair.__path__[0]), "torchair"):
            test_module(modname)

        test_module('torchair')

        msg = "All the APIs below do not meet our guidelines for public API from " \
              "pytorch wiki Public-API-definition-and-documentation.\n"
        msg += "Make sure that everything that is public is expected (in particular that the module " \
               "has a properly populated `__all__` attribute) and that everything that is supposed to be public " \
               "does look public (it does not start with `_` and has a `__module__` that is properly populated)."
        msg += "\n\nFull list:\n"
        msg += "\n".join(map(str, failure_list))

        # empty lists are considered false in python
        self.assertTrue(not failure_list, msg)


    def test_compatible_api(self):
        # 以下接口均为N+4版本保留兼容性的接口，将在一年之后删除(自2024/7/27始)
        try:
            from torchair.ge_concrete_graph import ge_apis as ge
            from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
            from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec, DataType
            from torchair.ge_concrete_graph.ge_graph import get_default_ge_graph, next_unique_name
            from torchair.ge_concrete_graph.ge_graph import compat_as_bytes, compat_as_bytes_list
            from torchair.ge_concrete_graph.ge_graph import get_invalid_desc, auto_convert_to_tensor
            import torchair.ge_concrete_graph.ge_converter.experimental.hcom_allgather
            from torchair.ge_concrete_graph.utils import dtype_promote
        except:
            raise AssertionError("import compatible api failed, UT failed")
        else:
            self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
