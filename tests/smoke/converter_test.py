import functools
from typing import Callable
import torch
import torchair as tng
from torchair.ge_concrete_graph.fx2ge_converter import Converter
from torchair.ge_concrete_graph.fx2ge_converter import _declare_supported_converters
from torchair import CompilerConfig
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor
aten = torch.ops.aten


def _eager_aten_call(*args, aten_op, **kwargs):
    return aten_op(*args, **kwargs)


def _test_converter(converter: Converter, *, backend, result_checker):
    if converter.supported_cases is None:
        print(f"No supported_cases for {converter._aten_op}", flush=True)
        return

    eager_func = functools.partial(
        _eager_aten_call, aten_op=converter._aten_op)
    compiled_func = torch.compile(eager_func, backend=backend)

    print(
        f"Testing {converter._aten_op} with {len(converter.supported_cases)} cases", flush=True)
    for testcase in converter.supported_cases:
        print(f"[RUN] {testcase.title}", flush=True)
        args = []
        for arg in testcase.args:
            if isinstance(arg, (list, tuple)):
                args.append([(v.t() if isinstance(v, _TypedTensor) else v)
                             for v in arg])
            elif isinstance(arg, _TypedTensor):
                args.append(arg.t())
            else:
                args.append(arg)
        kwargs = testcase.kwargs
        for k, v in kwargs.items():
            if isinstance(v, _TypedTensor):
                kwargs[k] = v.t()

        try:
            backend_results = compiled_func(*args, **kwargs)
        except Exception as e:
            backend_results = e

        try:
            eager_results = eager_func(*args, **kwargs)
        except Exception as e:
            eager_results = e

        if result_checker is not None:
            result_checker(
                testcase.title, backend_results, eager_results)
        del args, kwargs, backend_results, eager_results


def _test_converters(aten_ops, *, backend, result_checker):
    supported_converters = _declare_supported_converters()
    if aten_ops is None:
        aten_ops = supported_converters.keys()
    elif isinstance(aten_ops, Callable):
        aten_ops = (aten_ops, )
    else:
        assert isinstance(aten_ops, (list, tuple))

    for aten_op in aten_ops:
        if aten_op not in supported_converters.keys():
            raise RuntimeError(f"Cannot find testcase for {aten_op}")
        _test_converter(
            supported_converters[aten_op], backend=backend, result_checker=result_checker)


def check_tensor_same(a, b):
    assert a.dtype == b.dtype, f"Datatype mismatch {a.dtype} vs. {b.dtype}"
    assert a.size() == b.size(), f"Shape mismatch {a.size()} vs. {b.size()}"

    if a.dtype in (torch.float16, torch.float32, torch.float64):
        assert torch.allclose(
            a, b, rtol=1e-3, atol=1e-5), f"Value mismatch {a} vs. {b}"
    else:
        assert torch.all(a == b), f"Value mismatch {a} vs. {b}"


def _check_result(compiled_rets, eager_rets):
    if isinstance(eager_rets, Exception):
        raise eager_rets
    if isinstance(compiled_rets, Exception):
        raise compiled_rets
    assert type(compiled_rets) == type(
        eager_rets), f"result type mismatch {type(compiled_rets)} vs. {type(eager_rets)}"
    compiled_rets = (compiled_rets, ) if isinstance(
        compiled_rets, torch.Tensor) else compiled_rets
    eager_rets = (eager_rets, ) if isinstance(
        eager_rets, torch.Tensor) else eager_rets
    for c_ret, e_ret in zip(compiled_rets, eager_rets):
        assert type(c_ret) == type(
            e_ret), f"result type mismatch {type(c_ret)} vs. {type(e_ret)}"
        if isinstance(c_ret, (list, tuple)):
            for c_tensor, e_tensor in zip(c_ret, e_ret):
                check_tensor_same(c_tensor, e_tensor)
        else:
            assert isinstance(c_ret, torch.Tensor)
            check_tensor_same(c_ret, e_ret)


_FAILED_CASES = []


def check_result(case_title, compiled_rets, eager_rets, *, stop_when_error=False):
    global _FAILED_CASES
    try:
        _check_result(compiled_rets, eager_rets)
        print(f"[PASS] {case_title}", flush=True)
    except Exception as e:
        print(f"[FAILED] {case_title}", flush=True)
        if stop_when_error:
            raise e
        else:
            _FAILED_CASES.append((case_title, str(e)))
            del e


def test_converter(aten_ops=None, *, stop_when_error=False):
    global _FAILED_CASES
    _FAILED_CASES.clear()

    compile_backend = tng.get_npu_backend()
    result_checker = functools.partial(
        check_result, stop_when_error=stop_when_error)

    _test_converters(aten_ops, backend=compile_backend,
                     result_checker=result_checker)

    for i, (name, error) in enumerate(_FAILED_CASES):
        print(f"--------------------", flush=True)
        print(f"FAILED [{i+1}/{len(_FAILED_CASES)}] {name}", flush=True)
        print(f"Error: {error}", flush=True)

    assert len(_FAILED_CASES) == 0, f"{len(_FAILED_CASES)} testcases failed"


if __name__ == "__main__":
    # 不传入任何值表示测试全部的converter，
    # 可以通过传入单个op或者op列表来测试指定的op,
    # 例如：test_converter(aten.add.Tensor)只会测试aten.add.Tensor的converter
    test_converter()
