import functools
from typing import Callable
import torch
import torch_npu
import sys
import contextlib
import torchair as tng
from torchair._ge_concrete_graph.fx2ge_converter import Converter
from torchair._ge_concrete_graph.fx2ge_converter import _declare_supported_converters
from torchair import CompilerConfig
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor
aten = torch.ops.aten

_torch_npu_module = torch_npu
torch_npu.npu.set_device(0)


@contextlib.contextmanager
def _npu_executor_as_default():
    global _torch_npu_module
    try:
        sys.modules['torch_npu'] = _torch_npu_module
        yield
    finally:
        del sys.modules['torch_npu']


def as_tensor(spec: _TypedTensor):
    device = 'npu' if ('torch_npu' in sys.modules) else 'cpu'
    dims = spec.dims
    if spec.value is not None:
        return torch.tensor(spec.value, dtype=spec.dtype, device=device)

    if spec.value_range is not None:
        low = spec.value_range[0]
        high = spec.value_range[1]
        result = torch.rand(*dims) * (high - low) + low
        return result.to(spec.dtype).to(device)

    return torch.randn(*dims).to(spec.dtype).to(device)


def _eager_aten_call(aten_op):
    def inner_run(*args, **kwargs):
        outs = aten_op(*args, **kwargs)
        if isinstance(outs, int):
            return outs
        if isinstance(outs, (list, tuple)):
            return [out.clone() for out in outs]
        return outs.clone()
    return inner_run


def _assemble_testcase_inputs(testcase):
    args = []
    for arg in testcase.args:
        if isinstance(arg, (list, tuple)):
            args.append([(as_tensor(v) if isinstance(v, _TypedTensor) else v)
                         for v in arg])
        elif isinstance(arg, _TypedTensor):
            args.append(as_tensor(arg))
        else:
            args.append(arg)
    kwargs = {}
    for k, v in testcase.kwargs.items():
        if isinstance(v, _TypedTensor):
            kwargs[k] = as_tensor(v)
        else:
            kwargs[k] = v
    return args, kwargs


def reset_rng_state():
    torch.manual_seed(0)
    torch.npu.manual_seed(0)


def _test_converter(converter: Converter, *, backend, result_checker):
    if converter.supported_cases is None:
        print(f"No supported_cases for {converter._aten_op}", flush=True)
        return

    eager_func = _eager_aten_call(converter._aten_op)
    compiled_func = torch.compile(eager_func, backend=backend)

    print(
        f"Testing {converter._aten_op} with {len(converter.supported_cases)} cases", flush=True)
    for i, testcase in enumerate(converter.supported_cases):
        case_name = f"{converter._aten_op} testcase {i + 1}/{len(converter.supported_cases)} with inputs: {testcase}"
        print(f"[RUN] {case_name}", flush=True)

        args, kwargs = _assemble_testcase_inputs(testcase)

        reset_rng_state()
        try:
            backend_results = compiled_func(*args, **kwargs)
        except Exception as e:
            backend_results = e

        reset_rng_state()
        try:
            eager_results = eager_func(*args, **kwargs)
        except Exception as e:
            if 'torch_npu' in sys.modules:
                eager_results = e
            else:
                print(
                    f"[WARNNING] Fallback to get npu eager result as cpu error {e}", flush=True)
                try:
                    with _npu_executor_as_default():
                        npu_args, npu_kwargs = _assemble_testcase_inputs(
                            testcase)
                        eager_results = eager_func(*npu_args, **npu_kwargs)
                except Exception as e:
                    eager_results = e

        if result_checker is not None:
            result_checker(case_name, backend_results, eager_results)
        del args, kwargs, backend_results, eager_results


def _test_converters(aten_ops, *, backend, result_checker):
    try:
        import torchair._ge_concrete_graph.ge_converter.custom
    except Exception as e:
        print(f"Warning EXCEPTION when try to import custom op converters, will continue execute: {e}")
    supported_converters = _declare_supported_converters()
    if aten_ops is None:
        aten_ops = supported_converters.keys()
    elif isinstance(aten_ops, Callable):
        aten_ops = (aten_ops, )
    elif isinstance(aten_ops, str):
        prefix = aten_ops
        aten_ops = []
        for aten_op in supported_converters.keys():
            if f'{aten_op}'.startswith(prefix):
                print(
                    f"Converter of {aten_op} match prefix '{prefix}'", flush=True)
                aten_ops.append(aten_op)
        if len(aten_ops) == 0:
            raise RuntimeError(
                f"Cannot find testcase match prefix '{prefix}'")
    else:
        assert isinstance(aten_ops, (list, tuple))

    for aten_op in aten_ops:
        if aten_op not in supported_converters.keys():
            raise RuntimeError(f"Cannot find testcase for {aten_op}")
        _test_converter(
            supported_converters[aten_op], backend=backend, result_checker=result_checker)


def check_tensor_same(a, b):
    a = a.cpu()
    b = b.cpu()
    assert a.dtype == b.dtype, f"Datatype mismatch {a.dtype} vs. {b.dtype}"
    assert a.size() == b.size(), f"Shape mismatch {a.size()} vs. {b.size()}"

    if a.dtype in (torch.float16, torch.float32, torch.float64):
        assert torch.allclose(
            a, b, rtol=1e-3, atol=1e-5, equal_nan=True), f"Value mismatch {a} vs. {b}"
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
        compiled_rets, (torch.Tensor, int)) else compiled_rets
    eager_rets = (eager_rets, ) if isinstance(
        eager_rets, (torch.Tensor, int)) else eager_rets
    for c_ret, e_ret in zip(compiled_rets, eager_rets):
        assert type(c_ret) == type(
            e_ret), f"result type mismatch {type(c_ret)} vs. {type(e_ret)}"
        if isinstance(c_ret, (list, tuple)):
            for c_tensor, e_tensor in zip(c_ret, e_ret):
                check_tensor_same(c_tensor, e_tensor)
        else:
            if isinstance(c_ret, int):
                assert c_ret == e_ret
                continue
            assert isinstance(c_ret, torch.Tensor), f"unsupported result type {type(c_ret)}"
            check_tensor_same(c_ret, e_ret)


_FAILED_CASES = []


def check_result(case_name, compiled_rets, eager_rets, *, stop_when_error=False):
    global _FAILED_CASES
    try:
        _check_result(compiled_rets, eager_rets)
        print(f"[PASS] {case_name}", flush=True)
    except Exception as e:
        print(f"[FAILED] {case_name}", flush=True)
        if stop_when_error:
            raise e
        else:
            _FAILED_CASES.append((case_name, str(e)))
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

    if len(_FAILED_CASES) == 0:
        print('Test result: All testcases passed', flush=True)
    else:
        print(
            f"Test result: {len(_FAILED_CASES)} testcases failed", flush=True)


if __name__ == "__main__":
    case_pattern = None
    if len(sys.argv) == 2:
        case_pattern = sys.argv[1]
    test_converter(case_pattern)
