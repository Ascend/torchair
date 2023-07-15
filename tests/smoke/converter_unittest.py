import functools
import torch
import torchair as tng
from torchair.ge_concrete_graph.fx2ge_converter import Converter
from torchair.ge_concrete_graph.fx2ge_converter import test_converter as _test_inner
from torchair import CompilerConfig
aten = torch.ops.aten


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

    Converter.compile_backend = tng.get_npu_backend()
    Converter.result_checker = functools.partial(
        check_result, stop_when_error=stop_when_error)

    _test_inner(aten_ops)

    for i, (name, error) in enumerate(_FAILED_CASES):
        print(f"--------------------", flush=True)
        print(f"FAILED [{i+1}/{len(_FAILED_CASES)}] {name}", flush=True)
        print(f"Error: {error}", flush=True)

    assert len(_FAILED_CASES) == 0, f"{len(_FAILED_CASES)} testcases failed"


if __name__ == "__main__":
    test_converter()
