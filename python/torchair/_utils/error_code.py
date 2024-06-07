import os
import sys
import time
import traceback


def format_error_msg():
    error_msg = "\n[ERROR] {time} (PID:{pid}, Device:{device}, RankID:{rank})" \
                " {error_code} {submodule_name} {error_code_msg}"

    return error_msg.format(
        time=time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()),
        pid=os.getpid(),
        device=-1,
        rank=-1,
        error_code="ERR{:0>2d}{:0>3d}".format(3, 5),
        submodule_name="GRAPH",
        error_code_msg="internal error")


def get_error_msg(e):
    if "torch_npu" not in sys.modules:
        return format_error_msg()

    try:
        from torch_npu.utils._error_code import ErrCode, graph_error
    except (ModuleNotFoundError, ImportError):
        return format_error_msg()

    code = ErrCode.INTERNAL
    if isinstance(e, NotImplementedError):
        code = ErrCode.NOT_SUPPORT
    elif isinstance(e, ValueError) or isinstance(e, AssertionError):
        code = ErrCode.PARAM
    elif isinstance(e, FileNotFoundError):
        code = ErrCode.NOT_FOUND
    return graph_error(code)


def pretty_error_msg(func):
    def wapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = [str(e), traceback.format_exc(), get_error_msg(e)]
            raise type(e)("\n".join(msg))

    return wapper
