__all__ = ["cache_compile", "readable_cache"]

from typing import Optional, Callable

from npugraph_ex.configs.compiler_config import CompilerConfig
from npugraph_ex.inference._cache_compiler import cache_compile as _cache_compile
from npugraph_ex.inference._cache_compiler import readable_cache as _readable_cache


def cache_compile(func, *, config: Optional[CompilerConfig] = None, dynamic: bool = True,
                  cache_dir: Optional[str] = None, global_rank: Optional[int] = None, tp_rank: Optional[int] = None,
                  pp_rank: Optional[int] = None, **kwargs) -> Callable:
    return _cache_compile(func, config=config, dynamic=dynamic,
                          cache_dir=cache_dir, global_rank=global_rank,
                          tp_rank=tp_rank, pp_rank=pp_rank, **kwargs)


def readable_cache(cache_bin, print_output=True, file=None):
    return _readable_cache(cache_bin, print_output=print_output, file=file)
