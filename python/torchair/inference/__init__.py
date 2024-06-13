from typing import List, Optional, Callable, Dict

import torch
from torchair.inference._cache_compiler import cache_compile as _cache_compile
from torchair.inference._cache_compiler import readable_cache as _readable_cache
from torchair.inference._gear_utils import set_dim_gears as _set_dim_gears
from torchair.configs.compiler_config import CompilerConfig

__all__ = ["cache_compile", "readable_cache", "set_dim_gears"]


def cache_compile(func, *, config: Optional[CompilerConfig] = None, dynamic: bool = True,
                  cache_dir: Optional[str] = None, global_rank: Optional[int] = None, tp_rank: Optional[int] = None,
                  pp_rank: Optional[int] = None, **kwargs) -> Callable:
    return _cache_compile(func, config=config, dynamic=dynamic,
                          cache_dir=cache_dir, global_rank=global_rank,
                          tp_rank=tp_rank, pp_rank=pp_rank, **kwargs)


def readable_cache(cache_bin, print_output=True, file=None):
    return _readable_cache(cache_bin, print_output=print_output, file=file)


def set_dim_gears(t: torch.Tensor, dim_gears: Dict[int, List[int]]):
    _set_dim_gears(t, dim_gears=dim_gears)
