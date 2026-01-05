"""NPU basic configurations"""
import os
import re

from typing import Callable, Optional

from torchair._utils.error_code import pretty_error_msg


class OptionValue:
    """Options for setting npu basic configurations"""

    def __init__(self, default, optional=None):
        self.__default = default
        self.__optional = optional
        self._value = default

    def __bool__(self):
        return bool(self._value)

    @property
    def default(self):
        """Return property"""
        return self.__default

    @property
    def optional(self):
        """Return property"""
        return self.__optional

    @property
    def value(self):
        """Return option value"""
        if self._value is None:
            return None
        if str(self._value) == str(True):
            return "1"
        if str(self._value) == str(False):
            return "0"
        return str(self._value)

    @value.setter
    def value(self, v):
        if isinstance(self.__optional, (tuple, list,)) and v not in self.__optional:
            raise ValueError(f"Value {repr(v)} (type: {type(v)})"
                             f" not in optional list {repr(self.__optional)}"
                             f" (type: {None if len(self.__optional) == 0 else type(self.__optional[0])}).")
        self._value = v


class IntRangeValue(OptionValue):
    def __init__(self, default, value_min, value_max):
        super().__init__(default)
        self.__min = value_min
        self.__max = value_max

    @property
    def value(self):
        return super().value

    @value.setter
    def value(self, v):
        if not isinstance(v, int):
            raise ValueError(f'Please set integer type, but got {type(v)}')
        if v < self.__min or v > self.__max:
            raise ValueError(f'Please set value in [{self.__min}' + ', '
                             + f'{self.__max}], {str(v)} is out of range.')
        self._value = v


class IntListValue(OptionValue):
    def __init__(self, default=None):
        if default is not None and not isinstance(default, (list, tuple)):
            raise TypeError(f"default must be None or list, got {type(default)}")

        super().__init__(default, optional=None)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        if v is not None and not isinstance(v, (list, tuple)):
            raise TypeError(f"the set value must be None or list, got {type(v)}")

        v = [] if v is None else v
        if not all(isinstance(val, int) for val in v):
            raise TypeError(f"the set value must be int list, got list type: {[type(val) for val in v]}")

        self._value = v


class StrOptionValue(OptionValue):
    def __init__(self, default=None):
        if default is not None and not isinstance(default, str):
            raise TypeError(f"default must be str or None, got {type(default)}")
        super().__init__(default, optional=None)

    @property
    def value(self):
        return super().value

    @value.setter
    def value(self, v):
        if v is not None and not isinstance(v, str):
            raise TypeError(f"value must be str or None, got {type(v)}")
        self._value = v


class FileValue(OptionValue):
    @property
    def value(self):
        return super().value

    @value.setter
    def value(self, v):
        if v is not None:
            if not (os.path.exists(v) and os.path.isfile(v)):
                raise FileNotFoundError('Please set legal file path, '
                                        + f'{str(v)} is not found or is not a file!')
        self._value = v


class MustExistedPathValue(OptionValue):
    @property
    def value(self):
        return super().value

    @value.setter
    def value(self, v):
        if v is None or not (os.path.exists(v) and os.path.isdir(v)):
            raise FileNotFoundError('Please set legal dir path, '
                                    + f'{str(v)} is not found or is not a file directory!')
        self._value = v
        

class MustExistedFileAddr(OptionValue):
    @property
    def value(self):
        return super().value

    @value.setter
    def value(self, v, suffix=".json"):
        if v is not None:
            if v == "":
                raise ValueError(f'File path cannot be empty!')
            if not (os.path.exists(v) and v.endswith(suffix) and os.path.isfile(v)):
                raise FileNotFoundError('Please set legal file path, '
                                        + f'{str(v)} is not found or is not \'{suffix}\' file!')
        self._value = v


class RegexValue(OptionValue):
    def __init__(self, default, regex, example):
        super().__init__(default)
        self.__regex = regex
        self.__example = example

    @property
    def value(self):
        return super().value

    @value.setter
    def value(self, v):
        if not re.match(self.__regex, v):
            raise ValueError(f'Please set legal regex value format match {self.__regex}, '
                             + f'{str(v)} is Illegal format, correct example:{self.__example}.')
        self._value = v


class DeprecatedValue(OptionValue):
    def __init__(self, optional, *, replacement):
        super().__init__(None, optional)
        self.replacement = replacement


class CallableValue(OptionValue):
    def __init__(self, default=None):
        if default is not None and not isinstance(default, Callable):
            raise TypeError(f"default must be Callable or None, got {type(default)}")
        super().__init__(default, optional=None)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        if v is not None and not isinstance(v, Callable):
            raise TypeError(f"value must be Callable or None, got {type(v)}")
        self._value = v    


class NpuBaseConfig:
    """NPU basic configurations"""

    def __init__(self):
        self._fixed_attrs = []
        for k, v in self.__dict__.items():
            if isinstance(v, (OptionValue, NpuBaseConfig)):
                self._fixed_attrs.append(k)

    @pretty_error_msg
    def __setattr__(self, key, value):
        if hasattr(self, '_fixed_attrs'):
            if key not in self._fixed_attrs:
                raise ValueError(self.__class__.__name__ + " has no option " + key + ", all options " +
                                 str(self._fixed_attrs))
            if isinstance(getattr(self, key), OptionValue):
                getattr(self, key).value = value
            else:
                super().__setattr__(key, value)
        else:
            super().__setattr__(key, value)

    def as_dict(self, mode: Optional[str] = "max-autotune"):
        """Return updated local options and global options in dictionary format"""
        local_options = {}
        global_options = {}
        for k, v in self.__dict__.items():
            if k in self._fixed_attrs:
                if isinstance(v, DeprecatedValue) and v.value is not None:
                    if v.replacement is None:
                        print(f"[warning][npu fx compiler] Option '{k}' is deprecated and will be removed "
                              f"in future version. Please do not configure this option in the future.")
                    else:
                        print(f"[warning][npu fx compiler] Option '{k}' is deprecated and will be removed "
                              f"in future version. Please use '{v.replacement}' instead.")
                    local_options.update({k: v.value})
                elif isinstance(v, OptionValue) and v.value is not None:
                    local_options.update({k: v.value})
                elif isinstance(v, NpuBaseConfig):
                    local_option, global_option = v.as_dict(mode)
                    local_options.update(local_option)
                    global_options.update(global_option)
        return local_options, global_options
