"""NPU basic configurations"""


class OptionValue:
    """Options for setting npu basic configurations"""

    def __init__(self, default, optional):
        self.__default = default
        self.__optional = optional
        self.__value = default

    def __bool__(self):
        return bool(self.__value)

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
        if self.__value is None:
            return None
        if str(self.__value) == str(True):
            return "1"
        if str(self.__value) == str(False):
            return "0"
        return str(self.__value)

    @value.setter
    def value(self, v):
        if isinstance(self.__optional, (tuple, list,)) and v not in self.__optional:
            raise ValueError(
                "'" + str(v) + "' not in optional list " + str(self.__optional))
        self.__value = v


class DeprecatedValue(OptionValue):
    def __init__(self, optional, *, replacement):
        super().__init__(None, optional)
        self.replacement = replacement


class NpuBaseConfig:
    """NPU basic configurations"""

    def __init__(self):
        self._fixed_attrs = []
        for k, v in self.__dict__.items():
            if isinstance(v, (OptionValue, NpuBaseConfig)):
                self._fixed_attrs.append(k)

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

    def as_dict(self):
        """Return updated option in dictionary format"""
        options = {}
        for k, v in self.__dict__.items():
            if k in self._fixed_attrs:
                if isinstance(v, DeprecatedValue) and v.value is not None:
                    if v.replacement is None:
                        print(f"[warning][npu fx compiler] Option '{k}' is deprecated and will be removed "
                              f"in future version. Please do not configure this option in the future.")
                    else:
                        print(f"[warning][npu fx compiler] Option '{k}' is deprecated and will be removed "
                              f"in future version. Please use '{v.replacement}' instead.")
                    options.update({k: v.value})
                elif isinstance(v, OptionValue) and v.value is not None:
                    options.update({k: v.value})
                elif isinstance(v, NpuBaseConfig):
                    options.update(v.as_dict())
        return options
