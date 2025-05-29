import warnings
from functools import wraps
from typing import List


def deprecated_arg_warning(arg_name: str, arg_value: str, new_arg_name: str):
    warnings.warn(
        f"{arg_name} is deprecated and will be removed in the "
        f"future. Please use {new_arg_name} instead. "
        f"Received: {arg_value}",
        DeprecationWarning,
    )


def check_deprecated_args(deprecated_args: List[str], new_args: List[str]):
    """Decorator to check for deprecated arguments.

    Args:
        deprecated_args (List[str]): List of deprecated arguments.
        new_args (List[str]): List of new arguments.
    """
    def _wrapper(func):
        @wraps(func)
        def _inner(*args, **kwargs):
            for deprecated_arg, new_arg in zip(deprecated_args, new_args):
                if deprecated_arg in kwargs:
                    deprecated_arg_warning(
                        deprecated_arg, kwargs[deprecated_arg], new_arg
                    )
                    kwargs[new_arg] = kwargs[deprecated_arg]
                    del kwargs[deprecated_arg]
            return func(*args, **kwargs)
        return _inner
    return _wrapper
