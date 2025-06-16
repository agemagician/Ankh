import warnings
from functools import wraps
from typing import List


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
                    warnings.warn(
                        f"{deprecated_arg} is deprecated and will be removed "
                        f"in the future. Please use `{new_arg}` instead. "
                        f"Received: {kwargs[deprecated_arg]}",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    kwargs[new_arg] = kwargs[deprecated_arg]
                    del kwargs[deprecated_arg]
            return func(*args, **kwargs)

        return _inner

    return _wrapper
