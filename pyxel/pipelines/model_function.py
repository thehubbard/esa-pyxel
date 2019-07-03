"""TBW."""
import functools
import typing as t
from esapy_config import evaluate_reference


# FRED: What is `ModelFunction` ?
#       Is it possible to replace this by a `callable` ?
#       Is it possible to use a function with an inner function (==> a closure) ?
#       could be 'name' and 'enabled' stored in `ModelGroup` ?
class ModelFunction:
    """TBW."""

    def __init__(self,
                 func: t.Union[t.Callable, str],  # callable or str
                 name: t.Optional[str] = None,
                 arguments: t.Optional[dict] = None,
                 enabled: t.Optional[bool] = True):
        """TBW.

        :param name:
        :param enabled:
        :param arguments:
        """
        if callable(func):
            func = func.__module__ + '.' + func.__name__

        if arguments is None:
            arguments = {}
        self.func = func
        self.name = name
        self.enabled = enabled
        self.arguments = arguments
        # self.group = None               # TODO

    def __repr__(self) -> str:
        """TBW."""
        return 'ModelFunction(%(name)r, %(func)r, %(arguments)r, %(enabled)r)' % vars(self)

    # FRED: Is it needed ?  Where is the '__setstate__' ?
    def __getstate__(self) -> dict:
        """TBW."""
        return {
            'name': self.name,
            'func': self.func,
            'enabled': self.enabled,
            'arguments': self.arguments
        }

    # FRED: Replace this by __call__ ?
    @property
    def function(self) -> t.Callable:
        """TBW."""
        func_ref = evaluate_reference(self.func)
        if isinstance(func_ref, type):
            # this is a class type, instantiate it using default arguments.
            func_ref = func_ref()
            # HANS: should check whether or not it's callable.
        func = functools.partial(func_ref, **self.arguments)
        return func
