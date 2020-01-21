#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""
import functools
import typing as t

from pyxel.evaluator import evaluate_reference


# TODO: What is `ModelFunction` ?
#       Is it possible to replace this by a `callable` ?
#       Is it possible to use a function with an inner function (==> a closure) ?
#       could be 'name' and 'enabled' stored in `ModelGroup` ?
class ModelFunction:
    """TBW."""

    def __init__(
        self,
        func: t.Union[t.Callable, str],  # TODO: Replace by 'func: t.Callable'
        name: t.Optional[str] = None,
        arguments: t.Optional[dict] = None,
        enabled: bool = True,
    ):
        """TBW.

        :param name:
        :param enabled:
        :param arguments:
        """
        if callable(func):
            func = func.__module__ + "." + func.__name__

        if arguments is None:
            arguments = {}
        self.func = func
        self.name = name
        self.enabled = enabled  # type: bool
        self.arguments = arguments if arguments else {}  # type: dict
        # self.group = None               # TODO

    def __repr__(self) -> str:
        """TBW."""
        return "ModelFunction(%(name)r, %(func)r, %(arguments)r, %(enabled)r)" % vars(
            self
        )

    # TODO: Is this method needed ?
    def __getstate__(self) -> dict:
        """TBW."""
        return {
            "name": self.name,
            "func": self.func,
            "enabled": self.enabled,
            "arguments": self.arguments,
        }

    # TODO: Replace this by __call__ ?
    @property
    def function(self) -> t.Callable:
        """TBW."""
        func_ref = evaluate_reference(self.func)
        if isinstance(func_ref, type):
            # this is a class type, instantiate it using default arguments.
            func_ref = func_ref()
            # TODO: should check whether or not it's callable.
        func = functools.partial(func_ref, **self.arguments)
        return func
