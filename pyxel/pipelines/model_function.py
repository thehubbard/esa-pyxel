#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

"""TBW."""
import functools
import inspect
import typing as t
from collections import abc
from copy import deepcopy

if t.TYPE_CHECKING:
    from pyxel.detectors import Detector


# Declare type variable
T = t.TypeVar("T")


class Arguments(abc.Mapping):
    """TBW."""

    def __init__(self, arguments: t.Optional[dict] = None):
        """TBW."""
        self._arguments = arguments or {}

    def __repr__(self):
        """TBW."""
        return f"Arguments({self._arguments!r})"

    def __contains__(self, key):
        """TBW."""
        return key in self._arguments

    def __getitem__(self, key):
        """TBW."""
        if key not in self:
            raise KeyError(f"Argument '{key}' does not exist.")

        return self._arguments[key]

    def __len__(self):
        """TBW."""
        return len(self._arguments)

    def __iter__(self):
        """TBW."""
        return iter(self._arguments)

    def __getattr__(self, name: str):
        """TBW."""
        if name not in self._arguments:
            raise AttributeError(f"Argument {name!r} does not exist.")

        return self._arguments[name]

    def __dir__(self):
        """TBW."""
        return dir(type(self)) + list(self._arguments)

    def __deepcopy__(self, memo) -> "Arguments":
        """TBW."""
        return Arguments(deepcopy(self._arguments))


# TODO: What is `ModelFunction` ?
#       Is it possible to replace this by a `callable` ?
#       Is it possible to use a function with an inner function (==> a closure) ?
#       could be 'name' and 'enabled' stored in `ModelGroup` ?
class ModelFunction:
    """Create a wrapper function around a Model function.

    Examples
    --------
    >>> from pyxel.models.photon_generation.illumination import illumination
    >>> model_func = ModelFunction(func=illumination, name='illumination',
    ...                            arguments={'level': 1, 'option': 'foo'})

    Access basic parameters
    >>> model_func.name
    'illumination'
    >>> model_func.enabled
    True
    >>> model_func.arguments
    Arguments({'level': 1, 'option': 'foo'})

    Access the arguments with a ``dict`` interface
    >>> list(model_func.arguments)
    ['level', 'option']
    >>> model_func.arguments['level']
    1
    >>> model_func.arguments['level'] = 2
    TypeError: 'Arguments' object does not support item assignment

    Access the arguments with an attribute interface
    >>> model_func.arguments.level
    1
    """

    def __init__(
        self,
        func: t.Callable,
        name: str,
        arguments: t.Optional[dict] = None,
        enabled: bool = True,
    ):
        """TBW.

        Parameters
        ----------
        func
        name
        arguments
        enabled
        """
        assert not inspect.isclass(func)

        self._func = func  # type: t.Callable
        self._name = name
        self.enabled = enabled  # type: bool
        self._arguments = Arguments(arguments)
        # self.group = None               # TODO

    def __repr__(self) -> str:
        """TBW."""
        cls_name = self.__class__.__name__  # type: str
        func_name = self._func.__module__ + "." + self._func.__name__

        return (
            f"{cls_name}(name={self.name!r}, func={func_name}, "
            f"arguments={self.arguments!r}, enabled={self.enabled!r})"
        )

    @property
    def name(self) -> str:
        """TBW."""
        return self._name

    @property
    def arguments(self) -> Arguments:
        """TBW."""
        return self._arguments

    # # TODO: Replace this by __call__ ?
    # @property
    # def function(self) -> t.Callable:
    #     """TBW."""
    #     func_ref = evaluate_reference(self.func)  # type: t.Callable
    #     if isinstance(func_ref, type):
    #         # this is a class type, instantiate it using default arguments.
    #         func_ref = func_ref()
    #         # TODO: should check whether or not it's callable.
    #     func = functools.partial(func_ref, **self.arguments)
    #     return func

    def __call__(self, detector: "Detector") -> T:
        """TBW."""
        # func_ref = evaluate_reference(self.func)  # type: t.Callable

        # if inspect.isclass(self._func):
        #     # this is a class type, instantiate it using default arguments.
        #     func_ref = self._func()
        #     # TODO: should check whether or not it's callable.
        #     raise NotImplementedError
        #
        # else:
        #     func_ref = self._func

        func = functools.partial(self._func, **self.arguments)

        result = func(detector)  # type: T

        return result
