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

if t.TYPE_CHECKING:
    from pyxel.detectors import Detector


# Declare type variable
T = t.TypeVar("T")


# TODO: Improve this class. See issue #133.
class Arguments(dict):
    """TBW."""

    def __getattr__(self, name: str) -> t.Union[int, float]:
        if name not in self:
            raise AttributeError(f"Argument {name!r} does not exist.")

        result = self[name]
        assert isinstance(result, int) or isinstance(result, float)

        return result

    def __dir__(self):
        return dir(type(self)) + list(self)

    # def __deepcopy__(self, memo) -> "Arguments":
    #     """TBW."""
    #     return Arguments(deepcopy(self._arguments))


# TODO: Improve this class. See issue #132.
class ModelFunction:
    """Create a wrapper function around a Model function.

    Examples
    --------
    >>> from pyxel.models.photon_generation.illumination import illumination
    >>> model_func = ModelFunction(
    ...     func=illumination,
    ...     name="illumination",
    ...     arguments={"level": 1, "option": "foo"},
    ... )

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
    >>> model_func.arguments["level"]
    1
    >>> model_func.arguments["level"] = 2
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
        assert not inspect.isclass(func)

        self._func = func  # type: t.Callable
        self._name = name
        self.enabled = enabled  # type: bool

        if arguments is None:
            arguments = {}

        self._arguments = Arguments(arguments)
        # self.group = None               # TODO

    def __repr__(self) -> str:
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
    def arguments(self) -> dict:
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
