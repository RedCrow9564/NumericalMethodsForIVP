import numpy as np
import numba
from typing import List, Callable, Tuple, ClassVar, Iterable, Mapping, Union
import inspect
from enum import Enum


# Naming data types for type hinting.
Scalar = Union[complex, float, int]
Vector = Union[List[Scalar], np.ndarray, Scalar]
Matrix = Union[List[Vector], Vector, Scalar]
StaticField = ClassVar


class _MetaEnum(type):
    def __iter__(self):
        # noinspection PyUnresolvedReferences
        return self.enum_iter()

    def __contains__(self, item):
        # noinspection PyUnresolvedReferences
        return self.enum_contains(item)


class BaseEnum(Enum, metaclass=_MetaEnum):

    @classmethod
    def enum_iter(cls):
        return iter(cls.get_all_values())

    @classmethod
    def enum_contains(cls, item):
        return item in cls.get_all_values()

    @classmethod
    def get_all_values(cls):
        all_attributes = inspect.getmembers(cls, lambda a: not inspect.ismethod(a))
        all_attributes = [value for name, value in all_attributes if not (name.startswith('__') or name.endswith('__'))]
        return all_attributes


class Consts(BaseEnum):
    PI = np.pi
    E = np.e


def jit(signature_or_function=None, target: str = "cpu", parallel: bool = False) -> Callable:
    """
    A wrapper for numba.jit decorator. Used for accelerating functions.

    :param signature_or_function: Either a signature for numba's jit or a function for decorating.
    :param target: "cpu", "cuda" or "gpu"
    :param parallel: True or False, for distributing the function's computations over cores.
        For target other than "cpu" this MUST be False.
    :return: A decorated function.
    """

    # Handle signature
    if signature_or_function is None:
        # No signature, no function
        func_to_decorate = None
        func_signatures = None
    elif isinstance(signature_or_function, list):
        # A list of signatures is passed
        func_to_decorate = None
        func_signatures = signature_or_function
    elif numba.sigutils.is_signature(signature_or_function):
        # A single signature is passed
        func_to_decorate = None
        func_signatures = [signature_or_function]
    else:
        # A function is passed
        func_to_decorate = signature_or_function
        func_signatures = None

    wrapper = numba.jit(signature_or_function=func_signatures, nopython=True, target=target, parallel=parallel,
                        cache=True, nogil=True, fastmath=True)
    if func_to_decorate is not None:
        return wrapper(func_to_decorate)
    else:
        return wrapper


def range_wrap(start:int = 0, stop:int = 0, step:int = 1, parallel: bool = False) -> range:
    if parallel:
        return numba.prange(start, stop, step)
    else:
        return numba.range_iter32_type(start, stop, step)
