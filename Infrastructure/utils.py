import numpy as np
import numba
from typing import List, Callable, Tuple, ClassVar, Iterable, Mapping, Union


# Naming data types for type hinting.
Scalar = Union[float, int]
Vector = Union[List[Scalar], Scalar]
Matrix = Union[List[Vector], Vector, Scalar]
StaticField = ClassVar


def jit(signature_or_function=None, target: str = "cpu", parallel: bool = False) -> Callable:
    """
    A wrapper for numba.jit decorator. Used for accelerating functions.

    :param signature_or_function: Either a signature for numba's jit or a function for decorating.
    :param target: "cpu", "cuda" or "gpu"
    :param parallel: True or False, for distributing the function's computations over cores.
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
                        cache=True, nogil=True)
    if func_to_decorate is not None:
        return wrapper(func_to_decorate)
    else:
        return wrapper
