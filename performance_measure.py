from typing import Callable, Tuple
from timeit import timeit


def time_measure(func: Callable, func_input: Tuple, number_of_runs: int) -> float:
    total_time_in_seconds = timeit("func(*input)", number=number_of_runs, globals={'func': func, 'input': func_input})
    total_in_microns = total_time_in_seconds * 1e+6
    return total_in_microns / number_of_runs
