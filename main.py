import numpy as np
from Infrastructure.utils import Matrix, Scalar, jit
from performance_measure import time_measure


@jit
def exact_solution(x: Matrix, t: Scalar) -> Matrix:
    """
    The exact solution of the investigated PDE.
    :param x: The sampled space coordinates vector.
    :param t: A specific sampled time value.
    :return: The sampled solution's values as a matrix. Each component of the solution is spread
     over a row (and not a column)
    """
    pi_factor = 2 * np.pi
    factor = pi_factor * x + (pi_factor ** 2 + 1) * t
    time_coeff = np.e ** (3j * t)
    value = time_coeff * np.array([np.cos(factor), -1j * np.sin(factor)])
    return value.astype(np.complex64)


def non_homogeneous_term(x: Matrix, t: Scalar) -> Matrix:
    """
    The non-homogeneous term in the investigated PDE. Since out PDE is homogeneous, this will be a constant zero.
    :param x: The sampled space coordinates vector.
    :param t: A specific sampled time value.
    :return: A scalar integer zero.
    """
    return 0


if __name__ == "__main__":
    a = exact_solution(0, 0)
    print(a)
    print(time_measure(exact_solution, (0.0, 0.0), 10000))
