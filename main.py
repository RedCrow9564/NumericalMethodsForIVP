import numpy as np
from Infrastructure.utils import Matrix, Scalar, jit
from performance_measure import time_measure
from Infrastructure.Matrices.sparse_matrices import CirculantSparseMatrix


@jit
def exact_solution(x: Matrix, t: Scalar) -> Matrix:
    """
    The exact solution of the investigated PDE.
    :param x: The sampled space coordinates vector.
    :param t: A specific sampled time value.
    :return: The sampled solution's values as a matrix. Each component of the solution is spread
     over a row (and not a column)
    """
    pi_factor: float = 2 * np.pi
    factor: Matrix = pi_factor * x + (pi_factor ** 2 + 1) * t
    time_coeff: complex = np.e ** (3j * t)
    value: Matrix = time_coeff * np.vstack((np.cos(factor), -1j * np.sin(factor)))
    return value


def non_homogeneous_term(x: Matrix, t: Scalar) -> Matrix:
    """
    The non-homogeneous term in the investigated PDE. Since out PDE is homogeneous, this will be a constant zero.
    :param x: The sampled space coordinates vector.
    :param t: A specific sampled time value.
    :return: A scalar integer zero.
    """
    return 0


if __name__ == "__main__":
    n = 15
    x = np.linspace(0, 2 * np.pi, n)
    a = exact_solution(x, 0)
    b = CirculantSparseMatrix(n=n, nonzero_terms=[1, 2, 3], nonzero_indices=[0, 1, 2])
    print(b.dot(a))

    def f():
        b.dot(a)
    print(time_measure(f, (), 1000))
