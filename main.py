import numpy as np
from Infrastructure.utils import Matrix, Scalar, jit, Consts
from performance_measure import time_measure
from Infrastructure.Matrices.sparse_matrices import CirculantSparseMatrix, AlmostTridiagonalToeplitzMatrix
from Infrastructure.NumericalSchemes.schemes_factory import ModelName
from Infrastructure.Experiments.experiments import SingleLambdaManyNExperiments
from Infrastructure.Experiments.dt_initializer import DtInitializerMethod

PI: float = Consts.PI
E: float = Consts.E


def test_non_homogeneous_element(x: Matrix, t: Scalar) -> Matrix:
    result = 4 * PI ** 2 * np.cos(2 * PI * (x + t).reshape((1, -1)), dtype=complex)
    result -= 2 * PI * np.sin(2 * PI * (x + t).reshape((1, -1)), dtype=complex)
    return np.vstack((result, result))


def test_solution(x: Matrix, t: Scalar) -> Matrix:
    single_function_samples = np.cos(2 * PI * (x + t).reshape((1, -1)), dtype=complex)
    return np.vstack((single_function_samples, single_function_samples))


@jit
def exact_solution(x: Matrix, t: Scalar) -> Matrix:
    """
    The exact solution of the investigated PDE.
    :param x: The sampled space coordinates vector.
    :param t: A specific sampled time value.
    :return: The sampled solution's values as a matrix. Each component of the solution is spread
     over a row (and not a column)
    """
    pi_factor: float = 2 * PI
    factor: Matrix = pi_factor * x + (pi_factor ** 2 + 1) * t
    time_coeff: complex = E ** (3j * t)
    value: Matrix = time_coeff * np.vstack((np.cos(factor), -1j * np.sin(factor)))
    return value


def non_homogeneous_term(x: Matrix, t: Scalar) -> Matrix:
    """
    The non-homogeneous term in the investigated PDE. Since out PDE is homogeneous, this will be a constant zero.
    :param x: The sampled space coordinates vector.
    :param t: A specific sampled time value.
    :return: A scalar integer zero.
    """
    return np.zeros_like(x)


if __name__ == "__main__":
    n = 8
    n_list = np.power(2, list(range(4, n)))
    #x = np.linspace(0, 2 * Consts.PI, n + 1)
    #a = np.ones((1, n + 1), dtype=complex)  # exact_solution(x, 0)
    #a[0, :] *= 2
    #b = CirculantSparseMatrix(n=n + 1, nonzero_terms=[1, 2, 3], nonzero_indices=[0, 1, 2])
    #d = AlmostTridiagonalToeplitzMatrix(n + 1, [2, 2, 2])
    #print(b.dot(a))
    #print(d.inverse_solution(a))

    #def f():
    #    b.dot(a)
    # print(time_measure(f, (), 1000))

    first_t = 0
    last_t = 1
    first_x = 0
    last_x = 1
    lamda = 0.5

    A = 1j * np.array([[0, 1], [1, 0]])  # np.array([[1]])
    C = 1j * np.array(([[3, -1], [-1, 3]]))  # np.array([[0]])

    e = SingleLambdaManyNExperiments(ModelName.SchrodingerEquation_BackwardEuler, n_list, lamda, first_t, last_t,
                                     first_x, last_x, DtInitializerMethod.square, exact_solution, non_homogeneous_term,
                                     A, C)

    e.run_experiments()
    e.plot_results()
