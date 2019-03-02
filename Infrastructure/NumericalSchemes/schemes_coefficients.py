from Infrastructure.utils import Matrix
from scipy import sparse, linalg
import numpy as np


class SchemeCoefficient(object):
    def __init__(self, inner_row_coefficient: Matrix, between_rows_coefficient: Matrix) -> None:
        self._inner_row_coefficient: Matrix = inner_row_coefficient
        self._between_rows_coefficient: Matrix = between_rows_coefficient

    def dot(self, mat: Matrix) -> Matrix:
        return self._between_rows_coefficient.dot(self._inner_row_coefficient.dot(mat))


class FreeCoefficient(SchemeCoefficient):
    def __init__(self, between_rows_coefficient: Matrix) -> None:
        super(FreeCoefficient, self).__init__(inner_row_coefficient=0,
                                              between_rows_coefficient=between_rows_coefficient)

    def dot(self, mat: Matrix) -> Matrix:
        return self._between_rows_coefficient.dot(mat)


class ImplicitCoefficient(SchemeCoefficient):
    def __init__(self, n: int, inner_row_coefficient: Matrix, between_rows_coefficient: Matrix) -> None:
        super(ImplicitCoefficient, self).__init__(inner_row_coefficient=inner_row_coefficient,
                                                  between_rows_coefficient=between_rows_coefficient)
        total_reshaped_left_hand_matrix = sparse.kron(between_rows_coefficient, inner_row_coefficient.toarray()).toarray()
        total_reshaped_left_hand_matrix += sparse.identity(total_reshaped_left_hand_matrix.shape[0])
        self._lu_factors = linalg.lu_factor(total_reshaped_left_hand_matrix, overwrite_a=True, check_finite=False)

    def inverse_solution(self, current_state: Matrix) -> Matrix:
        original_shape = current_state.shape
        current_state = np.ravel(current_state)
        current_state = linalg.lu_solve(self._lu_factors, current_state, overwrite_b=True, check_finite=False)
        next_state = current_state.reshape(original_shape)
        return next_state

