import numpy as np
from .circulant_sparse_product import compute, solve_almost_tridiagonal_system


class CirculantSparseMatrix(object):

    def __init__(self, n, nonzero_terms, nonzero_indices):
        self._n = n
        self._terms = np.array(nonzero_terms, dtype=complex)
        self._indices = np.array(nonzero_indices, dtype=np.int32)

    def dot(self, current_state):
        next_state = np.empty_like(current_state, dtype=complex)
        temp_vector = np.zeros_like(next_state[0, :])
        for row_index, component_current_state in enumerate(current_state):
            compute(self._terms, self._indices, component_current_state, temp_vector)
            next_state[row_index, :] = temp_vector
        return next_state


class AlmostTridiagonalToeplitzMatrix(CirculantSparseMatrix):
    def __init__(self, n, nonzero_terms):
        super(AlmostTridiagonalToeplitzMatrix, self).__init__(n, nonzero_terms, nonzero_indices=[0, 1, n])
        self._diag_term = nonzero_terms[0]
        self._up_diag = nonzero_terms[1]
        self._sub_diag = nonzero_terms[2]

    def inverse_solution(self, current_state):
        empty_vector = np.empty_like(current_state[0, :])
        for row_index, component_current_state in enumerate(current_state):
            solve_almost_tridiagonal_system(self._diag_term, self._sub_diag, self._up_diag, self._n,
                                            current_state[row_index, :], empty_vector)
        return current_state


class IdentityMatrix(object):
    def __init__(self, n):
        self._n = n

    def dot(self, current_state):
        return current_state

    def inverse_solution(self, current_state):
        return current_state
