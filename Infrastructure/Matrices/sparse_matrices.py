import numpy as np
from numba import jitclass
from ..utils import range_wrap, jit
from .circulant_sparse_product import compute


class CirculantSparseMatrix(object):

    def __init__(self, n, nonzero_terms, nonzero_indices):
        self._n = n
        self._terms = np.array(nonzero_terms, dtype=complex)
        self._indices = np.array(nonzero_indices, dtype=np.int32)

    def dot(self, current_state):
        next_state = np.empty_like(current_state, dtype=complex)
        for row_index, component_current_state in enumerate(current_state):
            compute(self._terms, self._indices, component_current_state, next_state[row_index, :])
        return next_state
