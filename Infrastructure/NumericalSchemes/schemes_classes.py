import numpy as np
from collections import deque
from copy import deepcopy
from Infrastructure.utils import List, Matrix, Vector, Callable
from Infrastructure.Matrices.sparse_matrices import CirculantSparseMatrix, AlmostTridiagonalToeplitzMatrix,\
    IdentityMatrix


def _sum_all_previous_samples(previous_steps_coefficients: List[Matrix],
                              previous_steps_states: List[Matrix]) -> Matrix:
    sum: Matrix = np.zeros_like(previous_steps_states[0])
    for function_component, scheme_coefficient in zip(previous_steps_states, previous_steps_coefficients):
        sum += scheme_coefficient.dot(function_component)
    return sum


class NumericalScheme(object):
    def __init__(self, previous_steps_coefficients, implicit_component, current_state, initial_time, dx, dt, x_samples,
                 non_homogeneous_term, non_homogeneous_scaling_factor, initial_steps_schemes) -> None:
        self._previous_steps_coefficients: List[Matrix] = previous_steps_coefficients
        self.previous_samples_count = len(self._previous_steps_coefficients)
        self._implicit_component: Matrix = implicit_component
        self._current_state: Matrix = current_state
        self._previous_states: List[Matrix] = deque(maxlen=self.previous_samples_count)
        self._previous_states.append(deepcopy(self._current_state))
        self._current_step = 0
        self._current_time: float = initial_time
        self._dx: float = dx
        self._dt: float = dt
        self._x_samples: Vector = x_samples
        self._non_homogeneous_term: Callable = non_homogeneous_term
        self._non_homogeneous_scaling_factor: float = non_homogeneous_scaling_factor
        self._initialize_schemes = initial_steps_schemes

    def make_step(self):
        if self._current_step < self.previous_samples_count:  # If not all previous steps are initialized.
            initializer_scheme = self._initialize_schemes[0]
            if len(self._previous_states) < initializer_scheme.previous_samples_count:
                raise IOError("")
            else:
                needed_previous_states = self._previous_states[0:initializer_scheme.previous_samples_count]
                initializer_scheme._previous_states = needed_previous_states
                initializer_scheme.current_state = self._current_state
                self._current_state = initializer_scheme.make_step()
                self._initialize_schemes.pop()
            self._current_step += 1
        else:
            x_grid, current_t_grid = np.meshgrid(self._x_samples, self._current_time)
            non_homogeneous_element = self._non_homogeneous_term(x_grid, current_t_grid)[0]
            coefficients_sum: Matrix = _sum_all_previous_samples(self._previous_states,
                                                                 self._previous_steps_coefficients)
            coefficients_sum += non_homogeneous_element
            for row_index in len(coefficients_sum):
                function_component = coefficients_sum[row_index]
                self._current_state[row_index] = self._implicit_component.inverse_solution(function_component)
            self._previous_states.pop()

        self._previous_states.append(deepcopy(self._current_state))
        self._current_time += self._dt
        return self._current_state


class SchrodingerEquationForwardEuler(NumericalScheme):
    def __init__(self, current_state, n, initial_time, dx, dt, x_samples, non_homogeneous_term) -> None:
        ratio = dt / (2 * dx)
        transition_mat = CirculantSparseMatrix(n + 1, [1, ratio, -ratio], [0, 1, n])
        previous_coefficients = [transition_mat]
        implicit_component = IdentityMatrix(n)  # This method ix fully explicit.
        initial_steps_schemes = []  # This method is a One-Step method.
        non_homogeneous_scaling_factor = 2
        super(SchrodingerEquationForwardEuler, self).__init__(
            previous_coefficients, implicit_component, current_state, initial_time, dx, dt, x_samples,
            non_homogeneous_term, non_homogeneous_scaling_factor, initial_steps_schemes)
