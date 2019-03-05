import numpy as np
from collections import deque
from copy import deepcopy
from Infrastructure.Matrices.sparse_matrices import CirculantSparseMatrix, AlmostTridiagonalToeplitzMatrix,\
    IdentityMatrix
from Infrastructure.NumericalSchemes.schemes_coefficients import FreeCoefficient, SchemeCoefficient, ImplicitCoefficient
from Infrastructure.utils import List, Matrix, Vector, Callable, jit


def _sum_all_previous_samples(previous_steps_states: List[Matrix],
                              previous_steps_coefficients: List[Matrix]) -> Matrix:
    sum: Matrix = np.zeros_like(previous_steps_states[0])
    for function_component, scheme_coefficient in zip(previous_steps_states, previous_steps_coefficients):
        sum += scheme_coefficient.dot(function_component)
    return sum


class NumericalScheme(object):
    def __init__(self, previous_steps_coefficients, implicit_component, free_coefficient, current_state,
                 initial_time, dx, dt, x_samples, non_homogeneous_term, non_homogeneous_scaling_factor,
                 initial_steps_schemes) -> None:
        self._previous_steps_coefficients: List[Matrix] = previous_steps_coefficients
        self.previous_samples_count = len(self._previous_steps_coefficients)
        self._implicit_component: Matrix = implicit_component
        self.current_state: Matrix = current_state
        self._previous_states: List[Matrix] = deque(maxlen=self.previous_samples_count)
        self._previous_states.append(deepcopy(self.current_state))
        self._current_step = 0
        self._current_time: float = initial_time
        self._dx: float = dx
        self._dt: float = dt
        self._x_samples: Vector = x_samples
        self._non_homogeneous_term: Callable = non_homogeneous_term
        self._non_homogeneous_scaling_factor: float = non_homogeneous_scaling_factor
        self._initialize_schemes = initial_steps_schemes
        self._free_coefficient = free_coefficient

    def make_step(self):
        if self._current_step < self.previous_samples_count - 1:  # If not all previous steps are initialized.
            initializer_scheme = self._initialize_schemes[0]
            if len(self._previous_states) < initializer_scheme.previous_samples_count:
                raise IOError("")
            else:
                needed_previous_states = self._previous_states[0:initializer_scheme.previous_samples_count]
                initializer_scheme._previous_states = needed_previous_states
                initializer_scheme.current_state = self.current_state
                self.current_state = initializer_scheme.make_step()
                self._initialize_schemes.pop()
            self._current_step += 1
        else:
            x_grid, current_t_grid = np.meshgrid(self._x_samples, self._current_time)
            non_homogeneous_element = self._non_homogeneous_term(x_grid, current_t_grid)[0]
            coefficients_sum: Matrix = _sum_all_previous_samples(self._previous_states,
                                                                 self._previous_steps_coefficients)
            coefficients_sum += self._dt * self._free_coefficient.dot(self.current_state)
            coefficients_sum += self.current_state
            # Adding the non-homogeneous effect to our scheme.
            coefficients_sum += self._dt * self._non_homogeneous_scaling_factor * non_homogeneous_element
            self.current_state = self._implicit_component.inverse_solution(coefficients_sum)
            self._previous_states.pop()

        self._previous_states.appendleft(deepcopy(self.current_state))
        self._current_time += self._dt
        return self.current_state


class SchrodingerEquationForwardEuler(NumericalScheme):
    def __init__(self, current_state, n, initial_time, dx, dt, x_samples, A, C, non_homogeneous_term) -> None:
        # Creating the only coefficient matrix.
        ratio = dt / dx ** 2
        transition_mat = CirculantSparseMatrix(n + 1, [-2 * ratio, ratio, ratio], [0, 1, n])
        for k in range(len(C)):
            C[k, k] += 1
        free_coefficient = FreeCoefficient(C)
        previous_coefficients = [SchemeCoefficient(between_rows_coefficient=A, inner_row_coefficient=transition_mat)]
        implicit_component = IdentityMatrix(n)  # This method is fully explicit.
        initial_steps_schemes = []  # This method is a One-Step method.
        non_homogeneous_scaling_factor = 1
        super(SchrodingerEquationForwardEuler, self).__init__(
            previous_coefficients, implicit_component, free_coefficient, current_state, initial_time, dx, dt, x_samples,
            non_homogeneous_term, non_homogeneous_scaling_factor, initial_steps_schemes)


class SchrodingerEquationBackwardEuler(NumericalScheme):
    def __init__(self, current_state, n, initial_time, dx, dt, x_samples, A, C, non_homogeneous_term) -> None:
        # Creating the only coefficient matrix.
        ratio = dt / dx ** 2
        transition_mat = CirculantSparseMatrix(n + 1, [2*ratio, -ratio, -ratio], [0, 1, n])
        previous_coefficients = [np.zeros_like(A)]  # This method is fully implicit.
        implicit_component = ImplicitCoefficient(n, between_rows_coefficient=A, inner_row_coefficient=transition_mat)
        free_coefficient = FreeCoefficient(C)
        initial_steps_schemes = []  # This method is a One-Step method.
        non_homogeneous_scaling_factor = 1
        super(SchrodingerEquationBackwardEuler, self).__init__(
            previous_coefficients, implicit_component, free_coefficient, current_state, initial_time, dx, dt, x_samples,
            non_homogeneous_term, non_homogeneous_scaling_factor, initial_steps_schemes)


class SchrodingerEquationCrankNicholson(NumericalScheme):
    def __init__(self, current_state, n, initial_time, dx, dt, x_samples, A, C, non_homogeneous_term) -> None:
        # Creating the only coefficient matrix.
        ratio = dt / (2 * dx ** 2)
        explicit_mat = CirculantSparseMatrix(n + 1, [-2 * ratio, ratio, ratio], [0, 1, n])
        implicit_mat = CirculantSparseMatrix(n + 1, [2 * ratio, -ratio, -ratio], [0, 1, n])
        previous_coefficients = [SchemeCoefficient(between_rows_coefficient=A, inner_row_coefficient=explicit_mat)]
        implicit_component = ImplicitCoefficient(n, between_rows_coefficient=A, inner_row_coefficient=implicit_mat)
        free_coefficient = FreeCoefficient(C)
        initial_steps_schemes = []  # This method is a One-Step method.
        non_homogeneous_scaling_factor = 1
        super(SchrodingerEquationCrankNicholson, self).__init__(
            previous_coefficients, implicit_component, free_coefficient, current_state, initial_time, dx, dt, x_samples,
            non_homogeneous_term, non_homogeneous_scaling_factor, initial_steps_schemes)
