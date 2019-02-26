
class NumericalScheme(object):
    def __init__(self, previous_steps_coefficients, implicit_component, current_state, initial_time, dx, dt, x_samples,
                 non_homogeneous_term, non_homogeneous_scaling_factor):
        self._previous_steps_coefficients = previous_steps_coefficients
        self._implicit_component = implicit_component
        self._current_state = current_state
        self._current_time = initial_time
        self._dx = dx
        self._dt = dt
        self._x_samples = x_samples
        self._non_homogeneous_term = non_homogeneous_term
        self._non_homogeneous_scaling_factor = non_homogeneous_scaling_factor

    def make_step(self):
        # TODO: Implement.
        pass
