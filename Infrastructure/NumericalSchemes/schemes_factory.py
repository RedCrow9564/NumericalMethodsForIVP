
from Infrastructure.utils import BaseEnum
from .schemes_classes import *


class ModelName(BaseEnum):
    SchrodingerEquation_ForwardEuler = "Schrodinger-like Equation - Forward Euler"
    SchrodingerEquation_LeapFrog = "Schrodinger-like Equation - Leap Frog"
    SchrodingerEquation_BackwardEuler = "Schrodinger-like Equation - Backward Euler"
    SchrodingerEquation_CrankNicholson = "Schrodinger-like Equation - Crank Nicholson"
    SchrodingerEquation_DuFortFrankel = "Schrodinger-like Equation - DuFort Frankel"


# TODO: Replace None with the matching schemes classes, once they are implemented.
_models_names_to_objects = {
    ModelName.SchrodingerEquation_ForwardEuler: SchrodingerEquationForwardEuler,
    ModelName.SchrodingerEquation_LeapFrog: SchrodingerEquationLeapFrog,
    ModelName.SchrodingerEquation_BackwardEuler: SchrodingerEquationBackwardEuler,
    ModelName.SchrodingerEquation_CrankNicholson: SchrodingerEquationCrankNicholson,
    ModelName.SchrodingerEquation_DuFortFrankel: None
}


def create_model(model_name):
    if model_name in _models_names_to_objects:
        return _models_names_to_objects[model_name]
    else:
        raise NotImplementedError("Model name {0} is NOT implemented".format(model_name))





