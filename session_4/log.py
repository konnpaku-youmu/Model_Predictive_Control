from dataclasses import dataclass
from rcracers.simulator.core import BaseControllerLog, list_field
from dataclasses import dataclass, field

def new_list():
    return field(default_factory=list)

@dataclass
class ControllerLog(BaseControllerLog):
    solver_success: list = new_list()
    state_prediction: list = new_list()
    input_prediction: list = new_list()
    solver_time: list = list_field()