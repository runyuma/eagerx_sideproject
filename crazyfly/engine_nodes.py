from typing import Optional, List
import numpy as np

# IMPORT EAGERX
from eagerx.core.space import Space
from eagerx.core.constants import process
from eagerx.utils.utils import Msg
from eagerx.core.entities import EngineNode
import eagerx.core.register as register


class FloatMultiArrayOutput(EngineNode):
    @staticmethod
    def make(
            cls,
            name: str,
            rate: float,
            idx: Optional[list] = [0],
            process: Optional[int] = process.ENVIRONMENT,
            color: Optional[str] = "cyan",
    ):
        """
        FloatOutput spec
        :param idx: index of the value of interest from the array.
        """
        # Performs all the steps to fill-in the params with registered info about all functions.
        spec = cls.get_specification()
        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.inputs = ["observation_array"]
        spec.config.outputs = ["observation"]

        # Custom node params
        spec.config.idx = idx
        return spec

    def initialize(self, spec, simulator):
        self.idx = spec.config.idx
    @register.states()
    def reset(self):
        pass

    @register.inputs(observation_array=Space(dtype="float32"))
    @register.outputs(observation=Space(dtype="float32"))
    def callback(self, t_n: float, observation_array: Optional[Msg] = None):
        data = len(self.idx) * [0]
        for idx, _data in enumerate(self.idx):
            data[idx] = observation_array.msgs[-1].data[_data]
        # if statement to add yaw since Jacob didn't use that
        if self.idx[0] == 6 and self.idx[1] == 7:
            data.append(0)
        return dict(observation=np.array(data,dtype="float32"))

class OdeMultiInput(EngineNode):
    @classmethod
    def make(
            cls,
            name: str,
            rate: float,
            default_action: List,
            process: Optional[int] = process.ENGINE,
            color: Optional[str] = "green",
    ):
        """OdeInput spec"""
        spec = cls.get_specification()

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.inputs = ["tick", "commanded_thrust", "commanded_attitude"]
        spec.config.outputs = ["action_applied"]

        # Set custom node params
        spec.config.default_action = default_action
        return spec

    def initialize(self, spec, simulator):
        # We will probably use self.simulator in callback & reset.
        assert (
                spec.config.process == process.ENGINE
        ), "Simulation node requires a reference to the simulator, hence it must be launched in the Engine process"
        self.simulator = simulator
        self.default_action = np.array(spec.config.default_action)

    @register.states()
    def reset(self):
        self.simulator["input"] = np.squeeze(np.array(self.default_action))

    @register.inputs(tick=Space(shape=(), dtype="int64"), action=Space(dtype="float32"))
    @register.outputs(action_applied=Space(dtype="float32"))
    def callback(self, t_n: float,
                 commanded_thrust: Optional[Msg] = None,
                 commanded_attitude: Optional[Msg] = None,):
        assert isinstance(self.simulator, dict), (
                'Simulator object "%s" is not compatible with this simulation node.' % self.simulator
        )
        u = [np.squeeze(commanded_thrust.msgs[-1].data), np.squeeze(commanded_attitude.msgs[-1].data[0]),
             np.squeeze(commanded_attitude.msgs[-1].data[1])]
        action_applied = [commanded_thrust.msgs[-1], commanded_attitude.msgs[-1]]
        # Set action in simulator for next step.
        self.simulator["input"] = u

        # Send action that has been applied.
        return dict(action_applied=action_applied.msgs[-1])