import eagerx
import numpy as np
from typing import Dict
from eagerx.wrappers import Flatten
class Crazyfly_Env(eagerx.BaseEnv):
    def __init__(self, name: str, rate: float, graph: eagerx.Graph, engine: eagerx.specs.EngineSpec, eval: bool):
        """Initializes an environment with EAGERx dynamics.

        :param name: The name of the environment. Everything related to this environment
                     (parameters, topics, nodes, etc...) will be registered under namespace: "/[name]".
        :param rate: The rate (Hz) at which the environment will run.
        :param graph: The graph consisting of nodes and objects that describe the environment's dynamics.
        :param engine: The physics engine that will govern the environment's dynamics.
        :param eval: If True we will create an evaluation environment, i.e. not performing domain randomization.
        """
        # Make the backend specification
        from eagerx.backends.single_process import SingleProcess
        backend = SingleProcess.make()

        self.eval = eval

        # Maximum episode length
        self.max_steps = rate*8 if eval else rate*8 # todo

        # Step counter
        self.steps = None
        super().__init__(name, rate, graph, engine, backend, force_start=True)

    def step(self, action: Dict):
        observation = self._step(action)
        self.steps += 1
        cost = 0
        done = self.steps > self.max_steps
        info = {"TimeLimit.truncated": self.steps > self.max_steps}

        return observation, -cost, done, info
    def reset(self) -> Dict:
        states = self.state_space.sample()
        states["crazyflie/model_state"][:] = [0., 0., 1., 0., 0., 0., 0., 0., 0.]
        observation = self._reset(states)
        # Reset step counter
        self.steps = 0
        return observation