import eagerx
import numpy as np
from typing import Dict
from eagerx.wrappers import Flatten
class Double_PendulumEnv(eagerx.BaseEnv):
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
        self.max_steps = 270 if eval else 100

        # Step counter
        self.steps = None
        super().__init__(name, rate, graph, engine, backend, force_start=True)

    def step(self, action: Dict):
        """A method that runs one timestep of the environment's dynamics.

        :params action: A dictionary of actions provided by the agent.
        :returns: A tuple (observation, reward, done, info).

            - observation: Dictionary of observations of the current timestep.

            - reward: amount of reward returned after previous action

            - done: whether the episode has ended, in which case further step() calls will return undefined results

            - info: contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        # Take step
        observation = self._step(action)
        self.steps += 1

        # Extract observations
        cos_th1, sin_th1,cos_th2,sin_th2 = observation["angle"][0]
        thdot1,thdot2 = observation["angular_velocity"][0]
        u = action["voltage"][0]

        # Calculate reward
        # We want to penalize the angle error, angular velocity and applied voltage
        th = np.arctan2(sin_th1, cos_th1)
        cost = th ** 2 + 0.1 * (thdot1 / (1 + 10 * abs(th))) ** 2 + 0.01 * u ** 2

        # Determine done flag
        done = self.steps > self.max_steps

        # Set info:
        info = {"TimeLimit.truncated": self.steps > self.max_steps}

        return observation, -cost, done, info

    def reset(self) -> Dict:
        """Resets the environment to an initial state and returns an initial observation.
        :returns: The initial observation.
        """
        # Determine reset states
        states = self.state_space.sample()
        states["double_pendulum/model_state"][:] = [0.0, 0.0,0.0, 0.0]
        # todo:this problem!
        observation = self._reset(states)
        # Reset step counter
        self.steps = 0
        return observation