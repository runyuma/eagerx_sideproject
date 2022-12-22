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
        self.max_steps = 270 if eval else 200

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
        angle = np.array(observation["angle"]).reshape((4,1))
        vel = np.array(observation["angular_velocity"][0])
        u = action["voltage"][0]

        # Calculate reward
        # We want to penalize the angle error, angular velocity and applied voltage
        target_angle = np.array([-1,0,1,0]).reshape((4,1))
        target_vel = np.array([-0,0])
        theta1 = np.arctan2(observation["angle"][0][1], observation["angle"][0][0])
        theta2 = np.arctan2(observation["angle"][0][3], observation["angle"][0][2])
        # reward_factor = np.exp(abs(theta1))
        # D1 = np.diag([np.exp(np.pi)/reward_factor*1,np.exp(np.pi)/reward_factor*1,2,2])
        D2 = np.diag([0.05,0.25])
        pos = np.array([angle[0][0]+np.cos(theta1+theta2),angle[1][0]+np.sin(theta1+theta2)])
        target_pos = np.array([-2,0])
        Dp = np.diag([10, 5])
        cost = 0
        PRINT = False
        if ((pos - target_pos).T @ (pos - target_pos)) < 0.25:
            cost -= 100
            D2 = np.diag([1, 5])
            PRINT = True
            if ((pos - target_pos).T @ (pos - target_pos)) < 0.1:
                cost -= 100
                if abs(vel[1])<= 0.3:
                    cost -= 100*np.exp(8 * (0.3-abs(vel[1])))

        cost += (pos - target_pos).T @ Dp @ (pos - target_pos) + (vel - target_vel).T@D2@(vel - target_vel) + 0.001 * u ** 2
        # avoid local optima [pi,pi,0,0]
        if abs(theta1) > np.pi / 2:
            cost -= (abs(theta1) - np.pi / 2) * 20 * (np.pi / 2 - abs(theta2))
        if PRINT:
            print("reward:", -cost)

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
        theta1 = 3.14 * np.random.normal(0, 0.5)
        theta2 = 3.14 * np.random.normal(0, 0.25)
        vel1 = 3.14 * np.random.normal(0, 0.5)
        vel2 = 3.14 * np.random.normal(0, 0.2)
        states["double_pendulum/model_state"][:] = [theta1, theta2,vel1, vel2]
        # todo:this problem!
        observation = self._reset(states)
        # Reset step counter
        self.steps = 0
        return observation