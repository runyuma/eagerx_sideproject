import eagerx
import numpy as np
from typing import Dict
from eagerx.wrappers import Flatten
class Crazyfly_Env(eagerx.BaseEnv):
    def __init__(self, name: str, rate: float, graph: eagerx.Graph, engine: eagerx.specs.EngineSpec, eval: bool,height_only=False):
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
        self.rate = rate
        # Maximum episode length
        self.max_steps = rate*80 if eval else rate*8 # todo
        self.height_only = height_only
        # Step counter
        self.steps = None
        super().__init__(name, rate, graph, engine, backend, force_start=True)

    def step(self, action: Dict):
        # print("testing steps",self.steps,self.max_steps)
        observation = self._step(action)
        # print("original ob",observation["position"])
        self.steps += 1
        cost = 0
        info = {"TimeLimit.truncated": self.steps > self.max_steps}
        if self.steps == 1:
            print("intial_observation",observation)
        # print(self.max_steps,self.eval)
        print("observation",observation)

        pos = np.array(observation["position"][-1])
        if "velocity" in observation:
            vel = np.array(observation["velocity"][-1])
        else:
            vel = (observation["position"][-1] - observation["position"][-2])*self.rate
        if "u_applied" in observation:
            if len(observation["u_applied"])==2:
                cost += 0.25 * np.linalg.norm(observation["u_applied"][-1])
                cost += 0.5*np.linalg.norm(observation["u_applied"][-1]-observation["u_applied"][-2])
            elif len(observation["u_applied"])==1:
                cost += -0.25*np.linalg.norm(observation["u_applied"][-1])
            # print(vel)
        ori = np.array(observation["orientation"][-1])
        # cost+= -3
        pd = np.array([0, 0, -0.5])
        cost+= 1*np.linalg.norm(pos - pd)+2*np.linalg.norm(ori)+ 0.5*np.linalg.norm(vel)
        # print("cost",1*np.linalg.norm(pos - pd),2*np.linalg.norm(ori), 0.5*np.linalg.norm(vel))
        cost -= 1
        # print("vel",vel)
        if np.linalg.norm(vel)<0.25:
            cost-= 3
            # print("good vel")
        if np.linalg.norm(pos - pd)<0.1:
            cost-= 10
            # print("good")
        done = False
        pos_range = 2
        anlge_range = 1
        if self.steps > self.max_steps:
            done = True
        if (np.max(np.abs(pos)) > pos_range) \
                or (np.max(np.abs(ori)) > anlge_range) or pos[2] > 0:
            cost += 20
            print("out of range")
            done = True

        # # if not self.eval:
        # noise = np.random.normal(0, 0.005, 3)
        # pos_noisy = np.array(observation["position"][-1]) + noise
        # observation["position"][-1] = pos_noisy
        # observation["position"][:-1] = self.last_pos[1:]
        # self.last_pos = observation["position"]
        # print("pos",observation["position"])
        return observation, -cost, done, info
    def reset(self) -> Dict:
        p = np.random.rand()
        if p < 0.8:
            states = self.state_space.sample()
            x = np.random.normal(0, 0.25)
            y = np.random.normal(0, 0.25)
            z = np.random.normal(-0.5, 0.2)
            vx = np.random.normal(0, 0.1)
            vy = np.random.normal(0, 0.1)
            vz = np.random.normal(0, 0.1)
            roll = np.random.normal(0, 0.1)
            pitch = np.random.normal(0, 0.1)
            # print(states)
            states["crazyflie/model_state"][:] = [x, y, z, vx, vy, vz, roll, pitch, 0, ]

        else:
            states = self.state_space.sample()
            states["crazyflie/model_state"][:] = [0, 0, -0.5, 0, 0, 0, 0, 0, 0, ]
        observation = self._reset(states)
        self.last_pos = observation["position"]
        # Reset step counter
        self.steps = 0
        return observation
