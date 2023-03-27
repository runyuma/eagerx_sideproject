from typing import Optional, List
import eagerx
from eagerx import Space, ResetNode
from eagerx.core.specs import ResetNodeSpec
from eagerx.utils.utils import Msg
import numpy as np
import pygame
class ResetCF(ResetNode):
    @classmethod
    def make(cls,
        name: str,
        rate: float,
        threshold: float = 0.1,
        timeout: float = 3.0,
        u_range: Optional[List[float]] = None,
    ) -> ResetNodeSpec:
        spec = cls.get_specification()
        spec.config.update(name=name, rate=rate, process=eagerx.process.ENVIRONMENT, color="grey")
        spec.config.update(inputs=["pos", "orientation"], targets=["goal"], outputs=["commanded_thrust", 'commanded_attitude'])
        spec.config.update(u_range=u_range, threshold=threshold, timeout=timeout)

        return spec

    def reset(self):
        self.ts_start_routine = None

    def initialize(self, spec: ResetNodeSpec):
        self.threshold = spec.config.threshold
        self.timeout = spec.config.timeout
        self.u_min, self.u_max = spec.config.u_range
        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            print("No joystick found")
        else:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()

    @eagerx.register.inputs(pos = Space(low=[-2, -2, -0],high=[2, 2, 2], shape=(3,),dtype="float32"),
                            orientation=Space(low=[-1, -1, -1],high=[1, 1, 1], shape=(3,),dtype="float32"))
    @eagerx.register.targets(goal=Space(low=[-1000, -1000, 0, -100, -100, -100, -30, -30, -100000],high=[1000, 1000, 1000, 100, 100, 100, 30, 30, 100000], shape=(9,),dtype="float32"))
    @eagerx.register.outputs(commanded_thrust=Space(low=[10000],high=[60000], shape=(1,),dtype="float32"),
                             commanded_attitude=Space(low=[-30, -30, -100],high=[30, 30, 100], shape=(3,),dtype="float32"))
    def callback(self, t_n: float, goal: Msg, pos: Msg, orientation: Msg):
        if self.ts_start_routine is None:
            self.ts_start_routine = t_n
        # print(pos.msgs,orientation.msgs)
        # print("reseting", goal.msgs)
        done = (t_n - self.ts_start_routine) > self.timeout

        pygame.event.get()
        axis0 = self.joystick.get_axis(0)
        axis1 = self.joystick.get_axis(1)
        axis4 = self.joystick.get_axis(4)
        thrust = 10000 + (60000 - 10000) * (axis4 + 1) / 2
        roll = 30 * axis0
        pitch = 30 * axis1
        commanded_thrust = np.array([thrust], dtype="float32")
        commanded_attitude = np.array([roll, pitch, 0], dtype="float32")
        output_msgs = {"commanded_thrust": commanded_thrust,"commanded_attitude":commanded_attitude, "goal/done": done}
        return output_msgs