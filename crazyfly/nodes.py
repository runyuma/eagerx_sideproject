from typing import Optional, List, Dict, Any
import numpy as np
# IMPORT EAGERX
import eagerx
from eagerx.core.specs import NodeSpec
from eagerx.core.space import Space
from eagerx.core.constants import process
from eagerx.utils.utils import Msg
import eagerx.core.register as register
class Offset(eagerx.Node):
    @classmethod
    def make(
            cls,
            name: str,
            rate: float,
            process: int = eagerx.ENVIRONMENT,
            color: str = "cyan",
    ) -> NodeSpec:
        # Get base parameter specification with defaults parameters
        spec = cls.get_specification()
        # Adjust default params
        spec.config.update(name=name, rate=rate, process=process, color=color)
        # spec.config.update(inputs=[ "position"], outputs=["offset","offset_pos"])
        spec.config.inputs = ["raw_position"]
        spec.config.outputs = ["offset","offset_pos"]
        return spec

    def initialize(self, spec: NodeSpec):
        #camera matrix
        pass

    @register.states()
    def reset(self):
        # point list
        self.step = 0

    @register.inputs(
    raw_position=Space( dtype="float32")
    )
    @register.outputs(
    offset=Space(low=[-2, -2, -2], high=[2, 2, 0], shape=(3,), dtype="float32"),
    offset_pos=Space(low=[-2, -2, -2], high=[2, 2, 0], shape=(3,), dtype="float32"),)
    def callback(self, t_n: float,raw_position: Msg):
        if len(raw_position.msgs[-1].data) > 0:
            # print("position######################", raw_position)
            radius = 0.
            offset = np.array([radius*np.cos(self.step*np.pi/400),radius*np.sin(self.step*np.pi/400),0], dtype="float32")

            offset_pos = raw_position.msgs[-1].data - offset
            self.step += 1
            return dict(offset=offset,offset_pos=offset_pos)
        else:
            return dict(offset=np.array([0,0,-0], dtype="float32"),offset_pos=np.array([0,0,-0.5], dtype="float32"))
class low_pass_filter(eagerx.Node):
    @classmethod
    def make(
            cls,
            name: str,
            rate: float,
            process: int = eagerx.ENVIRONMENT,
            color: str = "cyan",
    ) -> NodeSpec:
        spec = cls.get_specification()
        # Adjust default params
        spec.config.update(name=name, rate=rate, process=process, color=color)
        # spec.config.update(inputs=[ "position"], outputs=["offset","offset_pos"])
        spec.config.inputs = ["position"]
        spec.config.outputs = ["filtered_velocity"]
        return spec
    def initialize(self, spec: NodeSpec):
        self.rate = spec.config.rate
        self.list_length = 3
    @register.states()
    def reset(self):
        # point list
        self.postion_list = []
        self.vel = np.array([0,0,0], dtype="float32")
    @register.inputs(position=Space(low=[-2, -2, -2], high=[2, 2, 0], shape=(3,), dtype="float32"))
    @register.outputs(filtered_velocity=Space(low=[-100, -100, -100],high=[100, 100, 100], shape=(3,),dtype="float32"))
    def callback(self, t_n: float,position: Msg):
        if len(position.msgs[-1].data) > 0:
            if len(self.postion_list) == 0:
                self.postion_list.append(np.array(position.msgs[-1].data))
                return dict(filtered_velocity=np.array([0, 0, 0], dtype="float32"))
            else:
                if (np.array(position.msgs[-1].data)==self.postion_list[-1]).all():
                    print("lost detection")
                    # when lost detection, use the last velocity
                    filtered_velocity = self.vel
                    self.postion_list.append(np.array(position.msgs[-1].data))
                    if len(self.postion_list) > self.list_length:
                        self.postion_list.pop(0)
                else:
                    self.postion_list.append(np.array(position.msgs[-1].data))
                    if len(self.postion_list) > self.list_length:
                        self.postion_list.pop(0)
                    filtered_velocity = (self.postion_list[-1] - self.postion_list[0])/(len(self.postion_list)-1)*self.rate
                    self.vel = filtered_velocity
                print("filtered_velocity", filtered_velocity)
                return dict(filtered_velocity=filtered_velocity)
        else:
            return dict(filtered_velocity=np.array([0,0,0], dtype="float32"))


