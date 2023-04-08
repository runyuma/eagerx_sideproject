# ROS IMPORTS
# from std_msgs.msg import Float32MultiArray, Float32

# EAGERx IMPORTS
from typing import List
from eagerx_ode.engine import OdeEngine
from eagerx_pybullet.engine import PybulletEngine
from eagerx import Object, EngineNode, EngineState, Space
from eagerx.core.specs import ObjectSpec
from eagerx.core.graph_engine import EngineGraph
import eagerx.core.register as register
from eagerx_reality.engine import RealEngine


class Crazyflie(Object):
    # entity_id = "Crazyflie"
    @classmethod
    @register.sensors(
        pos=Space(low=[-2, -2, -2],high=[2, 2, 0], shape=(3,),dtype="float32"),
        height=Space(low=[-2,-5],high=[2,5], shape=(2,),dtype="float32"),
        vel=Space(low=[-100, -100, -100],high=[100, 100, 100], shape=(3,),dtype="float32"),
        orientation=Space(low=[-1, -1, -1],high=[1, 1, 1], shape=(3,),dtype="float32"),
        gyroscope=Space(low=[-10, -10, -10],high=[10, 10, 10], shape=(3,),dtype="float32"),
        accelerometer=Space(low=[-10, -10, -10],high=[10, 10, 10], shape=(3,),dtype="float32"),
        state_estimator=Space(low=[-10, -10, -10, -10],high=[10, 10, 10, 10], shape=(4,),dtype="float32"),
        image=Space(dtype="uint8"),
        u_applied=Space(low=[10000,-30, -30],high=[40000,30, 30], shape=(3,),dtype="float32"),
    )
    @register.engine_states(
        pos=Space(),
        vel=Space(),
        orientation=Space(),
        angular_vel=Space(),
        lateral_friction=Space(low=0.1,high=0.5, shape=(),dtype="float32"),
        target_posture=Space(low=[-1,-1,0,-30,-30],high=[1,1,1,30,30], shape=(5,),dtype="float32"),
        model_state=Space(low=[-1000, -1000, -1000, -100, -100, -100, -30, -30, -100000],high=[1000, 1000, 1000, 100, 100, 100, 30, 30, 100000], shape=(9,),dtype="float32"),
    )
    @register.actuators(pwm_input=Space(low=[0.2, 0.2, 0],high=[0.2, 0.2, 0], shape=(3,),dtype="float32"),
                        desired_thrust=Space(),
                        desired_attitude=Space(),
                        commanded_thrust=Space(low=[10000],high=[40000], shape=(1,),dtype="float32"),
                        commanded_attitude=Space(low=[-30, -30, -100],high=[30, 30, 100], shape=(3,),dtype="float32"))
    # @register.config(urdf=None, fixed_base=True, self_collision=True, base_pos=[0, 0, 0], base_or=[0, 0, 0, 1])



    def make(
            cls,
            name: str,
            urdf: str = None,
            sensors: List[str] = None,
            states: List[str] = None,
            actuators: List[str] = None,
            rate: float = 50.0,
            render_shape: List[int] = None,
            render_fn: str = None,
            base_pos=None,
            base_or=None,
            self_collision=True,
            fixed_base=True,
            height_only=False,
            target_pos=False
    ):
        """Object spec of Solid"""
        # Performs all the steps to fill-in the params with registered info about all functions.
        # Crazyflie.initialize_spec(spec)

        # Modify default agnostic params
        # Only allow changes to the agnostic params (rates, windows, (space)converters, etc...
        spec = cls.get_specification()
        spec.config.name = name
        spec.config.height_only = height_only
        spec.config.target_pos = target_pos

        if not height_only:
            spec.config.sensors = ["pos", "vel", "orientation", ] if sensors is None else sensors
            spec.sensors.pos.rate = rate
            spec.sensors.vel.rate = rate
        if height_only:
            spec.config.sensors = ["height",  "orientation", ] if sensors is None else sensors
            spec.sensors.height.rate = rate
        spec.config.states = states if states is not None else ["pos", "vel", "orientation", "angular_vel",
                                                              "model_state"]
        spec.config.actuators = actuators if actuators is not None else []  # ["external_force"]

        # Add registered agnostic params
        # spec.config.urdf = urdf
        # spec.config.base_pos = base_pos if base_pos else [0, 0, 0]
        # spec.config.base_or = base_or if base_or else [0, 0, 0, 1]
        # spec.config.self_collision = self_collision
        # spec.config.fixed_base = fixed_base

        # Set image space
        spec.config.render_shape = render_shape if render_shape else [800, 800]
        spec.config.render_fn = render_fn if render_fn else "crazyflie_render_fn"
        shape = (spec.config.render_shape[0], spec.config.render_shape[1], 3)
        spec.sensors.image.space = Space(low=0, high=255, shape=shape, dtype="uint8")

        spec.sensors.orientation.rate = rate
        spec.sensors.image.rate = rate
        spec.sensors.u_applied.rate = rate
        spec.actuators.commanded_thrust.rate = rate
        spec.actuators.commanded_attitude.rate = rate

        return spec

    @staticmethod
    @register.engine(OdeEngine)
    def ode_engine(spec: ObjectSpec, graph: EngineGraph):
        # Import any object specific entities for this engine
        # import Crazyflie_Simulation.solid.ode  # noqa # pylint: disable=unused-import

        from eagerx_ode.engine_states import OdeEngineState, OdeParameters
        from crazyfly.engine_nodes import FloatMultiArrayOutput,OdeMultiInput,ActionApplied
        from crazyfly.engine_states import TargetState
        from eagerx_ode.engine_nodes import OdeOutput, OdeInput, OdeRender
        # Set object arguments
        spec.engine.ode = "crazyfly.crazyflie_ode/crazyflie_ode"
        render_fn = f"crazyfly.crazyflie_render/{spec.config.render_fn}"
        shape = spec.sensors.image.space.shape[:2]
        image = OdeRender.make("image", render_fn=render_fn, rate=spec.sensors.image.rate, process=2, shape=shape)

        # Set default parameters of crazyflie ode [mass, gain, time constant]
        spec.engine.ode_params = [0.03303, 1.1094, 0.183806]

        # Create engine_states
        spec.engine.states.model_state = OdeEngineState.make()

        # Create output engine node
        x = OdeOutput.make( "x", rate=spec.sensors.orientation.rate, process=2)

        # Create sensor engine nodes
        # FloatMultiArrayOutput
        height_only = spec.config.height_only
        orientation = FloatMultiArrayOutput.make("orientation", rate=spec.sensors.orientation.rate,
                                                 idx=[6, 7])
        action = OdeMultiInput.make("crazyflie_ode", rate=spec.actuators.commanded_thrust.rate,
                                    process=2, default_action=[10000, 0, 0])
        action_applied = ActionApplied.make("action_applied", rate=spec.actuators.commanded_thrust.rate,)

        pos = FloatMultiArrayOutput.make( "pos", rate=spec.sensors.pos.rate, idx=[0, 1, 2])
        vel = FloatMultiArrayOutput.make( "vel", rate=spec.sensors.vel.rate, idx=[3, 4, 5])
        graph.add([x, pos,vel, orientation, action,action_applied, image])
        graph.connect(source=x.outputs.observation, target=pos.inputs.observation_array)
        graph.connect(source=pos.outputs.observation, sensor="pos")
        graph.connect(source=x.outputs.observation, target=vel.inputs.observation_array)
        graph.connect(source=vel.outputs.observation, sensor="vel")


        # actuator
        graph.connect(actuator="commanded_attitude", target=action.inputs.commanded_attitude)
        graph.connect(actuator="commanded_thrust", target=action.inputs.commanded_thrust)
        graph.connect(source=action.outputs.action_applied, target=action_applied.inputs.action_applied, skip=True)


        # observation

        graph.connect(source=x.outputs.observation, target=orientation.inputs.observation_array)

        graph.connect(source=x.outputs.observation, target=image.inputs.observation)
        graph.connect(source=image.outputs.image, sensor="image")
        # sensors

        graph.connect(source=orientation.outputs.observation, sensor="orientation")

        graph.connect(source=action.outputs.action_applied, target=image.inputs.action_applied, skip=True)
        graph.connect(source=action_applied.outputs.action_applied, sensor="u_applied")

        
    @staticmethod
    @register.engine(RealEngine)
    def real_engine(spec: ObjectSpec, graph: EngineGraph):
        from crazyfly.engine_nodes import CrazyfliePosition, CrazyflieOrientation, CrazyflieInput,CrazyflieRender
        from crazyfly.engine_states import DummyState
        spec.engine.states.model_state = DummyState.make()
        pos = CrazyfliePosition.make("pos", rate=spec.sensors.pos.rate)
        orientation = CrazyflieOrientation.make("orientation", rate=spec.sensors.orientation.rate)
        action = CrazyflieInput.make("crazyflie_input", rate=spec.actuators.commanded_thrust.rate)
        graph.add([pos, orientation, action])
        graph.connect(actuator="commanded_attitude", target=action.inputs.commanded_attitude)
        graph.connect(actuator="commanded_thrust", target=action.inputs.commanded_thrust)
        graph.connect(source = pos.outputs.observation, sensor="pos")
        graph.connect(source = orientation.outputs.observation, sensor="orientation")

        from crazyfly.engine_nodes import  ActionApplied
        action_applied = ActionApplied.make("action_applied", rate=spec.actuators.commanded_thrust.rate, )
        graph.add([action_applied])
        graph.connect(source = action.outputs.action_applied, target=action_applied.inputs.action_applied, skip=True)
        graph.connect(source=action_applied.outputs.action_applied, sensor="u_applied")

        # image = CrazyflieRender.make("image", rate=spec.sensors.image.rate, process=2,)
        # graph.add([image])
        # graph.connect(source = pos.outputs.image, target=image.inputs.image)
        # graph.connect(source = image.outputs.renderimage, sensor="image")



if __name__ == "__main__":
    crazyflie = Crazyflie.make(name="crazyflie",
                               rate=50.0,
                               sensors=["pos", "orientation"],
                               actuators=["commanded_thrust", 'commanded_attitude'],
                               base_pos=[0, 0, 1], fixed_base=False,
                               states=["model_state"],
                               height_only=True)


