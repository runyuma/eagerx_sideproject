# ROS IMPORTS
from std_msgs.msg import Float32MultiArray, Float32

# EAGERx IMPORTS
from typing import List
from eagerx_ode.engine import OdeEngine
from eagerx_pybullet.engine import PybulletEngine
from eagerx import Object, EngineNode, EngineState, Space
from eagerx.core.specs import ObjectSpec
from eagerx.core.graph_engine import EngineGraph
import eagerx.core.register as register


class Crazyflie(Object):
    # entity_id = "Crazyflie"
    @classmethod
    @register.sensors(
        pos=Space(low=[-999, -999, -999],high=[999, 999, 999], shape=(3,),dtype="float32"),
        vel=Space(low=[-100, -100, -100],high=[100, 100, 100], shape=(3,),dtype="float32"),
        orientation=Space(low=[-1, -1, -1, -1],high=[1, 1, 1, 1], shape=(4,),dtype="float32"),
        gyroscope=Space(low=[-10, -10, -10],high=[10, 10, 10], shape=(3,),dtype="float32"),
        accelerometer=Space(low=[-10, -10, -10],high=[10, 10, 10], shape=(3,),dtype="float32"),
        state_estimator=Space(low=[-10, -10, -10, -10],high=[10, 10, 10, 10], shape=(4,),dtype="float32"),
    )
    @register.engine_states(
        pos=Space(),
        vel=Space(),
        orientation=Space(),
        angular_vel=Space(),
        lateral_friction=Space(low=0.1,high=0.5, shape=(),dtype="float32"),
        model_state=Space(low=[-1000, 1000, 0, -100, -100, -100, -30, -30, -100000],high=[1000, 1000, 1000, 100, 100, 100, 30, 30, 100000], shape=(9,),dtype="float32"),
    )
    @register.actuators(pwm_input=Space(low=[0.2, 0.2, 0],high=[0.2, 0.2, 0], shape=(3,),dtype="float32"),
                        desired_thrust=Space(),
                        desired_attitude=Space(),
                        commanded_thrust=Space(low=[10000],high=[60000], shape=(1,),dtype="float32"),
                        commanded_attitude=Space(low=[-30, -30, -30],high=[30, 30, 30], shape=(3,),dtype="float32"))
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
    ):
        """Object spec of Solid"""
        # Performs all the steps to fill-in the params with registered info about all functions.
        # Crazyflie.initialize_spec(spec)

        # Modify default agnostic params
        # Only allow changes to the agnostic params (rates, windows, (space)converters, etc...
        spec = cls.get_specification()
        spec.config.name = name
        spec.config.sensors = ["pos", "vel", "orientation", "gyroscope", "accelerometer", "state_estimator"]if sensors is None else sensors
        spec.config.states = states if states is not None else ["pos", "vel", "orientation", "angular_vel",
                                                                "model_state"]
        spec.config.actuators = actuators if actuators is not None else []  # ["external_force"]

        # Add registered agnostic params
        spec.config.urdf = urdf
        spec.config.base_pos = base_pos if base_pos else [0, 0, 0]
        spec.config.base_or = base_or if base_or else [0, 0, 0, 1]
        spec.config.self_collision = self_collision
        spec.config.fixed_base = fixed_base
        return spec
        # Add agnostic implementation


    @staticmethod
    @register.engine(OdeEngine)
    def ode_engine(spec: ObjectSpec, graph: EngineGraph):
        # Import any object specific entities for this engine
        # import Crazyflie_Simulation.solid.ode  # noqa # pylint: disable=unused-import

        from eagerx_ode.engine_states import OdeEngineState, OdeParameters
        from crazyfly.engine_nodes import FloatMultiArrayOutput,OdeMultiInput
        from eagerx_ode.engine_nodes import OdeOutput, OdeInput, OdeRender, ActionApplied
        # Set object arguments
        spec.OdeEngine.ode = "crazyfly.crazyflie_ode/crazyflie_ode"

        # Set default parameters of crazyflie ode [mass, gain, time constant]
        spec.OdeEngine.ode_params = [0.03303, 1.1094, 0.183806]

        # Create engine_states
        spec.engine.states.model_state = OdeEngineState.make()

        # Create output engine node
        x = OdeOutput.make( "x", rate=spec.sensors.gyroscope.rate, process=2)

        # Create sensor engine nodes
        # FloatMultiArrayOutput
        pos = FloatMultiArrayOutput.make( "pos", rate=spec.sensors.pos.rate, idx=[0, 1, 2])
        orientation = FloatMultiArrayOutput.make("orientation", rate=spec.sensors.orientation.rate,
                                      idx=[6, 7])

        # Create actuator engine nodes
        # todo:define input
        action = OdeMultiInput.make("crazyflie_ode", rate=spec.actuators.commanded_thrust.rate,
                                 process=2, default_action=[10000, 0, 0])
        print(spec.actuators.commanded_thrust.rate)
        # Connect all engine nodes
        graph.add([x, pos, orientation, action])
        # actuator
        graph.connect(actuator="commanded_attitude", target=action.inputs.commanded_attitude)
        graph.connect(actuator="commanded_thrust", target=action.inputs.commanded_thrust)

        # observation
        graph.connect(source=x.outputs.observation, target=pos.inputs.observation_array)
        graph.connect(source=x.outputs.observation, target=orientation.inputs.observation_array)

        # sensors
        graph.connect(source=pos.outputs.observation, sensor="pos")
        graph.connect(source=orientation.outputs.observation, sensor="orientation")
        print(graph.is_valid())
        # Check graph validity (commented out)
        # graph.is_valid(plot=True)
        # graph.gui()
if __name__ == "__main__":
    crazyflie = Crazyflie.make(name="crazyflie",
                               rate=50.0,
                               sensors=["pos", "orientation"],
                               actuators=["commanded_thrust", 'commanded_attitude'],
                               base_pos=[0, 0, 1], fixed_base=False,
                               states=["model_state"])

        # graph.gui()
