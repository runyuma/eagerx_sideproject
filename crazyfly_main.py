import eagerx
import stable_baselines3 as sb3
from crazyfly.objects import Crazyflie

rate = 50
graph = eagerx.Graph.create()
crazyflie = Crazyflie.make(name="crazyflie",
                               rate=50.0,
                               sensors=["pos", "orientation"],
                               actuators=["commanded_thrust", 'commanded_attitude'],
                               base_pos=[0, 0, 1], fixed_base=False,
                               states=["model_state"])
from eagerx_ode.engine import OdeEngine
ode_engine = OdeEngine.make(rate=rate)
graph.add(crazyflie)
# Connect Crazyflie inputs
graph.connect(action="desired_attitude", target=crazyflie.actuators.commanded_attitude)
graph.connect(action="desired_thrust", target=crazyflie.actuators.commanded_thrust)
# Connect Crazyflie outputs
graph.connect(source=crazyflie.sensors.orientation, observation="orientation")
graph.connect(source=crazyflie.sensors.pos, observation="position")
# Connect picture making node
# graph.connect(source=crazyflie.sensors.orientation, target=make_picture.inputs.orientation)
# graph.connect(source=crazyflie.sensors.pos, target=make_picture.inputs.position)
# graph.render(source=make_picture.outputs.image, rate=rate)
# print(graph.is_valid())
# graph.gui()