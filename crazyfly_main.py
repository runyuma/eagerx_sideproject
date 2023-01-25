import eagerx
import stable_baselines3 as sb3
from crazyfly.objects import Crazyflie
from crazyfly.crazyflie_env import Crazyfly_Env
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
# graph.gui(crazyflie)
train_env = Crazyfly_Env(name="train", rate=rate, graph=graph, engine=ode_engine, eval=False)
print("action_space: ", train_env.action_space)
print("observation_space: ", train_env.observation_space)
from eagerx.wrappers import Flatten
from stable_baselines3.common.env_checker import check_env
train_env = Flatten(train_env)
check_env(train_env)