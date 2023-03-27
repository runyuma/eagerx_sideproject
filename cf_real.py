import eagerx
import stable_baselines3 as sb3
from crazyfly.objects import Crazyflie
from crazyfly.crazyflie_env import Crazyfly_Env
import rospy
rospy.init_node("crazyfly")
rate = 60
graph = eagerx.Graph.create()
sensors = ["pos", "orientation",]
crazyflie = Crazyflie.make(name="crazyflie",
                        sensors=sensors,
                        states=["model_state"],
                        actuators=["commanded_thrust", 'commanded_attitude'],
                        rate=rate,
                        render_fn="crazyflie_render_fn",
                        base_pos=[0, 0, 1], fixed_base=False,
                        )
from eagerx_reality.engine import RealEngine
graph.add(crazyflie)
ode_engine = RealEngine.make(rate=rate)
graph.connect(action="desired_attitude", target=crazyflie.actuators.commanded_attitude)
graph.connect(action="desired_thrust", target=crazyflie.actuators.commanded_thrust)
graph.connect(source=crazyflie.sensors.orientation, observation="orientation")
graph.connect(source=crazyflie.sensors.pos, observation="position",window=2,delay=0.005)
# graph.gui()
from eagerx.wrappers import Flatten
from stable_baselines3.common.env_checker import check_env
from gym.wrappers.rescale_action import RescaleAction
train_env = Crazyfly_Env(name="train", rate=rate, graph=graph, engine=ode_engine, eval=False)
train_env = Flatten(train_env)
print("action_space: ", train_env.action_space)
print("observation_space: ", train_env.observation_space)
train_env = RescaleAction(train_env, min_action=-1.0, max_action=1.0)
# check_env(train_env)
train_env.gui()