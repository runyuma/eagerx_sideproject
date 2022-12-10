
import eagerx
from typing import Dict
import numpy as np
# from huggingface_sb3 import load_from_hub
# import stable_baselines3 as sb3

from double_pendulum.objects import Double_Pendulum

rate = 20.0
graph = eagerx.Graph.create()
sensors = ["theta", "theta_dot", "image"]
actuators = ["u"]
states = ["model_state",]
pendulum = Double_Pendulum.make("doubel_pendulum", rate=rate, actuators=actuators, sensors=sensors, states=states, render_fn="double_pendulum_render_fn")

from double_pendulum.processor import DecomposedAngle
pendulum.sensors.theta.processor = DecomposedAngle.make()
pendulum.sensors.theta.space.low = -1
pendulum.sensors.theta.space.high = 1
pendulum.sensors.theta.space.shape = [4]

graph.add(pendulum)

# Connect the pendulum to an action and observations
graph.connect(action="voltage", target=pendulum.actuators.u)
graph.connect(source=pendulum.sensors.theta, observation="angle")
graph.connect(source=pendulum.sensors.theta_dot, observation="angular_velocity")

# Render image
graph.render(source=pendulum.sensors.image, rate=rate)
Double_Pendulum.info()

from eagerx_ode.engine import OdeEngine
from double_pendulum.double_pendulum_env import Double_PendulumEnv
ode_engine = OdeEngine.make(rate=rate)
train_env = Double_PendulumEnv(name="train", rate=rate, graph=graph, engine=ode_engine, eval=False)
print("action_space: ", train_env.action_space)
print("observation_space: ", train_env.observation_space)
ode_render = pendulum.gui(OdeEngine, interactive=False, filename="ode_render.svg")
# from eagerx.wrappers import Flatten
# train_env = Flatten(train_env)
# model = sb3.SAC("MlpPolicy", train_env, verbose=1, learning_rate=7e-4)
# train_env.render("human")
# model.learn(total_timesteps=int(4000))
# train_env.close()
# model.save("double_pendulum")