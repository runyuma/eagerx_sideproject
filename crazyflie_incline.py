import eagerx
import stable_baselines3 as sb3
from crazyfly.objects import Crazyflie
from crazyfly.crazyflie_env import Crazyfly_Env,Crazyfly_Env_landing
RESET = 0
rate = 50
graph = eagerx.Graph.create()
crazyflie = Crazyflie.make(name="crazyflie",
                           sensors=["pos", "vel", "orientation", "image"],
                           states=["model_state","target_posture"],
                           actuators=["commanded_thrust", 'commanded_attitude'],
                           rate=50.0,
                           render_fn="crazyflie_render_fn",
                           base_pos=[0, 0, 1], fixed_base=False, height_only=False
                           )
from eagerx_ode.engine import OdeEngine
graph.add(crazyflie)
ode_engine = OdeEngine.make(rate=rate,real_time_factor=0)
graph.connect(action="desired_attitude", target=crazyflie.actuators.commanded_attitude)
graph.connect(action="desired_thrust", target=crazyflie.actuators.commanded_thrust)
# Connect Crazyflie outputs
graph.connect(source=crazyflie.sensors.orientation, observation="orientation")
graph.connect(source=crazyflie.sensors.pos, observation="position")
graph.connect(source=crazyflie.sensors.vel, observation="velocity")
# graph.connect(source=crazyflie.states.target_posture, observation="target_state")

graph.gui()
from eagerx.wrappers import Flatten
from stable_baselines3.common.env_checker import check_env
from gym.wrappers.rescale_action import RescaleAction
train_env = Crazyfly_Env_landing(name="train", rate=rate, graph=graph, engine=ode_engine, eval=False)
train_env = Flatten(train_env)
print("action_space: ", train_env.action_space)
print("observation_space: ", train_env.observation_space)
train_env = RescaleAction(train_env, min_action=-1.0, max_action=1.0)
# check_env(train_env)
# train_env.gui()
# if __name__ == '__main__':
    # model = sb3.SAC("MlpPolicy", train_env, verbose=1, learning_rate=7e-4, gamma=0.98,tensorboard_log="./sac_cf/")
    # model = sb3.SAC("MlpPolicy", train_env, verbose=1, learning_rate=7e-4, gamma=0.98)
    # train_env.render("human")
    # # model.learn(total_timesteps=int(160000))
    # model.learn(total_timesteps=int(80000))
    # train_env.close()
    # model.save("sac_cf")