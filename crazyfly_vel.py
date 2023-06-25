import eagerx
import stable_baselines3 as sb3
from crazyfly.objects import Crazyflie
from crazyfly.crazyflie_env import Crazyfly_Env
import helper
from crazyfly.crazyflie_render import Overlay
RESET = 0
rate = 60
graph = eagerx.Graph.create()
TRAIN = 0

sensors = ["pos","vel","orientation","image","u_applied"]
# sensors = ["pos", "orientation", "image","u_applied"]
# sensors = ["pos", "orientation", "image",]
crazyflie = Crazyflie.make(name="crazyflie",
                        sensors=sensors,
                        states=["model_state"],
                        actuators=["commanded_thrust", 'commanded_attitude'],
                        rate=rate,
                        render_fn="crazyflie_render_fn",
                        )
from eagerx_ode.engine import OdeEngine
graph.add(crazyflie)
ode_engine = OdeEngine.make(rate=rate)
# Connect Crazyflie inputs

graph.connect(action="desired_attitude", target=crazyflie.actuators.commanded_attitude)
graph.connect(action="desired_thrust", target=crazyflie.actuators.commanded_thrust)
# Connect Crazyflie outputs
graph.connect(source=crazyflie.sensors.orientation, observation="orientation")

# graph.connect(source=crazyflie.sensors.pos, observation="position")
if TRAIN:
    graph.connect(source=crazyflie.sensors.pos, observation="position",window=1,delay=0.005)
    graph.connect(source=crazyflie.sensors.vel, observation="velocity",delay=0.01)
else:
    from crazyfly.nodes import Offset
    offset = Offset.make("Offset", rate=rate)
    graph.add(offset)
    graph.connect(source=crazyflie.sensors.vel, observation="velocity")
    graph.connect(source=crazyflie.sensors.pos, target=offset.inputs.raw_position)
    graph.connect(source= offset.outputs.offset_pos, observation="position",window=1)
graph.connect(source=crazyflie.sensors.u_applied, observation="u_applied",window=2)
# graph.connect(source=crazyflie.sensors.vel, observation="velocity")
# graph.render(source=crazyflie.sensors.image, rate=rate)




overlay = Overlay.make("Overlay", rate=rate,engine="ode")
graph.add(overlay)
graph.connect(source=crazyflie.sensors.pos, target=overlay.inputs.pos)
graph.connect(source=crazyflie.sensors.orientation, target=overlay.inputs.orientation)
graph.connect(source=crazyflie.sensors.image, target=overlay.inputs.base_image)
graph.connect(action="desired_attitude", target=overlay.inputs.commanded_attitude)
graph.connect(action="desired_thrust", target=overlay.inputs.commanded_thrust)
graph.render(overlay.outputs.image, rate=30)
if RESET:
    from crazyfly.crazyflie_reset import ResetCF
    u_min = [10000.,-30.0,-30.,]
    u_max = [40000.,30.0,30.,]
    reset = ResetCF.make("ResetCF", rate, u_range=[u_min, u_max])
    graph.add([reset])
    graph.connect(action="desired_attitude", target=reset.feedthroughs.commanded_attitude)
    graph.connect(action="desired_thrust", target=reset.feedthroughs.commanded_thrust)
    graph.connect(source=crazyflie.states.model_state, target=reset.targets.goal)
    graph.connect(source=crazyflie.sensors.pos, target=reset.inputs.pos)
    graph.connect(source=crazyflie.sensors.orientation, target=reset.inputs.orientation)
    graph.connect(source=reset.outputs.commanded_thrust, target=crazyflie.actuators.commanded_thrust)
    graph.connect(source=reset.outputs.commanded_attitude, target=crazyflie.actuators.commanded_attitude)
# crazyflie.gui(OdeEngine)
graph.gui()
# Connect picture making node
# print(graph.is_valid())
from eagerx.wrappers import Flatten
from stable_baselines3.common.env_checker import check_env
from gym.wrappers.rescale_action import RescaleAction
train_env = Crazyfly_Env(name="train", rate=rate, graph=graph, engine=ode_engine, eval=False)
train_env = Flatten(train_env)
print("action_space: ", train_env.action_space)
print("observation_space: ", train_env.observation_space)
train_env = RescaleAction(train_env, min_action=-1.0, max_action=1.0)
# check_env(train_env)
# train_env.gui()
if __name__ == '__main__':
    if TRAIN:
        # model = sb3.SAC.load("sac_cf1.zip")
        model = sb3.SAC("MlpPolicy", train_env, verbose=1, learning_rate=7e-4, gamma=0.98,tensorboard_log="./sac_cf/")
        # model = sb3.SAC("MlpPolicy", train_env, verbose=1, learning_rate=7e-4, gamma=0.98)
        train_env.render("human")
        model.learn(total_timesteps=int(800000))
        # model.learn(total_timesteps=int(800000))
        train_env.close()
        model.save("sac_cf")
    else:
        sac_model = sb3.SAC.load("sac_cf.zip")
        helper.evaluate(sac_model, train_env, n_eval_episodes=5, episode_length=600, video_rate=rate, video_prefix="cf")
