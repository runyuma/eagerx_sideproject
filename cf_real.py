import eagerx
import stable_baselines3 as sb3
from crazyfly.objects import Crazyflie
from crazyfly.crazyflie_env import Crazyfly_Env
import rospy
from crazyfly.crazyflie_render import Overlay
rospy.init_node("crazyfly")
rate = 60
graph = eagerx.Graph.create()
sensors = ["pos", "orientation","u_applied","image"]
crazyflie = Crazyflie.make(name="crazyflie",
                        sensors=sensors,
                        states=["model_state"],
                        actuators=["commanded_thrust", 'commanded_attitude'],
                        rate=rate,
                        render_fn="crazyflie_render_fn",
                        )
from eagerx_reality.engine import RealEngine
graph.add(crazyflie)
engine = RealEngine.make(rate=rate)
RESET = False
if not RESET:
    graph.connect(action="desired_attitude", target=crazyflie.actuators.commanded_attitude)
    graph.connect(action="desired_thrust", target=crazyflie.actuators.commanded_thrust)
graph.connect(source=crazyflie.sensors.orientation, observation="orientation")
graph.connect(source=crazyflie.sensors.pos, observation="position",window=2,)
graph.connect(source=crazyflie.sensors.u_applied, observation="u_applied",window=2)

overlay = Overlay.make("Overlay", rate=rate)
graph.add(overlay)
graph.connect(source=crazyflie.sensors.pos, target=overlay.inputs.pos)
graph.connect(source=crazyflie.sensors.orientation, target=overlay.inputs.orientation)
graph.connect(source=crazyflie.sensors.image, target=overlay.inputs.base_image)
graph.connect(action="desired_attitude", target=overlay.inputs.commanded_attitude)
graph.connect(action="desired_thrust", target=overlay.inputs.commanded_thrust)
graph.render(overlay.outputs.image, rate=30)
from crazyfly.crazyflie_reset import ResetCFReal

u_min = [10000., -30.0, -30., ]
u_max = [40000., 30.0, 30., ]
if RESET:
    reset = ResetCFReal.make("ResetCF", rate, u_range=[u_min, u_max])
    graph.add(reset)
    graph.connect(action="desired_attitude", target=reset.feedthroughs.commanded_attitude)
    graph.connect(action="desired_thrust", target=reset.feedthroughs.commanded_thrust)
    graph.connect(source=crazyflie.states.model_state, target=reset.targets.goal)
    graph.connect(source=crazyflie.sensors.pos, target=reset.inputs.pos)
    graph.connect(source=crazyflie.sensors.orientation, target=reset.inputs.orientation)
    graph.connect(source=reset.outputs.commanded_thrust, target=crazyflie.actuators.commanded_thrust)
    graph.connect(source=reset.outputs.commanded_attitude, target=crazyflie.actuators.commanded_attitude)

# graph.gui()
from eagerx.wrappers import Flatten
from stable_baselines3.common.env_checker import check_env
from gym.wrappers.rescale_action import RescaleAction
test_env = Crazyfly_Env(name="test", rate=rate, graph=graph, engine=engine, eval=True)
test_env = Flatten(test_env)
print("action_space: ", test_env.action_space)
print("observation_space: ", test_env.observation_space)
test_env = RescaleAction(test_env, min_action=-1.0, max_action=1.0)
# check_env(train_env)
test_env.gui()
if __name__ == '__main__':
    # model = sb3.SAC("MlpPolicy", test_env, verbose=1, learning_rate=7e-4, gamma=0.98)
    #
    # test_env.render("human")
    # model.learn(total_timesteps=int(400000))
    # test_env.close()
    ##################

    import helper
    sac_model = sb3.SAC.load("sac_cf.zip")
    test_env.render("human")
    test_env.reset()
    helper.evaluate(sac_model,test_env,n_eval_episodes=2,episode_length=400, video_rate=rate,video_prefix="cfreal")