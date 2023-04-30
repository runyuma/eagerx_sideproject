from typing import Optional, List, Dict, Any
import numpy as np
# from std_msgs.msg import UInt64, Float32MultiArray
# IMPORT EAGERX
import eagerx
from eagerx.core.specs import NodeSpec
from eagerx.core.space import Space
from eagerx.core.constants import process
from eagerx.utils.utils import Msg
from eagerx.core.entities import EngineNode
import eagerx.core.register as register
import pygame
TEST_ROS = False
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import cv2
from cv2 import aruco
class FloatMultiArrayOutput(EngineNode):
    @classmethod
    def make(
            cls,
            name: str,
            rate: float,
            idx: Optional[list] = [0],
            process: Optional[int] = process.ENVIRONMENT,
            color: Optional[str] = "cyan",
    ):
        """
        FloatOutput spec
        :param idx: index of the value of interest from the array.
        """
        # Performs all the steps to fill-in the params with registered info about all functions.
        spec = cls.get_specification()
        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.inputs = ["observation_array"]
        spec.config.outputs = ["observation"]

        # Custom node params
        spec.config.idx = idx
        return spec

    def initialize(self, spec, simulator):
        self.idx = spec.config.idx
        if TEST_ROS:
            rospy.init_node("dummy_node", anonymous=True)
            self.subscriber = rospy.Subscriber("/dummy_input", Float32MultiArray, self.dummy_callback)
    def dummy_callback(self, msg):
        print("dummy callback")
    @register.states()
    def reset(self):
        # print("I am resetting ###############################################################################")
        pass

    @register.inputs(observation_array=Space(dtype="float32"))
    @register.outputs(observation=Space(dtype="float32"))
    def callback(self, t_n: float, observation_array: Optional[Msg] = None):
        data = len(self.idx) * [0]
        for idx, _data in enumerate(self.idx):
            data[idx] = observation_array.msgs[-1].data[_data]
        # if statement to add yaw since Jacob didn't use that
        if self.idx[0] == 6 and self.idx[1] == 7:
            # data.append(0)
            # quater = [0]*4
            # quater[0] = np.cos(data[0]/2)*np.cos(data[1]/2)
            # quater[1] = np.sin(data[0] / 2) * np.cos(data[1] / 2)
            # quater[2] = np.cos(data[0] / 2) * np.sin(data[1] / 2)
            # quater[3] = - np.sin(data[0] / 2) * np.sin(data[1] / 2)
            # data = quater
            data = [observation_array.msgs[-1].data[6],
                    observation_array.msgs[-1].data[7],
                    0,]

        return dict(observation=np.array(data,dtype="float32"))
class ActionApplied(EngineNode):
    @classmethod
    def make(
            cls,
            name: str,
            rate: float,
            process: Optional[int] = process.ENVIRONMENT,
            color: Optional[str] = "cyan",
    ):
        """
        FloatOutput spec
        :param idx: index of the value of interest from the array.
        """
        # Performs all the steps to fill-in the params with registered info about all functions.
        spec = cls.get_specification()
        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.inputs = ["action_applied"]
        spec.config.outputs = ["action_applied"]

        # Custom node params
        return spec

    def initialize(self, spec, simulator):
        pass
    @register.states()
    def reset(self):
        pass

    @register.inputs(action_applied=Space(low=[10000,-30, -30,], high=[60000,30, 30,], shape=(3,), dtype="float32"),)
    @register.outputs(action_applied=Space(dtype="float32"))
    def callback(self, t_n: float,
                 action_applied: Msg,
    ):
        # Send action that has been applied.
        if len(action_applied.msgs) > 0:
            data = action_applied.msgs[-1]
            # print("action applied: ", data)
        else:
            data = np.array([35000,0, 0,], dtype="float32")
        data = (data-np.array([10000,0, 0,], dtype="float32"))/np.array([50000, 30, 30,], dtype="float32")
        # print("action applied: ", type(data))
        return dict(action_applied=data)
class OdeMultiInput(EngineNode):
    @classmethod
    def make(
            cls,
            name: str,
            rate: float,
            default_action: List,
            process: Optional[int] = process.ENGINE,
            color: Optional[str] = "green",
    ):
        """OdeInput spec"""
        spec = cls.get_specification()

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.inputs = ["tick", "commanded_thrust", "commanded_attitude"]
        spec.config.outputs = ["action_applied"]

        # Set custom node params
        spec.config.default_action = default_action
        return spec

    def initialize(self, spec, simulator):
        # We will probably use self.simulator in callback & reset.
        assert (
                spec.config.process == process.ENGINE
        ), "Simulation node requires a reference to the simulator, hence it must be launched in the Engine process"
        self.simulator = simulator
        self.default_action = np.array(spec.config.default_action)

    @register.states()
    def reset(self):
        self.simulator["input"] = np.squeeze(np.array(self.default_action))

    @register.inputs(tick=Space(shape=(), dtype="int64"),
                     commanded_thrust=Space(dtype="float32"),
                     commanded_attitude=Space(dtype="float32"))
    @register.outputs(action_applied=Space(dtype="float32"))
    def callback(self, t_n: float,
                 tick: Msg,
                 commanded_thrust: Msg,
                 commanded_attitude: Msg,):
        assert isinstance(self.simulator, dict), (
                'Simulator object "%s" is not compatible with this simulation node.' % self.simulator
        )
        u = np.array([np.squeeze(commanded_thrust.msgs[-1].data), np.squeeze(commanded_attitude.msgs[-1].data[0]),
             np.squeeze(commanded_attitude.msgs[-1].data[1])], dtype="float32")
        action = [commanded_thrust.msgs[-1], commanded_attitude.msgs[-1]]
        # Set action in simulator for next step.
        self.simulator["input"] = u

        # Send action that has been applied.
        return dict(action_applied=u)

#### For real engine ###########################################################
class CrazyfliePosition(EngineNode):
    @classmethod
    def make(
            cls,
            name: str,
            rate: float = 60.0,
            process: Optional[int] = eagerx.process.NEW_PROCESS,
            color: Optional[str] = "cyan"
    ) -> NodeSpec:
        """CrazyfliePosition spec"""
        spec = cls.get_specification()

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.color = color
        spec.config.inputs = ["tick"]
        spec.config.outputs = ["observation", "image"]
        return spec
    def initialize(self, spec: NodeSpec, simulator: Any):
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.get_image_callback, queue_size=1, buff_size=10000000)
        self.image = None
        mtx = np.array(
            [610.408, 0.0, 328.45, 0.0, 609.199, 237.006, 0.0, 0.0,
             1.]).reshape(3, 3)
        self.distCoeffs = np.array([0.00, -0.00, -0.00, 0.00])
        h, w = 480, 640
        self.mtx, roi = cv2.getOptimalNewCameraMatrix(
            mtx, self.distCoeffs, (h, w), 0.0, (h, w)
        )
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(aruco_dict, parameters)
        self.step = 0

    @register.states()
    def reset(self):
        got_pos = False
        cv_img = self.image
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        while got_pos == False:
            markerCorners, markerIds, rejectedCandidates = self.detector.detectMarkers(gray)
            if markerIds is not None:
                if [0] in markerIds:
                    got_pos = True
                    markerCorners = markerCorners[np.where(markerIds == [0])[0][0]:np.where(markerIds == [0])[0][0] + 1]
                    markerIds = np.array([[0]])

                    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(markerCorners, 0.1, self.mtx, self.distCoeffs)
                    rvec = rvec[0][0]
                    tvec = tvec[0][0]
                    rmat = cv2.Rodrigues(rvec)[0]
                    self.rot = rmat
                    print("Rot: ", self.rot)
    @register.inputs(tick=Space(dtype="int64"))
    @register.outputs(observation=Space(dtype="float32")
        ,image=Space(dtype="uint8"))
    def callback(self, t_n: float, tick: Optional[Msg] = None):
        got_pos = False
        draw_image = True if int(t_n) % 4 else False
        cv_img = self.image
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        markerCorners, markerIds, rejectedCandidates = self.detector.detectMarkers(gray)
        if markerIds is not None:
            if [0] in markerIds:
                got_pos = True
                markerCorners = markerCorners[np.where(markerIds == [0])[0][0]:np.where(markerIds == [0])[0][0] + 1]
                markerIds = np.array([[0]])

                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 0.035, self.mtx, self.distCoeffs)
                (rvec - tvec).any()  # get rid of that nasty numpy value array error
                pos_x, pos_y, pos_z = tvec[0][0]
                pos = self.rot.T @np.array([pos_x, pos_y, pos_z])
                pos_x, pos_y, pos_z = pos
                pos_x = pos_x
                pos_y = pos_y
                pos_z = pos_z

                if draw_image:
                    for i in range(rvec.shape[0]):
                        frame = cv2.drawFrameAxes(cv_img, self.mtx, self.distCoeffs, rvec[i, :, :], tvec[i, :, :], 0.03)
                        aruco.drawDetectedMarkers(frame, markerCorners)
                    aruco.drawDetectedMarkers(cv_img, markerCorners, markerIds)
                    text_pos = "Position" + str((round(pos_x, 2), round(pos_y, 2), round(pos_z, 2)))
                    cv2.putText(cv_img, text_pos, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        else:
            print("No markers found")
        # cv2.imshow("cv_img", cv_img)
        # cv2.waitKey(1)
        if got_pos:
            pos = np.array([pos_x, pos_y, pos_z], dtype="float32")
            self.last_pos = pos
        else:
            pos = self.last_pos
        if not draw_image:
            cv_img = None
        return dict(observation=pos, image=cv_img)

    def get_image_callback(self, msg):
        cv_img= np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        self.image = cv_img

class CrazyflieOrientation(EngineNode):
    @classmethod
    def make(
            cls,
            name: str,
            rate: float = 60.0,
            process: Optional[int] = eagerx.process.NEW_PROCESS,
            color: Optional[str] = "cyan"
    ) -> NodeSpec:
        """CrazyfliePosition spec"""
        spec = cls.get_specification()

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.color = color
        spec.config.inputs = ["tick"]
        spec.config.outputs = ["observation"]
        return spec

    def initialize(self, spec: NodeSpec, simulator: Any):
        # rospy.init_node("crazyflie_orintation", anonymous=True)
        self.image_sub = rospy.Subscriber("/crazyflie/oriention", Float32MultiArray, self.get_ori_callback)

    @register.states()
    def reset(self):
        self.ori = np.array([0, 0, 0])
        pass

    @register.inputs(tick=Space(dtype="int64"))
    @register.outputs(observation=Space(dtype="float32"))
    def callback(self, t_n: float, tick: Optional[Msg] = None):
        return dict(observation=self.ori)
    def get_ori_callback(self, msg):
        # currently don't care about the orientation of yaw
        # print("ori",msg.data)
        self.ori = np.array([msg.data[0]*np.pi/180., msg.data[1]*np.pi/180., 0])

class CrazyflieInput(EngineNode):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        fixed_delay: float = 0.0,
        process: Optional[int] = eagerx.process.NEW_PROCESS,
        color: Optional[str] = "green",
    ):
        """CrazyflieInput spec"""
        spec = cls.get_specification()

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.color = color
        spec.config.inputs = ["tick", "commanded_thrust", "commanded_attitude"]
        spec.config.outputs = ["action_applied"]
        return spec

    def initialize(self, spec: NodeSpec, simulator: Any):
        # rospy.init_node("crazyflie_input", anonymous=True)
        self.pub = rospy.Publisher("/crazyflie/command", Float32MultiArray, queue_size=10)


    @register.states()
    def reset(self):
        pass

    @register.inputs(tick=Space(shape=(), dtype="int64"),
                     commanded_thrust=Space(dtype="float32"),
                     commanded_attitude=Space(dtype="float32"))
    @register.outputs(action_applied=Space(dtype="float32"))
    def callback(self, t_n: float,
                 tick: Msg,
                 commanded_thrust: Msg,
                 commanded_attitude: Msg,):
        if commanded_thrust is not None:
            u = np.array([np.squeeze(commanded_thrust.msgs[-1].data), np.squeeze(commanded_attitude.msgs[-1].data[0]),
                          np.squeeze(commanded_attitude.msgs[-1].data[1])], dtype="float32")
            action = Float32MultiArray()
            action.data = [commanded_attitude.msgs[-1].data[0],
                           commanded_attitude.msgs[-1].data[1],
                           0,
                           commanded_thrust.msgs[-1].data[0]]
            self.pub.publish(action)
            print("action", action.data)
        return dict(action_applied=u)
class CrazyflieRender(EngineNode):
    @classmethod
    def make(
        cls,
        name: str,
        rate: float,
        process: Optional[int] = process.ENGINE,
        color: Optional[str] = "cyan",
        shape: list = None,
    ):
        """OdeRender spec"""
        spec = cls.get_specification()

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = process
        spec.config.color = color
        spec.config.inputs = [ "image"]
        # spec.config.inputs = ["tick", "image", "action_applied"]
        spec.config.outputs = ["renderimage"]

        # Modify custom node params
        spec.config.shape = shape if isinstance(shape, list) else [480, 640]
        return spec
    def initialize(self, spec: "NodeSpec", simulator: Any):
        pass
    @register.states()
    def reset(self):
        # This sensor is stateless (in contrast to e.g. a Kalman filter).
        pass

    @register.inputs(image=Space(dtype="uint8"))
    @register.outputs(renderimage=Space(dtype="uint8"))
    def callback(self, t_n: float, image: Msg):
        if image is not None:
            print("image", image.msgs[-1].data.shape)
            cv_img = image.msgs[-1].data
            # cv2.imshow("cv_img", cv_img)
            # cv2.waitKey(1)
        else:
            cv_img = np.zeros(self.shape, dtype="uint8")
        return dict(renderimage=cv_img)

    def _set_render_toggle(self, msg):
        self.render_toggle = msg


