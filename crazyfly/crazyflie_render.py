import cv2
import numpy as np
from eagerx import register, Space
from eagerx.core.specs import NodeSpec
from eagerx.utils.utils import Msg
import eagerx
import rospy
class Overlay(eagerx.Node):
    @classmethod
    def make(
            cls,
            name: str,
            rate: float,
            process: int = eagerx.NEW_PROCESS,
            color: str = "cyan",
            engine: str = "real",
    ) -> NodeSpec:
        """Overlay spec"""
        # Get base parameter specification with defaults parameters
        spec = cls.get_specification()
        # Adjust default params
        spec.config.update(name=name, rate=rate, process=process, color=color)
        spec.config.update(inputs=["base_image", "commanded_thrust", "commanded_attitude", "pos", "orientation"], outputs=["image"], engine_type=engine)
        return spec

    def initialize(self, spec: NodeSpec):
        #camera matrix

        mtx = np.array(
            [610.408, 0.0, 328.45, 0.0, 609.199, 237.006, 0.0, 0.0,
             1.]).reshape(3, 3)
        self.distCoeffs = np.array([0.00, -0.00, -0.00, 0.00])
        self.h, self.w = 480, 640
        self.mtx, roi = cv2.getOptimalNewCameraMatrix(
            mtx, self.distCoeffs, (self.h, self.w), 0.0, (self.h, self.w)
        )
        self.odemtx = np.array([-640,0,0,0,640,0,]).reshape(2,3)
        self.odebias = np.array([320,320])
        self.engine_type = spec.config.engine_type

    @register.states()
    def reset(self):
        # point list
        self.path = []
        self.rot_mat = None

    @register.inputs(
    base_image=Space(dtype="uint8"),
     commanded_thrust=Space(dtype="float32"),
     commanded_attitude=Space(dtype="float32"),
    pos=Space(low=[-2, -2, -2], high=[2, 2, 0], shape=(3,), dtype="float32"),
    orientation=Space(low=[-1, -1, -1], high=[1, 1, 1], shape=(3,), dtype="float32"),
    )
    @register.outputs(image=Space(dtype="uint8"))
    def callback(self, t_n: float, base_image: Msg, commanded_thrust: Msg, commanded_attitude: Msg, pos: Msg, orientation: Msg):
        if self.engine_type == "real":
            if self.rot_mat is None:
                rot_mat = rospy.get_param("/crazyflie/rot")
                bias = rospy.get_param("/crazyflie/bias")
                self.rot_mat = np.array(rot_mat).reshape(3,3)
                self.bias = np.array(bias).reshape(3)
                print("rotmat###########################################",self.rot_mat)
        if len(base_image.msgs[-1].data) > 0:

            img = base_image.msgs[-1]
            if img.shape == (480, 640, 3):# Real engine
                self.path.append(self.mtx @np.linalg.inv(self.rot_mat.T)@ (pos.msgs[-1].data+self.bias))
                # self.path.append(self.mtx @ pos.msgs[-1].data)
                # print(self.mtx @np.linalg.inv(self.rot_mat.T)@ pos.msgs[-1].data)
                for point in self.path:
                    u, v = int(point[0] / point[2]), int(point[1] / point[2])
                    # u, v = cv2.projectPoints(point, orientation.msgs[-1].data, pos.msgs[-1].data, self.mtx, self.distCoeffs)[0][0]
                    # print(u, v)
                    img = cv2.circle(img, (u, v), 2, (0, 255, 0), 1)
            elif img.shape == (640, 640, 3):# ODE engine
                self.path.append(self.odemtx @ pos.msgs[-1].data+self.odebias)
                for point in self.path:
                    u, v = int(point[0]), int(point[1])
                    # u, v = cv2.projectPoints(point, orientation.msgs[-1].data, pos.msgs[-1].data, self.mtx, self.distCoeffs)[0][0]
                    # print(u, v)
                    img = cv2.circle(img, (u, v), 2, (0, 255, 0), 1)

            img2 = 255+img[:,:self.w//2]*0
            img = np.concatenate((img, img2), axis=1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            thrust = commanded_thrust.msgs[-1].data[0] if commanded_thrust else 0
            roll,pitch = commanded_attitude.msgs[-1].data[0], commanded_attitude.msgs[-1].data[1] if commanded_attitude else 0


            text = "Thrust: " + str(int(thrust/1000))+"k"
            text_x, text_y = self.w//4+self.w - 100, 75
            img = cv2.putText(img, text, (text_x, text_y), font, 1, (200, 0, 0))

            p1 = (self.w//4+self.w, 100)
            p2 = (self.w//4+self.w+int((thrust-25000)/250), 150)
            img = cv2.rectangle(
                img,
                (self.w//4+self.w - 100, 100),
                (self.w//4+self.w + 100, 150),
                (125, 125, 125),
                -1,
            )
            img = cv2.rectangle(img, p1, p2, (150, 0, 0), -1)

            text = "Roll: " + str(round(roll,2))
            text_x, text_y = self.w//4+self.w - 100, 175
            img = cv2.putText(img, text, (text_x, text_y), font, 1, (200, 0, 0))
            img = cv2.rectangle(
                img,
                (self.w//4+self.w - 100, 200),
                (self.w//4+self.w + 100, 250),
                (125, 125, 125),
                -1,
            )
            p1 = (self.w//4+self.w, 200)
            p2 = (self.w//4+self.w+int((roll)*3), 250)
            img = cv2.rectangle(img, p1, p2, (150, 0, 0), -1)

            text = "Pitch: " + str(round(pitch,2))
            text_x, text_y = self.w//4+self.w - 100, 275
            img = cv2.putText(img, text, (text_x, text_y), font, 1, (200, 0, 0))
            img = cv2.rectangle(
                img,
                (self.w//4+self.w - 100, 300),
                (self.w//4+self.w + 100, 350),
                (125, 125, 125),
                -1,
            )
            p1 = (self.w//4+self.w, 300)
            p2 = (self.w//4+self.w+int((pitch)*3), 350)
            img = cv2.rectangle(img, p1, p2, (150, 0, 0), -1)
            img = cv2.resize(img, (0, 0), fx=0.7, fy=0.7)
            return dict(image=img)
        else:
            return dict(image=np.zeros((0, 0, 3), dtype="uint8"))
def crazyflie_render_fn(img, observation, action):
    height, width, _ = img.shape
    side_length = min(width, height)
    state = observation.msgs[-1].data
    if action is not None:
        if len(action.msgs) > 0:
            act = action.msgs[-1]
        else:
            act = None
    else:
        act = None
    img += 255
    pos_x, pos_y, pos_z = state[0], state[1], state[2]
    velx, vely, velz = state[3], state[4], state[5]
    roll, pitch, yaw = state[6], state[7], 0
    text_pos = "Position" + str((round(pos_x,2), round(pos_y,2), round(pos_z,2)))
    text_vel = "Velocity" + str((round(velx,2), round(vely,2), round(velz,2)))
    cv2.putText(img, text_pos, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(img, text_vel, (20+side_length//2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if act is not None:
        text_act = "Action" + str((int(act.data[0]), round(act.data[1],2), round(act.data[2],2)))
        cv2.putText(img, text_act, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    default_height = -0.5
    R = rotation_matrix(roll, pitch, yaw)
    pos = np.array([pos_x, pos_y, pos_z])
    # pos = [0,0,1]
    arm = 0.05*np.array([[1, 0, 0],
                    [0, 1, 0],
                    [-1, 0, 0],
                    [0, -1, 0]
                    ])
    arm_pos = pos + (R @ arm.T).T
    ratio = side_length

    ## X,Z dimension
    # img = cv2.circle(
    #     img, (-int(pos[1] * ratio)+side_length//2, height - int(pos[2]* ratio) ),
    #     4,
    #     (0, 0, 0),
    #     -1
    # )
    # img = cv2.arrowedLine(
    #     img,(-int(pos[1] * ratio)+side_length//2, height - int(pos[2]* ratio) ),
    #     (int(-vely*100)-int(pos[1] * ratio)+side_length//2, height - int(velz*100) - int(pos[2]* ratio) ),
    #     (0, 255, 0),
    #     2
    # )
    # img = cv2.arrowedLine(
    #     img, (-int(pos[1] * ratio) + side_length // 2, height - int(pos[2] * ratio)),
    #     (int(pos_y * ratio) - int(pos[1] * ratio) + side_length // 2, height + int((pos_z+0.5) * ratio) - int(pos[2] * ratio)),
    #     (255, 255, 0),
    #     3
    # )
    # for i in range(4):
    #     img = cv2.circle(
    #         img, (-int(arm_pos[i][1]*ratio)+side_length//2, height-int(arm_pos[i][2]*ratio)),
    #         6,
    #         (0, 0, 0),
    #         -1
    #     )
    #     img = cv2.line(
    #         img,
    #         (-int(arm_pos[i][1]*ratio)+side_length//2, height-int(arm_pos[i][2]*ratio)),
    #         (-int(pos[1] * ratio)+side_length//2, height - int(pos[2]* ratio) ),
    #         (0, 0, 255),
    #         2,
    #     )
    ## X,Y dimension
    img = cv2.circle(
        img, (-int(pos[0] * ratio)+side_length//2, int(pos[1] * ratio)+side_length//2 ),
        4,
        (0, 0, 0),
        -1
    )
    img = cv2.arrowedLine(
        img,(-int(pos[0] * ratio)+side_length//2, int(pos[1] * ratio)+side_length//2 ),
        (int(-velx*100)-int(pos[0] * ratio)+side_length//2, int(vely*100)+int(pos[1] * ratio)+side_length//2 ),
        (0, 255, 0),
        2
    )
    img = cv2.arrowedLine(
        img, (-int(pos[0] * ratio)+side_length//2, int(pos[1] * ratio)+side_length//2),
        (side_length//2, side_length//2),
        (255, 255, 0),
        3
    )
    for i in range(4):
        img = cv2.circle(
            img, (-int(arm_pos[i][0]*ratio)+side_length//2, int(arm_pos[i][1]*ratio)+side_length//2),
            6,
            (0, 0, 0),
            -1
        )
        img = cv2.line(
            img,
            (-int(arm_pos[i][0]*ratio)+side_length//2, int(arm_pos[i][1]*ratio)+side_length//2),
            (-int(pos[0] * ratio)+side_length//2, int(pos[1] * ratio)+side_length//2 ),
            (0, 0, 255),
            2,
        )
    return img
def rotation_matrix(roll, pitch, yaw):
    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])
    pitch = -pitch
    R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
    R_roll = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
    R = R_yaw @ R_roll @ R_pitch
    return R
class psudo_ob:
    def __init__(self,msg_list):
        self.msgs = msg_list

class psudo_msg:
    def __init__(self,state):
        self.data = state

if __name__ == '__main__':
    img = np.zeros((800, 800,3), np.uint8)
    img.fill(255)
    ob = psudo_ob([psudo_msg([0.5, 0.5, 1., 0.3, 0.8, 0., 0, 0, 0.])])
    crazyflie_render_fn(img,ob,None)
    cv2.imshow('image', img)

    cv2.waitKey(0)