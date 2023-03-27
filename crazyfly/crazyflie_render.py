import cv2
import numpy as np
def crazyflie_render_fn(img, observation, action):
    height, width, _ = img.shape
    side_length = min(width, height)
    state = observation.msgs[-1].data

    if len(action.msgs) > 0:
        act = action.msgs[-1]
    else:
        act = None
    img += 255
    pos_x, pos_y, pos_z = state[0], state[1], state[2]
    velx, vely, velz = state[3], state[4], state[5]
    roll, pitch, yaw = state[6], state[7], 0
    text_pos = "Position" + str((round(pos_x,2), round(pos_y,2), round(pos_z,2)))
    text_vel = "Velocity" + str((round(velx,2), round(vely,2), round(velz,2)))
    cv2.putText(img, text_pos, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
    cv2.putText(img, text_vel, (20+side_length//2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    if act is not None:
        text_act = "Action" + str((int(act.data[0]), round(act.data[1],2), round(act.data[2],2)))
        cv2.putText(img, text_act, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    default_height = -0.5
    R = rotation_matrix(roll, pitch, yaw)
    pos = np.array([pos_x, pos_y, pos_z])
    pos = [0,0,1]
    arm = 0.1*np.array([[1, 0, 0],
                    [0, 1, 0],
                    [-1, 0, 0],
                    [0, -1, 0]
                    ])
    arm_pos = pos + (R @ arm.T).T
    ratio = side_length//2

    # X,Z dimension
    img = cv2.circle(
        img, (-int(pos[1] * ratio)+side_length//2, height - int(pos[2]* ratio) ),
        4,
        (0, 0, 0),
        -1
    )
    img = cv2.arrowedLine(
        img,(-int(pos[1] * ratio)+side_length//2, height - int(pos[2]* ratio) ),
        (int(-vely*100)-int(pos[1] * ratio)+side_length//2, height - int(velz*100) - int(pos[2]* ratio) ),
        (0, 255, 0),
        2
    )
    img = cv2.arrowedLine(
        img, (-int(pos[1] * ratio) + side_length // 2, height - int(pos[2] * ratio)),
        (int(pos_y * ratio) - int(pos[1] * ratio) + side_length // 2, height + int((pos_z+0.5) * ratio) - int(pos[2] * ratio)),
        (255, 255, 0),
        3
    )
    for i in range(4):
        img = cv2.circle(
            img, (-int(arm_pos[i][1]*ratio)+side_length//2, height-int(arm_pos[i][2]*ratio)),
            6,
            (0, 0, 0),
            -1
        )
        img = cv2.line(
            img,
            (-int(arm_pos[i][1]*ratio)+side_length//2, height-int(arm_pos[i][2]*ratio)),
            (-int(pos[1] * ratio)+side_length//2, height - int(pos[2]* ratio) ),
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
    ob = psudo_ob([psudo_msg([0.5, 0., 1., 0., 0., 0., 0, 0.5, 0.])])
    crazyflie_render_fn(img,ob,None)
    cv2.imshow('image', img)

    cv2.waitKey(0)