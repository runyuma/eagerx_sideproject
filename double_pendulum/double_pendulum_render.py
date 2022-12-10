import cv2
import numpy as np
def double_pendulum_render_fn(img, observation, action):
    height, width, _ = img.shape
    side_length = min(width, height)
    state = observation.msgs[-1].data

    img += 255
    length = side_length // 5
    sin_theta, cos_theta = np.sin(state[0]), np.cos(state[0])

    # Draw pendulum
    img = cv2.line(
        img,
        (width // 2, height // 2),
        (width // 2 + int(length * sin_theta), height // 2 - int(length * cos_theta)),
        (0, 0, 255),
        max(side_length // 32, 1),
    )

    # Draw mass
    img = cv2.circle(
        img, (width // 2 + int(length * sin_theta), height // 2 - int(length * cos_theta)), side_length // 24, (0, 0, 0), -1
    )
    sin_theta2, cos_theta2 = np.sin(state[0]+state[1]), np.cos(state[0]+state[1])
    img = cv2.line(
        img,
        (width // 2 + int(length * sin_theta), height // 2 - int(length * cos_theta)),
        (width // 2 + int(length * sin_theta) + int(length * sin_theta2), height // 2 - int(length * cos_theta)+ int(length * cos_theta2)),
        (0, 0, 255),
        max(side_length // 32, 1),
    )
    img = cv2.circle(
        img, (width // 2 + int(length * sin_theta) + int(length * sin_theta2), height // 2 - int(length * cos_theta)+ int(length * cos_theta2)), int(0.75*side_length / 24),
        (0, 0, 0), -1
    )
    # Draw velocity vector
    img = cv2.arrowedLine(
        img,
        (width // 2 + int(length * sin_theta) + int(length * sin_theta2), height // 2 - int(length * cos_theta)+ int(length * cos_theta2)),
        (
            width // 2 + int(length * (sin_theta + state[2] * cos_theta / 10)+ length * (sin_theta2 + state[3] * cos_theta2 / 10)),
            height // 2 + int(length * (-cos_theta + state[2] * sin_theta / 10) + length * (-cos_theta2 + state[3] * sin_theta2 / 10)),
        ),
        (0, 0, 255),
        max(side_length // 240, 1),
    )
class psudo_ob:
    def __init__(self,msg_list):
        self.msgs = msg_list

class psudo_msg:
    def __init__(self,state):
        self.data = state

if __name__ == '__main__':
    img = np.zeros((500, 500,3), np.uint8)
    img.fill(255)
    ob = psudo_ob([psudo_msg([1,1,1,1])])
    double_pendulum_render_fn(img,ob,None)
    cv2.imshow('image', img)

    cv2.waitKey(0)