import cv2
import numpy as np
def double_pendulum_render_fn(img, observation, action):
    height, width, _ = img.shape
    side_length = min(width, height)
    state = observation.msgs[-1].data

    img += 255
    length = side_length // 5
    sin_theta, cos_theta = np.sin(state[0]), np.cos(state[0])

    if len(action.msgs) > 0:
        u = action.msgs[-1].data[0]
    else:
        u = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Applied Voltage"+str(u)
    text_size = cv2.getTextSize(text, font, 0.5, 2)[0]
    text_x = int((width - text_size[0]) / 2)
    text_y = int(text_size[1])
    img = cv2.putText(img, text, (text_x, text_y), font, 0.5, (0, 0, 0))

    # Draw pendulum
    img = cv2.line(
        img,
        (width // 2, height // 2),
        (width // 2 + int(length * sin_theta), height // 2 + int(length * cos_theta)),
        (0, 0, 255),
        max(side_length // 32, 1),
    )

    # Draw mass
    img = cv2.circle(
        img, (width // 2 + int(length * sin_theta), height // 2 + int(length * cos_theta)), side_length // 24, (0, 0, 0), -1
    )
    sin_theta2, cos_theta2 = np.sin(state[0]+state[1]), np.cos(state[0]+state[1])
    img = cv2.line(
        img,
        (width // 2 + int(length * sin_theta), height // 2 + int(length * cos_theta)),
        (width // 2 + int(length * sin_theta) + int(length * sin_theta2), height // 2 + int(length * cos_theta)+ int(length * cos_theta2)),
        (0, 0, 255),
        max(side_length // 32, 1),
    )
    img = cv2.circle(
        img, (width // 2 + int(length * sin_theta) + int(length * sin_theta2), height // 2 + int(length * cos_theta) + int(length * cos_theta2)), int(0.75*side_length / 24),
        (0, 0, 0), -1
    )
    # Draw velocity vector
    img = cv2.arrowedLine(
        img,
        (width // 2 + int(length * sin_theta) + int(length * sin_theta2), height // 2 + int(length * cos_theta)+ int(length * cos_theta2)),
        (
            width // 2 + int(length * (sin_theta + min(state[2],10) * cos_theta / 10)+ length * (sin_theta2 + min(state[3],10) * cos_theta2 / 10)),
            height // 2 + int(length * (cos_theta + min(state[2],10) * sin_theta / 10) + length * (cos_theta2 + min(state[3],10) * sin_theta2 / 10)),
        ),
        (0, 0, 255),
        max(side_length // 240, 1),
    )
    return img
class psudo_ob:
    def __init__(self,msg_list):
        self.msgs = msg_list

class psudo_msg:
    def __init__(self,state):
        self.data = state

if __name__ == '__main__':
    img = np.zeros((500, 500,3), np.uint8)
    img.fill(255)
    ob = psudo_ob([psudo_msg([-0.05596173,  0.06649979, -0.02689438,  0.05752617])])
    double_pendulum_render_fn(img,ob,None)
    cv2.imshow('image', img)

    cv2.waitKey(0)