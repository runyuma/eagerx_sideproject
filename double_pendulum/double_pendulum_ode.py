from math import sin, cos, exp

import numpy as np


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + exp(-x))
    else:
        return exp(x) / (1.0 + exp(x))


def sigmoid_d(x):
    if x >= 0:
        return exp(-x) / (exp(-x) + 1) ** 2
    else:
        return exp(x) / (exp(x) + 1) ** 2


def double_pendulum_ode(
    x,
    t,
    u,
    J1=0.0667,
    J2=0.0427,
    m1=1.25,
    m2=0.8,
    l1=0.4,
    l2=0.4,
    b1=0.08,
    b2=0.02,
    c1=0.2,
    c2=0.2,
):
    g = 9.81
    P1 = J1 + m1*c1*c1 + m2*l1*l1
    P2 = J2 + m2 * c2 * c2
    P3 = m2 * l1 * c2
    F1 = (m1*c1 + m2*l2)*g
    F2 = m2*c2*g
    alpha1 = x[0]
    alpha2 = x[1]
    alpha_dot1 = x[2]
    alpha_dot2 = x[3]
    M = np.array([[P1+P2+2*P3*cos(alpha2),P2+P3*cos(alpha2)],
                  [P2+P3*cos(alpha2),P2]])
    C = np.array([[b1-P3*alpha_dot2*sin(alpha2),-P3*(alpha_dot1+alpha_dot2)*sin(alpha2)],
                  [P3*alpha_dot1*sin(alpha2),b2]])
    G = np.array([[-F1*sin(alpha1)-F2*sin(alpha1+alpha2)],
                  [-F2*F2*sin(alpha1+alpha2)]])
    U = np.array([[0],
                  [u]])
    alpha = np.array([[alpha1],
                  [alpha2]])
    ddx = np.linalg.inv(M)@(U - G - C@alpha)
    ddx1 = ddx.tolist()[0][0]
    ddx2 = ddx.tolist()[1][0]
    return [x[2], x[3], ddx1, ddx2]


def double_pendulum_dfun(
    x,
    t,
    u,
    J1=0.0667,
    J2=0.0427,
    m1=1.25,
    m2=0.8,
    l1=0.4,
    l2=0.4,
    b1=0.08,
    b2=0.02,
    c1=0.2,
    c2=0.2,
):
    g = 9.81
    P1 = J1 + m1 * c1 * c1 + m2 * l1 * l1
    P2 = J2 + m2 * c2 * c2
    P3 = m2 * l1 * c2
    F1 = (m1 * c1 + m2 * l2) * g
    F2 = m2 * c2 * g
    alpha1 = x[0]
    alpha2 = x[1]
    alpha_dot1 = x[2]
    alpha_dot2 = x[3]
    M = np.array([[P1 + P2 + 2 * P3 * cos(alpha2), P2 + P3 * cos(alpha2)],
                  [P2 + P3 * cos(alpha2), P2]])
    Dfun = np.array([[0,0,1,0],
                     [0,0,0,1],
                     [0,0,0,0],
                     [0,0,0,0]])
    daddda = np.array([[0,0,0,0],
                     [0,0,0,0]])
    daddda[:,0:1] = np.array([[-F1*cos(alpha1)-F2*cos(alpha1+alpha2)],
                              [-F2*cos(alpha1+alpha2)]])
    daddda[:, 1:2] = np.array([[- F2 * cos(alpha1 + alpha2)],
                               [-F2 * cos(alpha1 + alpha2)]])
    daddda[:, 2:3] = np.array([[b1 - 2 * P3 * alpha_dot2 * sin(alpha2)],
                               [2 * P3 * alpha_dot1 * sin(alpha2)]])
    daddda[:, 3:4] = np.array([[- 2 * P3 * alpha_dot1 * sin(alpha2) - 2 * P3 * alpha_dot2 * sin(alpha2)],
                               [b2]])
    daddda = np.linalg.inv(M)@daddda
    Dfun[2:4,:] = daddda
    res = Dfun.tolist()
    return res
