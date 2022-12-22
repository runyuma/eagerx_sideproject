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
    J1=0.037,
    J2=0.000111608131930852,
    m1=0.18,
    m2=0.0691843934004535,
    l1=0.1,
    l2=0.1,
    b1=0.975872107940422,
    b2=1.07098956449896e-05,
    c1=0.06,
    c2=0.0185223578523340,
    K = 1.09724557347983
):
    g = 9.81
    P3 = m2 * l1 * c2
    F1 = (m1*c1 + m2*l1)*g
    F2 = m2*c2*g
    alpha1 = x[0]
    alpha2 = x[1]
    alpha_dot1 = x[2]
    alpha_dot2 = x[3]
    M = np.array([[J1 + J2 + m2 * l1 * l1 + 2 * P3 * cos(alpha2), J2 + P3 * cos(alpha2)],
                  [J2 + P3 * cos(alpha2), J2]])
    C = np.array([[b1-P3*alpha_dot2*sin(alpha2),-P3*(alpha_dot2)*sin(alpha2)],
                  [P3*alpha_dot1*sin(alpha2),b2]])
    G = np.array([[-F1*sin(alpha1)-F2*sin(alpha1+alpha2)],
                  [-F2*sin(alpha1+alpha2)]])
    U = np.array([[K*u],
                  [0]])
    alpha_dot = np.array([[alpha_dot1],
                  [alpha_dot2]])
    # Minv = 1/(np.linalg.det(M)+0.000001)*np.array([[J2, -(J2 + P3 * cos(alpha2))],
    #                                             [-(J2 + P3 * cos(alpha2)), J1 + J2 + 2 * P3 * cos(alpha2)]])
    Minv = np.linalg.inv(M)
    totoal_torque = U + G
    coli = C@alpha_dot
    ddx = Minv@(totoal_torque - coli)
    ddx1 = ddx.tolist()[0][0]
    ddx2 = ddx.tolist()[1][0]
    # print(x,[x[2], x[3], ddx1, ddx2],u,np.linalg.det(M))
    # print("force",totoal_torque,coli)
    return [x[2], x[3], ddx1, ddx2]


def double_pendulum_dfun(
    x,
    t,
    u,
    J1=0.037,
    J2=0.000111608131930852,
    m1=0.18,
    m2=0.0691843934004535,
    l1=0.1,
    l2=0.1,
    b1=0.975872107940422,
    b2=1.07098956449896e-05,
    c1=0.06,
    c2=0.0185223578523340,
    K = 1.09724557347983
):
    g = 9.81
    P3 = m2 * l1 * c2
    F1 = (m1 * c1 + m2 * l2) * g
    F2 = m2 * c2 * g
    alpha1 = x[0]
    alpha2 = x[1]
    alpha_dot1 = x[2]
    alpha_dot2 = x[3]
    M = np.array([[J1 + J2 + 2 * P3 * cos(alpha2), J2 + P3 * cos(alpha2)],
                  [J2 + P3 * cos(alpha2), J2]])
    # as P3 is extremely small, we can ignore the derivative of P3
    Dfun = np.array([[0,0,1,0],
                     [0,0,0,1],
                     [0,0,0,0],
                     [0,0,0,0]])
    daddda = np.array([[0,0,0,0],
                     [0,0,0,0]])
    daddda[:,0:1] = np.array([[-F1*cos(alpha1)-F2*cos(alpha1+alpha2)],
                              [-F2*cos(alpha1+alpha2)]])
    daddda[:, 1:2] = np.array([[- F2 * cos(alpha1 + alpha2) + 2*P3*alpha_dot1*alpha_dot2*cos(alpha2) + P3*alpha_dot2*alpha_dot2*cos(alpha2)],
                               [-F2 * cos(alpha1 + alpha2)- P3*alpha_dot1*alpha_dot1*cos(alpha2)]])
    daddda[:, 2:3] = - np.array([[b1 - 2 * P3 * alpha_dot2 * sin(alpha2)],
                               [2 * P3 * alpha_dot1 * sin(alpha2)]])
    daddda[:, 3:4] = - np.array([[- 2 * P3 * alpha_dot1 * sin(alpha2) - 2 * P3 * alpha_dot2 * sin(alpha2)],
                               [b2]])
    Minv = 1 / (np.linalg.det(M) + 0.00001) * np.array([[J2, -(J2 + P3 * cos(alpha2))],
                                                        [-(J2 + P3 * cos(alpha2)), J1 + J2 + 2 * P3 * cos(alpha2)]])
    daddda = Minv@daddda
    Dfun[2:4,:] = daddda
    res = Dfun.tolist()
    return res
