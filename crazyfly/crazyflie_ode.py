from math import sin, cos, pi, sqrt


def crazyflie_ode(x, t, u, mass, gain_constant, time_constant):
    """ The ODE calculations obtained from Jacob Kooi
    source: https://doi.org/10.48550/arXiv.2103.09043
    """
    # States are: [x, y, z, x_dot, y_dot, z_dot, phi, theta, thrust_state]
    # u = [PWM_c, Phi_c, Theta_c] = [10000 to 60000, deg, deg]
    # param = [mass, gain constant, time constant]
    def force_to_pwm(input_force):
        # Just the inversion of pwm_to_force
        a = 4 * 2.130295e-11
        b = 4 * 1.032633e-6
        c = 5.485e-4 - input_force
        d = b ** 2 - 4 * a * c
        pwm = (-b + sqrt(d)) / (2 * a)
        return pwm

    param = [mass, gain_constant, time_constant]
    pwm_commanded = u[0]
    a_ss = -15.4666  # State space A
    b_ss = 1  # State space B
    c_ss = 3.5616e-5  # State space C
    d_ss = 7.2345e-8  # State space AD
    force = 4 * (c_ss * x[8] + d_ss * pwm_commanded)  # Thrust force
    # force *= 0.88                                                  # This takes care of the motor thrust gap sim2real
    pwm_drag = force_to_pwm(force)  # Symbolic PWM to approximate rotor drag
    dragxy = 9.1785e-7 * 4 * (0.04076521 * pwm_drag + 380.8359)  # Fa,xy
    dragz = 10.311e-7 * 4 * (0.04076521 * pwm_drag + 380.8359)  # Fa,z
    # phi_commanded = u[1] * pi / 6  # Commanded phi in radians
    # theta_commanded = u[2] * pi / 6  # Commanded theta in radians
    phi_commanded = u[1] * pi / 180  # Commanded phi in radians
    theta_commanded = u[2] * pi / 180  # Commanded theta in radians
    # dx = [x[3],  # x_dot
    #       x[4],  # y_dot
    #       x[5],  # z_dot
    #       (sin(x[7])) * (force - dragxy * x[3]) / param[0],  # x_ddot
    #       (sin(x[6]) * cos(x[7])) * (force - dragxy * x[4]) / param[0],  # y_ddot
    #       #todo: sign is incorrect
    #       (cos(x[6]) * cos(x[7])) * (force - dragz * x[5]) / param[0] - 9.81,  # z_ddot
    #       (param[1] * phi_commanded - x[6]) / param[2],  # Phi_dot
    #       (param[1] * theta_commanded - x[7]) / param[2],  # Theta_dot
    #       a_ss * x[8] + b_ss * pwm_commanded]  # Thrust_state dot
    dx = [x[3],  # x_dot
          x[4],  # y_dot
          x[5],  # z_dot
          - (sin(x[7])) * (force - dragxy * x[3]) / param[0],  # x_ddot
          - (sin(x[6]) * cos(x[7])) * (force - dragxy * x[4]) / param[0],  # y_ddot
          # todo: sign is incorrect
          (cos(x[6]) * cos(x[7])) * (force - dragz * x[5]) / param[0] - 9.81,  # z_ddot
          (param[1] * phi_commanded - x[6]) / param[2],  # Phi_dot
          (param[1] * theta_commanded - x[7]) / param[2],  # Theta_dot
          a_ss * x[8] + b_ss * pwm_commanded]  # Thrust_state dot
    # print((sin(x[6]) * cos(x[7])),(force - dragxy * x[4]) )
    return dx
