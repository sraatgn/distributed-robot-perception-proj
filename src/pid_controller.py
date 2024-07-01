import numpy as np

class PIDController:
    ''' Returns PID control output '''
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp    # proportional gain
        self.Ki = Ki    # integral gain
        self.Kd = Kd    # derivative gain
        self.integral = np.zeros(2)
        self.prev_error = np.zeros(2)

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative