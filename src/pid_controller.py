import numpy as np

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp #assign the proportional gain
        self.Ki = Ki #assign the integral gain
        self.Kd = Kd #assign the derivative gain
        self.integral = np.zeros(2)
        self.prev_error = np.zeros(2)

    def compute(self, error, dt):
        self.integral += error * dt #Updates the integral term by adding the current error multiplied by the time step.
        derivative = (error - self.prev_error) / dt #Updates the previous error term with the current error.
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative
