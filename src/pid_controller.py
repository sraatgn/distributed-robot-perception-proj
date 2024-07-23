import numpy as np


class PIDController:
    def __init__(self, Kp, Ki, Kd, output_limits=(None, None)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = np.zeros(2)
        self.prev_error = np.zeros(2)
        self.output_limits = output_limits

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        # Applica limiti all'output
        if self.output_limits[0] is not None:
            output = np.maximum(output, self.output_limits[0])
        if self.output_limits[1] is not None:
            output = np.minimum(output, self.output_limits[1])

        return output
