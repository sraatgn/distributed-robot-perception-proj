import numpy as np
from itertools import product
from src.drone import Drone
from src.pid_controller import  PIDController
from IPython import embed

def simulate_drone_with_pid(pid, target, num_steps=100, dt=0.1, H_rel=None):
    F = np.array([[1, 0], [0, 1]])
    G = np.eye(2)
    Q = 0.1 * np.eye(2)
    H = np.array([[1, 0], [0, 1]])
    R = 0.1 * np.eye(2)
    x0 = np.array([0, 0])
    P0 = 0.1 * np.eye(2)

    # Use provided H_rel or create a default one for testing
    if H_rel is None:
        H_rel = np.array([[-1, 0, 1, 0], [0, -1, 0, 1]])  # Simplified example

    drone = Drone(1, x0, P0, F, G, Q, H, H_rel, R, pid)
    mse = 0

    for _ in range(num_steps):
        current_position = drone.positions[-1]
        error = target - current_position
        control_output = pid.compute(error, dt)
        u = np.zeros(2)
        u[:2] = control_output
        drone.predict(u)
        mse += np.sum(error ** 2)

    mse /= num_steps
    return mse

def grid_search_pid(target, Kp_values, Ki_values, Kd_values, num_steps=100, dt=0.1):
    best_params = None
    best_mse = float('inf')

    for Kp in Kp_values:
        for Ki in Ki_values:
            for Kd in Kd_values:
                pid = PIDController(Kp, Ki, Kd)
                mse = simulate_drone_with_pid(pid, target, num_steps, dt)
                if mse < best_mse:
                    best_mse = mse
                    best_params = (Kp, Ki, Kd)

    return best_params, best_mse