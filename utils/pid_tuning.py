import numpy as np
from itertools import product
from src.drone import Drone
from src.pid_controller import PIDController

def simulate_drone_with_pid(pid, target, num_steps=100, dt=0.1):
    F = np.array([[1, 0, 0.1, 0], [0, 1, 0, 0.1], [0, 0, 1, 0], [0, 0, 0, 1]])
    G = np.eye(4)
    Q = 0.1 * np.eye(4)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    R = 0.1 * np.eye(2)
    x0 = np.array([0, 0, 0, 0])
    P0 = 0.1 * np.eye(4)

    drone = Drone(1, x0, P0, F, G, Q, H, R, pid)
    mse = 0

    for _ in range(num_steps):
        current_position = drone.positions[-1]
        error = target - current_position
        control_output = pid.compute(error, dt)
        u = np.zeros(4)
        u[:2] = control_output
        drone.predict(u)
        mse += np.sum(error ** 2)

    mse /= num_steps
    return mse

def grid_search_pid(target, Kp_values, Ki_values, Kd_values, num_steps=100, dt=0.1):
    best_params = None
    best_mse = float('inf')

    for Kp, Ki, Kd in product(Kp_values, Ki_values, Kd_values):
        pid = PIDController(Kp, Ki, Kd)
        mse = simulate_drone_with_pid(pid, target, num_steps, dt)
        if mse < best_mse:
            best_mse = mse
            best_params = (Kp, Ki, Kd)
        print(f"Tested Kp={Kp}, Ki={Ki}, Kd={Kd}, MSE={mse}")

    return best_params, best_mse



"""
simulate_drone_with_pid: Simulates a drone controlled by a PID controller and calculates the Mean Squared Error (MSE) between the drone's position and a target position over a number of steps.
grid_search_pid: Performs a grid search over specified ranges of PID parameters to find the combination that results in the lowest MSE.
"""