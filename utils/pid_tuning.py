import numpy as np
from itertools import product
from src.drone import Drone
from src.pid_controller import  PIDController
from IPython import embed

import utils.formation as formation

def simulate_drone_with_pid(num_agents, initial_positions, pid, target, num_steps=100, dt=0.1, H_rel=None):
    #F = np.array([[1, 0], [0, 1]])
    F = formation.initialize_state_transition_matrix(num_agents, initial_positions)
    #G = np.eye(2)
    G = formation.initialize_state_transition_matrix(num_agents, initial_positions)
    #Q = 0.1 * np.eye(2)
    Q = formation.initialize_process_noise_covariance_matrix(num_agents, initial_positions)
    #H = np.array([[1, 0], [0, 1]])
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]) # simple obs matrix for single pair of robots
    #R = 0.1 * np.eye(2)
    R = formation.initialize_measurement_noise_covariance_matrix(num_agents, measurement_noise_variance=0.1)

    x0 = initial_positions[0]  # Initialize the first drone's position
    state_dim = x0.shape[0]
    P0 = 0.1 * np.eye(state_dim)

    # Use provided H_rel or create a default one for testing
    if H_rel is None:
        H_rel = np.array([[-1, 0, 1, 0], [0, -1, 0, 1]])  # Simplified example

    # consider first drone (arbitrary)
    drone = Drone(1, x0, P0, 
        formation.extract_submatrix(F, 0, state_dim), # F_i
        formation.extract_submatrix(G, 0, state_dim), # G_i
        formation.extract_submatrix(Q, 0, state_dim), # Q_i
        formation.extract_submatrix(H, 0, state_dim), # H_i 
        H_rel, 
        formation.extract_submatrix(R, 0, state_dim),
        pid)
    mse = 0

    for _ in range(num_steps):
        current_position = drone.positions_pred[-1]
        error = target - current_position
        control_output = pid.compute(error, dt)
        u = np.zeros(2)
        u[:2] = control_output
        drone.predict(u)
        mse += np.sum(error ** 2)

    mse /= num_steps
    return mse

def grid_search_pid(num_agents, initial_positions, target, Kp_values, Ki_values, Kd_values, num_steps=100, dt=0.1):
    best_params = None
    best_mse = float('inf')

    for Kp in Kp_values:
        for Ki in Ki_values:
            for Kd in Kd_values:
                pid = PIDController(Kp, Ki, Kd)
                mse = simulate_drone_with_pid(num_agents, initial_positions, pid, target, num_steps, dt)
                if mse < best_mse:
                    best_mse = mse
                    best_params = (Kp, Ki, Kd)

    return best_params, best_mse

