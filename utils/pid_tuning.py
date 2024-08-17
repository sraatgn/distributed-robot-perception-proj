import numpy as np
from itertools import product
from src.drone import Drone
from src.pid_controller import  PIDController
from IPython import embed
import matplotlib.pyplot as plt
import seaborn as sns

import utils.formation as formation

def simulate_drone_with_pid(num_agents, initial_positions, pid, target, num_steps=100, sensing_range=10, dt=0.1, H_rel=None):
    F = formation.initialize_state_transition_matrix(num_agents, initial_positions)
    G = formation.initialize_state_transition_matrix(num_agents, initial_positions)
    Q = formation.initialize_process_noise_covariance_matrix(num_agents, initial_positions)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]) # simple obs matrix for single pair of robots
    R = formation.initialize_measurement_noise_covariance_matrix(num_agents, measurement_noise_variance=0.1)

    x0 = initial_positions[0]  # Initialize the first drone's position
    state_dim = x0.shape[0]
    P0 = 0.1 * np.eye(state_dim)

    # Use provided H_rel or create a default one for testing
    if H_rel is None:
        H_rel = np.array([[-1, 0, 1, 0], [0, -1, 0, 1]])  # Simplified example

    # consider first drone (arbitrary)
    drone = Drone(1, sensing_range, x0, P0, 
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

def grid_search_pid(num_agents, initial_positions, target, Kp_values, Ki_values, Kd_values, num_steps=100, sensing_range=10, dt=0.1):
    best_params = None
    best_mse = float('inf')
    results = []

    for Kp in Kp_values:
        for Ki in Ki_values:
            for Kd in Kd_values:
                pid = PIDController(Kp, Ki, Kd)
                mse = simulate_drone_with_pid(num_agents, initial_positions, pid, target, num_steps, sensing_range, dt)
                results.append((Kp, Ki, Kd, mse))
                if mse < best_mse:
                    best_mse = mse
                    best_params = (Kp, Ki, Kd)

    return best_params, best_mse, results

## PLOTS
def plot_scatter(results):
    Kp_vals = [result[0] for result in results]
    Ki_vals = [result[1] for result in results]
    Kd_vals = [result[2] for result in results]
    mse_vals = [result[3] for result in results]

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    sc1 = ax[0].scatter(Kp_vals, mse_vals, c=Kd_vals, cmap='viridis')
    ax[0].set_xlabel('Kp')
    ax[0].set_ylabel('MSE')
    ax[0].set_title('MSE vs Kp')
    fig.colorbar(sc1, ax=ax[0], label='Kd')

    sc2 = ax[1].scatter(Ki_vals, mse_vals, c=Kd_vals, cmap='viridis')
    ax[1].set_xlabel('Ki')
    ax[1].set_ylabel('MSE')
    ax[1].set_title('MSE vs Ki')
    fig.colorbar(sc2, ax=ax[1], label='Kd')

    sc3 = ax[2].scatter(Kd_vals, mse_vals, c=Ki_vals, cmap='viridis')
    ax[2].set_xlabel('Kd')
    ax[2].set_ylabel('MSE')
    ax[2].set_title('MSE vs Kd')
    fig.colorbar(sc3, ax=ax[2], label='Ki')

    plt.show()

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def plot_3d_surface(results, Kp_values, Ki_values, Kd_values):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    Kp_vals = []
    Ki_vals = []
    Kd_vals = []
    mse_vals = []

    for result in results:
        Kp_vals.append(result[0])
        Ki_vals.append(result[1])
        Kd_vals.append(result[2])
        mse_vals.append(result[3])

    sc = ax.scatter(Kp_vals, Ki_vals, Kd_vals, c=mse_vals, cmap=cm.viridis)
    ax.set_xlabel('Kp')
    ax.set_ylabel('Ki')
    ax.set_zlabel('Kd')
    plt.colorbar(sc, ax=ax, label='MSE')
    plt.title('3D Surface plot of PID parameter tuning')
    plt.show()

def plot_heatmaps(results, Kp_values, Ki_values, Kd_values):
    for Kd in Kd_values:
        mse_matrix = np.zeros((len(Kp_values), len(Ki_values)))

        for Kp_idx, Kp in enumerate(Kp_values):
            for Ki_idx, Ki in enumerate(Ki_values):
                for result in results:
                    if result[0] == Kp and result[1] == Ki and result[2] == Kd:
                        mse_matrix[Kp_idx, Ki_idx] = result[3]

        plt.figure(figsize=(10, 8))
        sns.heatmap(mse_matrix, xticklabels=Ki_values, yticklabels=Kp_values, annot=True, fmt=".4g", cmap="viridis")
        plt.title(f'Heatmap for Kd = {Kd}')
        plt.xlabel('Ki')
        plt.ylabel('Kp')
        plt.show()









