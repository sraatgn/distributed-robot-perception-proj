import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.linalg import sqrtm
from itertools import product
import random
from IPython import embed

from src.drone import Drone
from src.pid_controller import PIDController
from utils.metrics import calculate_rmse, calculate_detection_metrics, calculate_nmse, compute_rmse
import utils.plotting as plots 
from utils.pid_tuning import simulate_drone_with_pid, grid_search_pid, plot_scatter, plot_3d_surface, plot_heatmaps
import utils.formation as formation
from utils.measurements import simulate_measurement


def simulate_fire_detection(
    num_agents=4,
    sensing_range=10,
    num_iterations=100,
    dt = 1.0,                   # Tempo tra i passi della simulazione
    formation_type='circle',
    formation_radius=5, 
    fire_position = np.array([15, 15]),
    sim_communication_loss=False, 
    random_iter=False
):

    random.seed(74)  # Ensure reproducibility
    np.random.seed(74)  # Seed for numpy random functions

    assert num_agents >= 2, f"Sir, this is a distributed system. Why would you use {num_agents} drone(s)?"
    measurements_history = []

    # Inizializzazione droni (state x and covariance P)
    initial_positions = [np.array([np.random.uniform(0, 30), np.random.uniform(0, 30)]) for _ in range(num_agents)]
    state_dim = initial_positions[0].shape[0]
    initial_covariances = [0.1 * np.eye(state_dim) for _ in range(num_agents)]
    
    # Parametri del modello
    F = formation.initialize_state_transition_matrix(num_agents, initial_positions)
    # init G with the same func as F: the process noise affects the state components directly
    G = formation.initialize_state_transition_matrix(num_agents, initial_positions)
    Q = formation.initialize_process_noise_covariance_matrix(num_agents, initial_positions, default_variance=[0.1, 0.5])
    H_rel = formation.initialize_relative_observation_matrix(num_agents, initial_positions)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]) # simple obs matrix for single pair of robots
    R = formation.initialize_measurement_noise_covariance_matrix(num_agents, measurement_noise_variance=0.1)

    # Parametri del PID Controller
    Kp_values = np.arange(0.5, 1.5, 0.25)
    Ki_values = np.arange(0.05, 0.15, 0.05)
    Kd_values = np.arange(0.02, 0.07, 0.02)
    best_params, best_mse, results = grid_search_pid(num_agents, initial_positions, fire_position, Kp_values, Ki_values, Kd_values, num_iterations, sensing_range, dt)
    print(f"Best PID parameters: Kp={best_params[0]}, Ki={best_params[1]}, Kd={best_params[2]}, MSE={best_mse}")
    pid_controller = PIDController(*best_params)

    # Inizializzazione droni
    drones = [Drone(
        i+1, 
        sensing_range,
        initial_positions[i], 
        initial_covariances[i], 
        formation.extract_submatrix(F, i, state_dim), # F_i
        formation.extract_submatrix(G, i, state_dim), # G_i
        formation.extract_submatrix(Q, i, state_dim), # Q_i
        formation.extract_submatrix(H, i, state_dim), # H_i
        H_rel,
        formation.extract_submatrix(R, i, state_dim), # R_i
        pid_controller)
        for i in range(num_agents)]

    for drone in drones:
        drone.neighbors = [d for d in drones if d.id != drone.id]

    # Offset formazione
    formation_offsets = formation.calculate_formation_offsets(formation_type, num_agents, formation_radius)

    for k in range(num_iterations):  
        drones_shf = list(drones) # duplicate for shuffling (keep original ordered list)
        if random_iter:
            random.shuffle(drones_shf)
        # else iterate on the ordered list (copy)

        for drone in drones_shf:
            u = formation.compute_control_input(drone, fire_position, formation_offsets, dt)
            # Simulate true positions with noise (for RMSE)
            drone.true_positions.append(drone.x[:2] + np.random.normal(0, 0.2, size=2))  

            ## PREDICTION
            pred_state = drone.predict(u)
        
        measurement_taken = False
        for drone_i in drones_shf:
            if measurement_taken:
                break # no more than one interim master per iteration

            # create list without i-th drone to iterate on pairs
            other_drones = [drone for drone in drones_shf if drone.id != drone_i.id]
            for drone_j in other_drones:
                distance = np.linalg.norm(drone_i.x[:2] - drone_j.x[:2])
                if distance < sensing_range:
                    # Relative measurement: eucledian distance 
                    z = simulate_measurement(drone_i.x, drone_j.x, drone_i.R)
                    # Become interim master 
                    interim_master = drone_i
                    interim_master.update(z, H, interim_master.R, drone_j) #H_rel
                    # update for other drones is called in interim_master.update()
                    print(f"Relative measurement at step {k} between Drone {drone_i.id} and Drone {drone_j.id}")
                    
                    measurement_taken = True
                    measurements_history.append((drone_i.id, drone_j.id))
                    break

        if measurement_taken == False:
            for drone in drones:
                pass # update = prediction
        
        for drone in drones:
            if drone.detect_fire(fire_position):
                print(f"Fire detected by Drone {drone.id} at position {drone.x[:2]}")

        ## COMMUNICATION LOSS
        if sim_communication_loss:
            if k == 25:
                print("Simulating communication loss for Drone 3")
                drones[2].active = False
            if k == 35:
                print("Simulating recovery for Drone 3")
                drones[2].active = True
    
    ## PLOT TRAJECTORIES or ANIMATION: uncomment to execute
    #plots.animate_simulation(drones, fire_position)
    plots.animate_simulation_expected_traj(drones, fire_position)
    #plots.plot_trajectories(drones, fire_position, plot_pred=False)
    #plots.plot_trajectories(drones, fire_position)
    #plots.visualize_measurement_network(measurements_history, num_agents)

    print(f"PID best parameters selected: Kp={best_params[0]}, Ki={best_params[1]}, Kd={best_params[2]}, MSE={best_mse}")
    print("---------------------------------------")

    return drones, num_iterations



if __name__ == "__main__":

    drones, num_iterations = simulate_fire_detection(sim_communication_loss=False, random_iter=False)

    ## PLOTS & EVALUATION: uncomment to execute
    # rmse_pred = calculate_rmse(drones, num_iterations, after_update=False)
    # rmse_upt = calculate_rmse(drones, num_iterations, after_update=True)
    # plots.plot_error(num_iterations, rmse_pred, rmse_upt)

    # plots.plot_variances_over_time(drones) 
    # plots.plot_variances_over_time(drones, loc='y') 

    # plots.plot_kalman_gain(drones)


