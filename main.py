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
from utils.metrics import calculate_rmse, calculate_detection_metrics
from utils.plotting import plot_simulation, animate_simulation, plot_rmse
from utils.pid_tuning import simulate_drone_with_pid, grid_search_pid
import utils.formation as formation

random.seed(123)

def simulate_fire_detection(
    num_agents=4,
    sensing_range=10,
    num_iterations=30,
    formation_type='circle',
    formation_radius=5
):

    measurements_history = []

    # Inizializzazione droni (state x and covariance P)
    initial_positions = [np.array([np.random.uniform(0, 30), np.random.uniform(0, 30)]) for _ in range(num_agents)]
    state_dim = initial_positions[0].shape[0]
    initial_covariances = [0.1 * np.eye(state_dim) for _ in range(num_agents)]
    
    # Parametri del modello
    #F = np.array([[1, 0, 0.1, 0], [0, 1, 0, 0.1], [0, 0, 1, 0], [0, 0, 0, 1]])
    F = formation.initialize_state_transition_matrix(num_agents, initial_positions)
    # for now we init G with the same func as F: the process noise affects the state components directly
    #G = np.eye(4)
    G = formation.initialize_state_transition_matrix(num_agents, initial_positions)
    # assuming same noise variance for all drones
    #Q = 0.1 * np.eye(4)
    Q = formation.initialize_process_noise_covariance_matrix(num_agents, initial_positions)
    # Initialize the relative observation matrix
    H_rel = formation.initialize_relative_observation_matrix(num_agents, initial_positions)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]) # simple obs matrix for single pair of robots
    #R = 0.1 * np.eye(2)
    R = formation.initialize_measurement_noise_covariance_matrix(num_agents, measurement_noise_variance=0.1)

    # Parametri del PID Controller
    target = np.array([15, 15]) # fire
    Kp_values = np.arange(0.5, 1.5, 0.5)
    Ki_values = np.arange(0.05, 0.15, 0.05)
    Kd_values = np.arange(0.02, 0.07, 0.02)
    best_params, best_mse = grid_search_pid(num_agents, initial_positions, target, Kp_values, Ki_values, Kd_values, num_iterations)
    print(f"Best PID parameters: Kp={best_params[0]}, Ki={best_params[1]}, Kd={best_params[2]}, MSE={best_mse}")
    pid_controller = PIDController(*best_params)

    drones = [Drone(
        i+1, 
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

    # Posizione fissa del fuoco (usato anche come centro della formazione)
    fire_position = np.array([15, 15])  

    # Offset formazione
    formation_offsets = formation.calculate_formation_offsets(formation_type, num_agents, formation_radius)

    dt = 1.0  # Tempo tra i passi della simulazione
    for k in range(num_iterations):  # Estensione del numero di iterazioni per avvicinarsi al fuoco
        for drone in drones:
            u = formation.compute_control_input(drone, fire_position, formation_offsets, dt)
            drone.predict(u)

        #if k % 2 == 0:
        #H_rel = np.block([[-H, H]])
        z = np.array([1, 1])  # Misurazione di esempio
        R_rel = 0.1 * np.eye(2)
        measurement_taken = False

        for i in range(len(drones)):
            for j in range(i + 1, len(drones)):
                distance = np.linalg.norm(drones[i].x[:2] - drones[j].x[:2])
                if distance < sensing_range:
                    # Example measurement as the midpoint
                    z = (drones[i].x[:2] + drones[j].x[:2]) / 2 
                    # Become interim master 
                    interim_master = drones[i]
                    interim_master.update(z, H, R_rel, drones[j]) #H_rel
                    print(f"Update at step {k} between Drone {drones[i].id} and Drone {drones[j].id}")
                    # Update for all other drones
                    for drone in drones:
                        if drone.id != interim_master.id:
                            interim_master.update(z, H, R_rel, drone)
                    measurement_taken = True
                    measurements_history.append(f"Measurement at step {k} between Drone {drones[i].id} and Drone {drones[j].id} ")
                    break
            if measurement_taken:
                # no more than one interim master per iteration
                break

        if not measurement_taken:
            for drone in drones:
                pass # update = prediction

        for drone in drones:
            if drone.detect_fire(fire_position):
                print(f"Fire detected by Drone {drone.id} at position {drone.x[:2]}")

        if k == 25:
            print("Simulating communication loss for Drone 2")
            drones[1].active = False

        if k == 35:
            print("Simulating recovery for Drone 2")
            drones[1].active = True

    animate_simulation(drones, fire_position)

    rmse_pred = calculate_rmse(drones, num_iterations, after_update=False)
    rmse_upt = calculate_rmse(drones, num_iterations, after_update=True)
    plot_rmse(num_iterations, rmse_pred, rmse_upt)
    #precision, recall, f1 = calculate_detection_metrics(drones, fire_position)

    #print(f"RMSE: {rmse}")
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    print(f"PID best parameters selected: Kp={best_params[0]}, Ki={best_params[1]}, Kd={best_params[2]}, MSE={best_mse}")
    print("---------------------------------------")
    for m in measurements_history:
        print(m)



if __name__ == "__main__":
    simulate_fire_detection(num_agents=5)
