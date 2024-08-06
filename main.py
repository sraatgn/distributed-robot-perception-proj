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

#random.seed(123)

def simulate_fire_detection(
    num_agents=4,
    sensing_range=15,
    num_iterations=50,
    dt = 1.0,                   # Tempo tra i passi della simulazione
    formation_type='circle',
    formation_radius=5
):

    assert num_agents >= 2, f"Sir, this is a distributed system. Why would you use {num_agents} drone(s)?"
    
    fire_position = np.array([15, 15])
    measurements_history = []

    # Inizializzazione droni (state x and covariance P)
    initial_positions = [np.array([np.random.uniform(0, 30), np.random.uniform(0, 30)]) for _ in range(num_agents)]
    state_dim = initial_positions[0].shape[0]
    initial_covariances = [0.1 * np.eye(state_dim) for _ in range(num_agents)]
    
    # Parametri del modello
    F = formation.initialize_state_transition_matrix(num_agents, initial_positions)
    # for now we init G with the same func as F: the process noise affects the state components directly
    G = formation.initialize_state_transition_matrix(num_agents, initial_positions)
    Q = formation.initialize_process_noise_covariance_matrix(num_agents, initial_positions, default_variance=[0.1, 0.5])
    H_rel = formation.initialize_relative_observation_matrix(num_agents, initial_positions)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]) # simple obs matrix for single pair of robots
    R = formation.initialize_measurement_noise_covariance_matrix(num_agents, measurement_noise_variance=0.1)

    # Parametri del PID Controller
    Kp_values = np.arange(0.5, 1.5, 0.25)
    Ki_values = np.arange(0.05, 0.15, 0.05)
    Kd_values = np.arange(0.02, 0.07, 0.02)
    best_params, best_mse, results = grid_search_pid(num_agents, initial_positions, fire_position, Kp_values, Ki_values, Kd_values, num_iterations)
    print(f"Best PID parameters: Kp={best_params[0]}, Ki={best_params[1]}, Kd={best_params[2]}, MSE={best_mse}")
    pid_controller = PIDController(*best_params)

    # Inizializzazione droni
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

    # Offset formazione
    formation_offsets = formation.calculate_formation_offsets(formation_type, num_agents, formation_radius)

    for k in range(num_iterations):  
        for drone in drones:
            u = formation.compute_control_input(drone, fire_position, formation_offsets, dt)
            # Simulate true positions with noise (for RMSE)
            #drone.true_positions.append(drone.x[:2] + np.random.normal(0, 0.1, size=2))  
            true_pos = np.dot(drone.F, drone.true_positions[-1]) + np.dot(drone.G, u)
            drone.true_positions.append(true_pos)

            ## PREDICTION
            pred_state = drone.predict(u)
        
        measurement_taken = False

        for i in range(len(drones)):
            if measurement_taken:
                break # no more than one interim master per iteration

            for j in range(i + 1, len(drones)):
                distance = np.linalg.norm(drones[i].x[:2] - drones[j].x[:2])
                if distance < sensing_range:
                    # Example measurement as the midpoint
                    #z = (drones[i].x[:2] + drones[j].x[:2]) / 2 
                    # Relative measurement: eucledian distance 
                    z = simulate_measurement(drones[i].x, drones[j].x, drones[i].R)
                    # Become interim master 
                    interim_master = drones[i]
                    interim_master.update(z, H, interim_master.R, drones[j]) #H_rel
                    print(f"Relative measurement at step {k} between Drone {drones[i].id} and Drone {drones[j].id}")

                    ## NOT NEEDED BC THE UPDATE FOR OTHER DRONES IS CALLED ABOVE IN interim_master.update()
                    # # Update for all other drones
                    # for drone in drones:
                    #     if drone.id != interim_master.id:
                    #         drone.update(z, H, drone.R, interim_master)
                    measurement_taken = True
                    measurements_history.append(f"Measurement at step {k} between Drone {drones[i].id} and Drone {drones[j].id} ")
                    break

        if measurement_taken == False:
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

    plots.animate_simulation(drones, fire_position)

    # print(f"PID best parameters selected: Kp={best_params[0]}, Ki={best_params[1]}, Kd={best_params[2]}, MSE={best_mse}")
    # print("---------------------------------------")
    # for m in measurements_history:
    #     print(m)

    return drones



if __name__ == "__main__":

    drones = simulate_fire_detection()

    ## PLOTS & EVALUATION
    # rmse_pred = calculate_rmse(drones, num_iterations, after_update=False)
    # rmse_upt = calculate_rmse(drones, num_iterations, after_update=True)
    # plot_rmse(num_iterations, rmse_pred, rmse_upt)

    # plot_kalman_gain(drones)

    plots.plot_x_variances_over_time(drones) 

    ##########################
    ## CODE FOR RUNNING MULTIPLE SIMULATIONS AND PLOTTING RMSE FOR ONE DRONE
    # # Number of simulations to run
    # NUM_SIM = 1
    # NUM_ITER = 50

    # # Arrays to store RMSEs for each simulation
    # rmse_pred_all = np.zeros((NUM_SIM, NUM_ITER))
    # rmse_update_all = np.zeros((NUM_SIM, NUM_ITER))

    # for sim in range(NUM_SIM):
    #     # Simulate and return list of drones
    #     drones = simulate_fire_detection(num_iterations=NUM_ITER)
    #     plot_trajectories(drones, fire_position=np.array([15, 15]))

    #     # Build lists for drone 1 for evaluation
    #     for k in range(NUM_ITER):
    #         true_positions = [drone.true_positions[k] for drone in drones if drone.id == 1]
    #         pred_positions = [drone.positions_pred[k] for drone in drones if drone.id == 1]
    #         updated_positions = [drone.positions_upt[k] for drone in drones if drone.id == 1]
            
    #         if true_positions and pred_positions:
    #             rmse_pred_all[sim, k] = compute_rmse(true_positions, pred_positions)
            
    #         if true_positions and updated_positions:
    #             rmse_update_all[sim, k] = compute_rmse(true_positions, updated_positions)

    #     # Compute mean RMSE over all simulations
    # rmse_pred_mean = np.mean(rmse_pred_all, axis=0)
    # rmse_update_mean = np.mean(rmse_update_all, axis=0)

    # # Plot the results
    # plt.figure(figsize=(10, 5))
    # plt.plot(rmse_pred_mean, label='RMSE after Prediction')
    # plt.plot(rmse_update_mean, label='RMSE after Update')
    # plt.xlabel('Time Step')
    # plt.ylabel('RMSE')
    # plt.title('RMSE of Drone Position Estimation')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

