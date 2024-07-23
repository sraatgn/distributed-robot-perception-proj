import numpy as np
import random
import matplotlib.pyplot as plt
from src.drone import Drone
from src.pid_controller import PIDController
from utils.metrics import calculate_rmse, calculate_detection_metrics
from utils.plotting import animate_simulation
import matplotlib; matplotlib.use("TkAgg")
from utils.pid_tuning import grid_search_pid
from utils.pid_tuning import visualize_pid_tuning
from utils.pid_tuning import box_plot_pid_tuning
from utils.pid_tuning import line_plot_pid_tuning

def form_circle_around_fire(drones, fire_position, radius=5):
    num_drones = len(drones)
    angle_between_drones = 2 * np.pi / num_drones
    for i, drone in enumerate(drones):
        angle = i * angle_between_drones
        target_position = fire_position + radius * np.array([np.cos(angle), np.sin(angle)])
        drone.target_position = target_position

def simulate_fire_detection(num_drones=9):
    measurements_history = []

    # Parametri del modello
    F = np.array([
        [1, 0, 0.1, 0, 0.5 * 0.1**2, 0],
        [0, 1, 0, 0.1, 0, 0.5 * 0.1**2],
        [0, 0, 1, 0, 0.1, 0],
        [0, 0, 0, 1, 0, 0.1],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])
    G = np.array([
        [0.5 * 0.1**2, 0],
        [0, 0.5 * 0.1**2],
        [0.1, 0],
        [0, 0.1],
        [1, 0],
        [0, 1]
    ])
    Q = 0.1 * np.eye(2)  # Matrice di covarianza del rumore di processo
    H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0]
    ])
    R = 0.1 * np.eye(2)

    # Parametri del PID Controller
    Kp_values = np.arange(0.5, 2.0, 0.5)
    Ki_values = np.arange(0.05, 0.25, 0.05)
    Kd_values = np.arange(0.02, 0.1, 0.02)

    target_position = np.array([15, 15])
    best_params, best_mse, results = grid_search_pid(target_position, Kp_values, Ki_values, Kd_values)
    print(f"Best PID parameters: Kp={best_params[0]}, Ki={best_params[1]}, Kd={best_params[2]} with MSE={best_mse}")

    visualize_pid_tuning(results)
    box_plot_pid_tuning(results)
    line_plot_pid_tuning(results)

    best_pid_controller = PIDController(*best_params)

    # Inizializzazione dei droni con posizioni casuali
    drones = []
    for i in range(num_drones):
        x0 = np.array([np.random.uniform(0, 30), np.random.uniform(0, 30), 0, 0, 0, 0])
        P0 = 0.1 * np.eye(6)
        drone = Drone(i + 1, x0, P0, F, G, Q, H, R, best_pid_controller)
        drones.append(drone)

        # Stampa delle matrici iniziali del drone
        print(f"Drone {drone.id} - Initial Matrices:")
        print(f"F:\n{drone.F}")
        print(f"G:\n{drone.G}")
        print(f"Q:\n{drone.Q}")
        print(f"H:\n{drone.H}")
        print(f"R:\n{drone.R}")
        print(f"P:\n{drone.P}")
        print(f"Initial State x0:\n{drone.x}\n")

    for drone in drones:
        drone.neighbors = [d for d in drones if d.id != drone.id]

    fire_position = np.array([random.uniform(0, 30), random.uniform(0, 30)])  # Posizione casuale del fuoco
    interim_master = None
    fire_detected = False
    interim_masters = []

    dt = 1.0  # Tempo tra i passi della simulazione
    for k in range(30):  # Estensione del numero di iterazioni per avvicinarsi al fuoco
        print(f"\nSimulation Step {k}:")
        for drone in drones:
            if fire_detected and hasattr(drone, 'target_position'):
                error = drone.target_position - drone.x[:2]
                control_output = drone.pid_controller.compute(error, dt)
                u = np.hstack((control_output, [0, 0, 0, 0]))
                print(f"Drone {drone.id} - Control Output: {control_output}, Error: {error}")
            else:
                if interim_master and drone.id == interim_master.id:
                    error = fire_position - drone.x[:2]
                    control_output = drone.pid_controller.compute(error, dt)
                    u = np.hstack((control_output, [0, 0, 0, 0]))
                    print(f"Drone {drone.id} (Interim Master) - Control Output: {control_output}, Error: {error}")
                else:
                    random_direction = np.random.uniform(-1, 1, 2)
                    u = np.hstack((random_direction, [0, 0, 0, 0]))

                repulsion = np.zeros(2)
                for neighbor in drone.neighbors:
                    if np.linalg.norm(drone.x[:2] - neighbor.x[:2]) < 5:
                        repulsion += (drone.x[:2] - neighbor.x[:2]) / np.linalg.norm(drone.x[:2] - neighbor.x[:2])
                u[:2] += repulsion * 0.1  # Scala la repulsione

            drone.predict(u)

        H_rel = np.block([[-H, H]])
        R_rel = 0.1 * np.eye(2)
        sensing_range = 10  # Define the sensing range
        measurement_taken = False

        for i in range(len(drones)):
            for j in range(i + 1, len(drones)):
                distance = np.linalg.norm(drones[i].x[:2] - drones[j].x[:2])
                if distance < sensing_range:
                    z = (drones[i].x[:2] + drones[j].x[:2]) / 2
                    interim_master = drones[i]
                    interim_master.update(z, H_rel, R_rel, drones[j])
                    print(f"Update at step {k} between Drone {drones[i].id} and Drone {drones[j].id}")
                    for drone in drones:
                        if drone.id != interim_master.id:
                            interim_master.update(z, H_rel, R_rel, drone)
                    measurement_taken = True
                    measurements_history.append(f"Measurement at step {k} between Drone {drones[i].id} and Drone {drones[j].id}")
                    break
            if measurement_taken:
                break

        if not measurement_taken:
            for drone in drones:
                pass  # update = prediction

        for drone in drones:
            if drone.detect_fire(fire_position):
                print(f"Fire detected by Drone {drone.id} at position {drone.x[:2]}")
                interim_master = drone
                fire_detected = True
                form_circle_around_fire(drones, fire_position)  # Arrange drones in a circle around the fire
                break

        print(f"Step {k}: Interim Master is Drone {interim_master.id if interim_master else 'None'}")
        interim_masters.append(interim_master)

        # Stampa delle matrici dei droni dopo ogni aggiornamento
        for drone in drones:
            print(f"Drone {drone.id} - Matrices after step {k}:")
            print(f"F:\n{drone.F}")
            print(f"G:\n{drone.G}")
            print(f"Q:\n{drone.Q}")
            print(f"H:\n{drone.H}")
            print(f"R:\n{drone.R}")
            print(f"P:\n{drone.P}")
            print(f"State x:\n{drone.x}\n")

    # Calcolo delle metriche finali
    rmse = calculate_rmse(drones)
    precision, recall, f1 = calculate_detection_metrics(drones, fire_position)
    print(f"Final RMSE: {rmse}")
    print(f"Final Detection Metrics - Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

    animate_simulation(drones, fire_position, interim_masters)

if __name__ == "__main__":
    simulate_fire_detection()