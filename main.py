import numpy as np
import random
from src.drone import Drone
from src.pid_controller import PIDController
from utils.metrics import calculate_rmse, calculate_detection_metrics
from utils.plotting import animate_simulation
import matplotlib; matplotlib.use("TkAgg")
from utils.pid_tuning import grid_search_pid

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
    Kp_values = np.arange(0.5, 1.5, 0.5)
    Ki_values = np.arange(0.05, 0.15, 0.05)
    Kd_values = np.arange(0.02, 0.07, 0.02)
    best_params = (1.0, 0.1, 0.02)
    pid_controller = PIDController(*best_params)

    # Inizializzazione dei droni con posizioni casuali
    drones = []
    for i in range(num_drones):
        x0 = np.array([np.random.uniform(0, 30), np.random.uniform(0, 30), 0, 0, 0, 0])
        P0 = 0.1 * np.eye(6)
        drone = Drone(i + 1, x0, P0, F, G, Q, H, R, pid_controller)
        drones.append(drone)

    for drone in drones:
        drone.neighbors = [d for d in drones if d.id != drone.id]

    fire_position = np.array([random.uniform(0, 30), random.uniform(0, 30)])  # Posizione casuale del fuoco
    interim_master = None
    fire_detected = False

    dt = 1.0  # Tempo tra i passi della simulazione
    for k in range(30):  # Estensione del numero di iterazioni per avvicinarsi al fuoco
        for drone in drones:
            if fire_detected and hasattr(drone, 'target_position'):
                error = drone.target_position - drone.x[:2]
                u = np.hstack((drone.pid_controller.compute(error, dt), [0, 0, 0, 0]))
            else:
                if interim_master and drone.id == interim_master.id:
                    error = fire_position - drone.x[:2]
                    u = np.hstack((drone.pid_controller.compute(error, dt), [0, 0, 0, 0]))
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

    animate_simulation(drones, fire_position, interim_master)

if __name__ == "__main__":
    simulate_fire_detection()
