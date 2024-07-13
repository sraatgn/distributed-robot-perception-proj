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
    results = []

    for Kp, Ki, Kd in product(Kp_values, Ki_values, Kd_values):
        pid = PIDController(Kp, Ki, Kd)
        mse = simulate_drone_with_pid(pid, target, num_steps, dt)
        results.append((Kp, Ki, Kd, mse))
        if mse < best_mse:
            best_mse = mse
            best_params = (Kp, Ki, Kd)
        print(f"Tested Kp={Kp}, Ki={Ki}, Kd={Kd}, MSE={mse}")

    return best_params, best_mse, results

def visualize_pid_tuning(results):
    import matplotlib.pyplot as plt

    Kp_values = sorted(set(result[0] for result in results))
    Ki_values = sorted(set(result[1] for result in results))
    Kd_values = sorted(set(result[2] for result in results))

    Kp_mse = {Kp: np.full(len(Ki_values), np.nan) for Kp in Kp_values}
    Ki_mse = {Ki: np.full(len(Kd_values), np.nan) for Ki in Ki_values}
    Kd_mse = {Kd: np.full(len(Kp_values), np.nan) for Kd in Kd_values}

    for Kp, Ki, Kd, mse in results:
        Ki_index = Ki_values.index(Ki)
        Kd_index = Kd_values.index(Kd)
        Kp_index = Kp_values.index(Kp)

        Kp_mse[Kp][Ki_index] = mse
        Ki_mse[Ki][Kd_index] = mse
        Kd_mse[Kd][Kp_index] = mse

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    for Kp in Kp_values:
        plt.plot(Ki_values, Kp_mse[Kp], label=f'Kp={Kp}')
    plt.xlabel('Ki')
    plt.ylabel('MSE')
    plt.title('MSE vs Ki for different Kp values')
    plt.legend()

    plt.subplot(3, 1, 2)
    for Ki in Ki_values:
        plt.plot(Kd_values, Ki_mse[Ki], label=f'Ki={Ki}')
    plt.xlabel('Kd')
    plt.ylabel('MSE')
    plt.title('MSE vs Kd for different Ki values')
    plt.legend()

    plt.subplot(3, 1, 3)
    for Kd in Kd_values:
        plt.plot(Kp_values, Kd_mse[Kd], label=f'Kd={Kd}')
    plt.xlabel('Kp')
    plt.ylabel('MSE')
    plt.title('MSE vs Kp for different Kd values')
    plt.legend()

    plt.tight_layout()
    plt.show()
