import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.linalg import sqrtm

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = np.zeros(2)
        self.prev_error = np.zeros(2)

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

class Drone:
    def __init__(self, id, x0, P0, F, G, Q, H, R, pid_controller):
        self.id = id
        self.x = x0  # Initial state
        self.P = P0  # Initial covariance
        self.F = F  # State transition matrix
        self.G = G  # Control matrix
        self.Q = Q  # Process noise covariance
        self.H = H  # Observation matrix
        self.R = R  # Measurement noise covariance
        self.U = np.eye(F.shape[0])  # Initial state transition matrix
        self.neighbors = []  # List of neighbors
        self.active = True  # Active state of the drone
        self.positions = [x0[:2]]  # Estimated positions for plotting
        self.true_positions = [x0[:2]]  # True positions for error calculation
        self.pid_controller = pid_controller  # PID controller

    def predict(self, u):
        if not self.active:
            return
        # State propagation
        self.x = self.F @ self.x + u
        # Covariance propagation
        self.P = self.F @ self.P @ self.F.T + self.G @ self.Q @ self.G.T
        # Update the state transition matrix
        self.U = self.F @ self.U
        # Store positions for plotting
        self.positions.append(self.x[:2])
        self.true_positions.append(self.x[:2] + np.random.normal(0, 0.1, size=2))  # Simulate true positions with noise

    def update(self, z, H_rel, R_rel, other_drone):
        if not self.active or not other_drone.active:
            return

        # Combined state of both drones
        combined_state = np.concatenate((self.x, other_drone.x))
        # Relative measurement error
        ra = z - H_rel @ combined_state  # Innovation

        # Combined covariance (block diagonal with covariances of both drones)
        P_combined = np.block([[self.P, np.zeros_like(self.P)], [np.zeros_like(self.P), other_drone.P]])

        # Innovation covariance
        Sab = R_rel + H_rel @ P_combined @ H_rel.T

        while np.any(np.linalg.eigvals(Sab) <= 0):
            Sab += np.eye(Sab.shape[0]) * 1e-6

        S_ab_inv_sqrt = np.linalg.inv(np.real(sqrtm(Sab)))

        # Kalman Gain
        Gamma_a = self.P @ H_rel[:, :self.P.shape[0]].T @ S_ab_inv_sqrt.T
        Gamma_b = other_drone.P @ H_rel[:, self.P.shape[0]:].T @ S_ab_inv_sqrt.T

        # State update
        self.x = self.x + Gamma_a @ ra
        other_drone.x = other_drone.x + Gamma_b @ ra

        # Covariance update
        self.P = self.P - Gamma_a @ Sab @ Gamma_a.T
        other_drone.P = other_drone.P - Gamma_b @ Sab @ Gamma_b.T

        # Prepare the update message
        update_message = {
            "a": self.id,
            "b": other_drone.id,
            "ra": ra,
            "Gamma_a": Gamma_a,
            "Gamma_b": Gamma_b,
            "Phi_a": self.P,
            "Phi_b": other_drone.P,
            "Sab": Sab
        }

        # Broadcast the update message to other drones
        self.broadcast_update(update_message)

    def broadcast_update(self, update_message):
        for neighbor in self.neighbors:
            if neighbor.id != update_message["a"] and neighbor.id != update_message["b"]:
                neighbor.process_update(update_message)

    def process_update(self, update_message):
        # Extract data from the update message
        ra = update_message["ra"]
        Gamma_a = update_message["Gamma_a"]
        Gamma_b = update_message["Gamma_b"]
        Sab = update_message["Sab"]
        S_ab_inv_sqrt = np.linalg.inv(np.real(sqrtm(Sab)))

        # Calculate the update matrix for other drones
        Gamma_j_a = self.P @ (Gamma_a @ S_ab_inv_sqrt)
        Gamma_j_b = self.P @ (Gamma_b @ S_ab_inv_sqrt)
        Gamma_j = Gamma_j_a - Gamma_j_b

        # State and covariance update
        self.x = self.x + Gamma_j @ ra
        self.P = self.P - Gamma_j @ Sab @ Gamma_j.T

    def check_sensor(self, z):
        if not self.active:
            return False
        if np.any(np.isnan(z)) or np.any(np.isinf(z)):
            print(f"Sensor failure detected on drone {self.id}")
            return False
        return True

    def detect_fire(self, fire_position):
        if not self.active:
            return False
        fire_detected = np.linalg.norm(fire_position - self.x[:2]) < 5
        return fire_detected

def calculate_rmse(drones):
    total_error = 0
    count = 0
    for drone in drones:
        for est_pos, true_pos in zip(drone.positions, drone.true_positions):
            total_error += np.linalg.norm(est_pos - true_pos)**2
            count += 1
    rmse = np.sqrt(total_error / count)
    return rmse

def calculate_detection_metrics(drones, fire_position):
    y_true = []
    y_pred = []
    for drone in drones:
        for pos in drone.positions:
            y_true.append(np.linalg.norm(pos - fire_position) < 5)
            y_pred.append(drone.detect_fire(fire_position))
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1

def simulate_fire_detection():
    # Parametri del modello
    F = np.array([[1, 0, 0.1, 0], [0, 1, 0, 0.1], [0, 0, 1, 0], [0, 0, 0, 1]])
    G = np.eye(4)
    Q = 0.1 * np.eye(4)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    R = 0.1 * np.eye(2)

    # Parametri del PID Controller
    Kp = 1.0
    Ki = 0.1
    Kd = 0.05
    pid_controller = PIDController(Kp, Ki, Kd)

    # Inizializzazione dei droni con posizioni casuali
    x0_1 = np.array([np.random.uniform(0, 30), np.random.uniform(0, 30), 0, 0])
    P0_1 = 0.1 * np.eye(4)
    x0_2 = np.array([np.random.uniform(0, 30), np.random.uniform(0, 30), 0, 0])
    P0_2 = 0.1 * np.eye(4)
    x0_3 = np.array([np.random.uniform(0, 30), np.random.uniform(0, 30), 0, 0])
    P0_3 = 0.1 * np.eye(4)
    x0_4 = np.array([np.random.uniform(0, 30), np.random.uniform(0, 30), 0, 0])
    P0_4 = 0.1 * np.eye(4)

    drone1 = Drone(1, x0_1, P0_1, F, G, Q, H, R, pid_controller)
    drone2 = Drone(2, x0_2, P0_2, F, G, Q, H, R, pid_controller)
    drone3 = Drone(3, x0_3, P0_3, F, G, Q, H, R, pid_controller)
    drone4 = Drone(4, x0_4, P0_4, F, G, Q, H, R, pid_controller)

    drones = [drone1, drone2, drone3, drone4]
    for drone in drones:
        drone.neighbors = [d for d in drones if d.id != drone.id]

    fire_position = np.array([15, 15])  # Posizione fissa del fuoco

    dt = 1.0  # Tempo tra i passi della simulazione
    for k in range(10):  # Estensione del numero di iterazioni per avvicinarsi al fuoco
        for drone in drones:
            # Calcolo dell'errore e del controllo PID per avvicinarsi al fuoco
            error = fire_position - drone.x[:2]
            u = np.hstack((drone.pid_controller.compute(error, dt), [0, 0]))  # Movimento verso il fuoco

            # Componente di repulsione per evitare collisioni
            repulsion = np.zeros(2)
            for neighbor in drone.neighbors:
                if np.linalg.norm(drone.x[:2] - neighbor.x[:2]) < 5:  # Soglia di repulsione
                    repulsion += (drone.x[:2] - neighbor.x[:2]) / np.linalg.norm(drone.x[:2] - neighbor.x[:2])
            u[:2] += repulsion * 0.1  # Scala la repulsione

            drone.predict(u)

        if k % 2 == 0:
            H_rel = np.block([[-H, H]])
            z = np.array([1, 1])  # Misurazione di esempio
            R_rel = 0.1 * np.eye(2)
            for i in range(len(drones)):
                for j in range(i + 1, len(drones)):
                    if drones[i].check_sensor(z) and drones[j].check_sensor(z):
                        drones[i].update(z, H_rel, R_rel, drones[j])
                        print(f"Update at step {k} between Drone {drones[i].id} and Drone {drones[j].id}")
                        for drone in drones:
                            print(f"Drone {drone.id} state: {drone.x}, covariance: {drone.P}")

        for drone in drones:
            if drone.detect_fire(fire_position):
                print(f"Fire detected by Drone {drone.id} at position {drone.x[:2]}")

        if k == 25:
            print("Simulating communication loss for Drone 2")
            drone2.active = False

        if k == 35:
            print("Simulating recovery for Drone 2")
            drone2.active = True

    animate_simulation(drones, fire_position)

    rmse = calculate_rmse(drones)
    precision, recall, f1 = calculate_detection_metrics(drones, fire_position)

    print(f"RMSE: {rmse}")
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

def plot_simulation(drones, fire_position):
    plt.figure()
    for drone in drones:
        positions = np.array(drone.positions)
        plt.plot(positions[:, 0], positions[:, 1], label=f'Drone {drone.id}')
        plt.scatter(positions[-1, 0], positions[-1, 1], s=100)  # Mark the final position

    plt.scatter(fire_position[0], fire_position[1], color='red', label='Fire Detected')

    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.title('Drone Fire Detection Simulation')
    plt.grid()
    plt.show()

def animate_simulation(drones, fire_position):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 30)  # Adjust according to the simulation boundaries
    ax.set_ylim(0, 30)  # Adjust according to the simulation boundaries
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Drone Fire Detection Simulation')
    ax.grid()

    lines = [ax.plot([], [], label=f'Drone {drone.id}')[0] for drone in drones]
    final_positions = [ax.scatter([], [], s=100) for _ in drones]
    fire_scatter = ax.scatter(fire_position[0], fire_position[1], color='red', label='Fire Detected')

    ax.legend()

    def init():
        for line in lines:
            line.set_data([], [])
        for final_pos in final_positions:
            final_pos.set_offsets(np.empty((0, 2)))
        fire_scatter.set_offsets(np.array([fire_position]))
        return lines + final_positions + [fire_scatter]

    def update(frame):
        for line, final_pos, drone in zip(lines, final_positions, drones):
            positions = np.array(drone.positions[:frame])
            if positions.size > 0:
                line.set_data(positions[:, 0], positions[:, 1])
                final_pos.set_offsets(positions[-1].reshape(1, 2))

        return lines + final_positions + [fire_scatter]

    ani = animation.FuncAnimation(
        fig, update, frames=range(1, max(len(drone.positions) for drone in drones) + 1),
        init_func=init, blit=True, repeat=True, interval=500
    )

    plt.show()

if __name__ == "__main__":
    simulate_fire_detection()
