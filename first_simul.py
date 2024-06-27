import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.linalg import sqrtm

class Drone:
    def __init__(self, id, x0, P0, F, G, Q, H, R):
        self.id = id
        self.x = x0  # Stato iniziale
        self.P = P0  # Covarianza iniziale
        self.F = F  # Matrice di transizione di stato
        self.G = G  # Matrice di controllo
        self.Q = Q  # Covarianza del rumore di processo
        self.H = H  # Matrice di osservazione
        self.R = R  # Covarianza del rumore di osservazione
        self.U = np.eye(F.shape[0])  # Matrice di transizione di stato iniziale
        self.neighbors = []  # Lista dei vicini
        self.active = True  # Stato attivo del drone
        self.positions = [x0[:2]]  # Posizioni stimate per il plotting
        self.true_positions = [x0[:2]]  # Posizioni vere per il calcolo dell'errore

    def predict(self, u):
        if not self.active:
            return
        # Propagazione dello stato
        self.x = self.F @ self.x + u
        # Propagazione della covarianza
        self.P = self.F @ self.P @ self.F.T + self.G @ self.Q @ self.G.T
        # Aggiornamento della matrice di transizione di stato
        self.U = self.F @ self.U
        # Memorizzazione delle posizioni per il plotting
        self.positions.append(self.x[:2])
        self.true_positions.append(self.x[:2] + np.random.normal(0, 0.1, size=2))  # Simulazione delle posizioni vere con rumore

    def update(self, z, H_rel, R_rel, other_drone):
        if not self.active or not other_drone.active:
            return

        # Stato combinato dei due droni
        combined_state = np.concatenate((self.x, other_drone.x))
        # Calcolo dell'errore di misurazione relativo
        ra = z - H_rel @ combined_state

        # Covarianza combinata (blocco diagonale con covarianze dei due droni)
        P_combined = np.block([[self.P, np.zeros_like(self.P)], [np.zeros_like(self.P), other_drone.P]])

        # Definizione delle matrici di osservazione relative
        Ha = H_rel[:, :self.P.shape[0]]  # Estrarre la parte di H_rel corrispondente al drone a
        Hb = H_rel[:, self.P.shape[0]:]  # Estrarre la parte di H_rel corrispondente al drone b

        # Matrici di transizione di stato (per esempio, identità se non altrimenti definite)
        Phi_a = np.eye(self.P.shape[0])
        Phi_b = np.eye(other_drone.P.shape[0])

        # Matrice di cross-covarianza Pi_ab (assumiamo una definizione appropriata o calcolabile)
        Pi_ab = np.zeros((self.P.shape[0], other_drone.P.shape[0]))  # Placeholder

        # Calcolo della covarianza dell'innovazione S_{ab} con la sottrazione inclusa
        Sab = (R_rel + Ha @ self.P @ Ha.T + Hb @ other_drone.P @ Hb.T
               - Ha @ Phi_a @ Pi_ab @ Phi_b.T @ Hb.T)

        # Aggiungi una piccola perturbazione alla diagonale di Sab se non è positiva definita
        while np.any(np.linalg.eigvals(Sab) <= 0):
            Sab += np.eye(Sab.shape[0]) * 1e-6

        # Assicurati che Sab sia reale e positiva definita
        S_ab_inv_sqrt = np.linalg.inv(np.real(sqrtm(Sab)))

        # Formula per Gamma_a
        Gamma_a = (np.linalg.inv(Phi_a) @ Phi_a @ Pi_ab @ Phi_b @ Hb.T - np.linalg.inv(Phi_a) @ self.P @ Ha.T) @ S_ab_inv_sqrt

        # Formula per Gamma_b
        Gamma_b = (np.linalg.inv(Phi_b) @ other_drone.P @ Hb.T - Pi_ab @ Phi_a @ Ha.T) @ S_ab_inv_sqrt

        # Aggiornamento dello stato e della covarianza per entrambi i droni
        self.x = self.x + Gamma_a @ ra
        other_drone.x = other_drone.x + Gamma_b @ ra

        self.P = self.P - Gamma_a @ Ha @ self.P
        other_drone.P = other_drone.P - Gamma_b @ Hb @ other_drone.P

        # Preparazione del messaggio di aggiornamento
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

        # Trasmissione del messaggio di aggiornamento agli altri droni
        self.broadcast_update(update_message)

    def broadcast_update(self, update_message):
        for neighbor in self.neighbors:
            if neighbor.id != update_message["a"] and neighbor.id != update_message["b"]:
                neighbor.process_update(update_message)

    def process_update(self, update_message):
        # Estrazione dei dati dal messaggio di aggiornamento
        ra = update_message["ra"]
        Gamma_a = update_message["Gamma_a"]
        Gamma_b = update_message["Gamma_b"]
        Sab = update_message["Sab"]
        S_ab_inv_sqrt = np.linalg.inv(np.real(sqrtm(Sab)))

        # Calcolo della matrice di aggiornamento per gli altri droni
        Gamma_j_a = self.P @ (Gamma_a @ S_ab_inv_sqrt)
        Gamma_j_b = self.P @ (Gamma_b @ S_ab_inv_sqrt)
        Gamma_j = Gamma_j_a - Gamma_j_b

        # Aggiornamento dello stato e della covarianza
        self.x = self.x + Gamma_j @ ra
        self.P = self.P - Gamma_j @ Sab @ Gamma_j.T

    def check_sensor(self, z):
        if not self.active:
            return False
        if np.any(np.isnan(z)) or np.any(np.isinf(z)):
            print(f"Sensor failure detected on drone {self.id}")
            return False
        return True

    def detect_fire(self, z):
        if not self.active:
            return False
        fire_detected = np.linalg.norm(z[:2] - self.x[:2]) < 5
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

def calculate_detection_metrics(drones, fire_positions):
    y_true = []
    y_pred = []
    for drone in drones:
        for pos in drone.positions:
            y_true.append(any(np.linalg.norm(pos - fire) < 5 for fire in fire_positions))
            y_pred.append(drone.detect_fire(pos))
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

    # Inizializzazione dei droni
    x0_1 = np.array([0, 0, 0, 0])
    P0_1 = 0.1 * np.eye(4)
    x0_2 = np.array([10, 10, 0, 0])
    P0_2 = 0.1 * np.eye(4)
    x0_3 = np.array([20, 0, 0, 0])
    P0_3 = 0.1 * np.eye(4)
    x0_4 = np.array([0, 20, 0, 0])
    P0_4 = 0.1 * np.eye(4)

    drone1 = Drone(1, x0_1, P0_1, F, G, Q, H, R)
    drone2 = Drone(2, x0_2, P0_2, F, G, Q, H, R)
    drone3 = Drone(3, x0_3, P0_3, F, G, Q, H, R)
    drone4 = Drone(4, x0_4, P0_4, F, G, Q, H, R)

    drones = [drone1, drone2, drone3, drone4]
    for drone in drones:
        drone.neighbors = [d for d in drones if d.id != drone.id]

    fire_positions = []

    for k in range(10):
        for drone in drones:
            u = np.zeros(4)
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
            z = drone.H @ drone.x  # Misurazione di esempio
            if drone.detect_fire(z):
                print(f"Fire detected by Drone {drone.id} at position {drone.x[:2]}")
                fire_positions.append(drone.x[:2])

        if k == 5:
            print("Simulating communication loss for Drone 2")
            drone2.active = False

        if k == 7:
            print("Simulating recovery for Drone 2")
            drone2.active = True

    animate_simulation(drones, fire_positions)

    rmse = calculate_rmse(drones)
    precision, recall, f1 = calculate_detection_metrics(drones, fire_positions)

    print(f"RMSE: {rmse}")
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

def plot_simulation(drones, fire_positions):
    plt.figure()
    for drone in drones:
        positions = np.array(drone.positions)
        plt.plot(positions[:, 0], positions[:, 1], label=f'Drone {drone.id}')
        plt.scatter(positions[-1, 0], positions[-1, 1], s=100)  # Mark the final position

    if fire_positions:
        fire_positions = np.array(fire_positions)
        plt.scatter(fire_positions[:, 0], fire_positions[:, 1], color='red', label='Fire Detected')

    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.title('Drone Fire Detection Simulation')
    plt.grid()
    plt.show()

def animate_simulation(drones, fire_positions):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 30)  # Adjust according to the simulation boundaries
    ax.set_ylim(0, 30)  # Adjust according to the simulation boundaries
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Drone Fire Detection Simulation')
    ax.grid()

    lines = [ax.plot([], [], label=f'Drone {drone.id}')[0] for drone in drones]
    final_positions = [ax.scatter([], [], s=100) for _ in drones]
    fire_scatter = ax.scatter([], [], color='red', label='Fire Detected')

    ax.legend()

    def init():
        for line in lines:
            line.set_data([], [])
        for final_pos in final_positions:
            final_pos.set_offsets(np.empty((0, 2)))
        fire_scatter.set_offsets(np.empty((0, 2)))
        return lines + final_positions + [fire_scatter]

    def update(frame):
        for line, final_pos, drone in zip(lines, final_positions, drones):
            positions = np.array(drone.positions[:frame])
            if positions.size > 0:
                line.set_data(positions[:, 0], positions[:, 1])
                final_pos.set_offsets(positions[-1].reshape(1, 2))

        if fire_positions:
            fire_positions_arr = np.array(fire_positions)
            fire_scatter.set_offsets(fire_positions_arr)

        return lines + final_positions + [fire_scatter]

    ani = animation.FuncAnimation(
        fig, update, frames=range(1, max(len(drone.positions) for drone in drones) + 1),
        init_func=init, blit=True, repeat=True, interval=500
    )

    plt.show()

if __name__ == "__main__":
    simulate_fire_detection()
