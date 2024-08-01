import numpy as np
from scipy.linalg import sqrtm
from IPython import embed

class Drone:
    def __init__(self, id, x0, P0, F, G, Q, H, H_rel, R, pid_controller):
        self.id = id
        self.x = x0  # Initial state
        self.P = P0  # Initial covariance
        self.U = np.eye(F.shape[0])  # Phi (time-varying local variable)
        self.F = F  # State transition matrix
        self.G = G  # Control matrix
        self.Q = Q  # Process noise covariance
        self.H = H  # Observation matrix
        self.H_rel = H_rel # Relative observations matrix
        self.R = R  # Measurement noise covariance
        self.neighbors = []  # List of neighbors
        self.active = True  # Active state of the drone
        self.positions_pred = [x0[:2]]  # Estimated (after predictions) positions for plotting
        self.positions_upt = [x0[:2]]  # Estimated (after update) positions for plotting
        self.true_positions = [x0[:2]]  # True positions for error calculation
        self.kalmangains = [] # For saving kalman gain
        self.pid_controller = pid_controller  # PID controller

    def predict(self, u):
        if not self.active:
            return
        # State propagation
        self.x = self.F @ self.x + u
        # Covariance propagation
        self.P = self.F @ self.P @ self.F.T + self.G @ self.Q @ self.G.T
        # Update Phi
        self.U = self.F @ self.U
        # Store positions for plotting
        self.positions_pred.append(self.x[:2])

        return self.x
        

    def update(self, z, H_rel, R_rel, other_drone):
        if not self.active or not other_drone.active:
            return

        # Combined state of both drones
        combined_state = np.concatenate((self.x, other_drone.x))

        ## INTERIM MASTER COMPUTATIONS
        # Relative measurement error
        ra = z - H_rel @ combined_state  # Innovation
        # Combined covariance (block diagonal with covariances of both drones)
        P_combined = np.block([[self.P, np.zeros_like(self.P)], [np.zeros_like(self.P), other_drone.P]])
        # Innovation covariance
        Sab = R_rel + H_rel @ P_combined @ H_rel.T

        while np.any(np.linalg.eigvals(Sab) <= 0):
            Sab += np.eye(Sab.shape[0]) * 1e-6
        S_ab_inv_sqrt = np.linalg.inv(np.real(sqrtm(Sab)))

        # Intermediate matrices 
        Gamma_a = self.P @ H_rel[:, :self.P.shape[0]].T @ S_ab_inv_sqrt.T
        Gamma_b = other_drone.P @ H_rel[:, self.P.shape[0]:].T @ S_ab_inv_sqrt.T

        ## UPDATES
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

        # save updated state (IM) for metrics
        self.positions_upt.append(self.x[:2])

        ## KALMAN GAIN
        K_a = self.U @ Gamma_a
        # save Kalman gain
        self.kalmangains.append(K_a)
    
        # Broadcast the update message to other drones
        self.broadcast_update(update_message)

    def broadcast_update(self, update_message):
        for neighbor in self.neighbors:
            if neighbor.id != update_message["a"]: #and neighbor.id != update_message["b"]:
                neighbor.process_update(update_message)

    def process_update(self, update_message):
        # Extract data from the update message
        ra = update_message["ra"]
        Gamma_a = update_message["Gamma_a"]
        Gamma_b = update_message["Gamma_b"]
        Sab = update_message["Sab"]
        S_ab_inv_sqrt = np.linalg.inv(np.real(sqrtm(Sab)))

        # Calculate the intermediate matrix for other drones
        Gamma_j_a = self.P @ (Gamma_a @ S_ab_inv_sqrt)
        Gamma_j_b = self.P @ (Gamma_b @ S_ab_inv_sqrt)
        Gamma_j = Gamma_j_a - Gamma_j_b

        # State and covariance update
        self.x = self.x + Gamma_j @ ra
        self.P = self.P - Gamma_j @ Sab @ Gamma_j.T

        # save for metrics
        self.positions_upt.append(self.x[:2])

        ## KALMAN GAIN
        K_i = self.U @ Gamma_j
        # save kalman gain
        self.kalmangains.append(K_i)

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
