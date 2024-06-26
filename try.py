import numpy as np
from IPython import embed
from scipy.linalg import sqrtm

class Drone:
    def __init__(self, drone_id, state_dim, control_dim, process_noise_dim, Qi, R, neighbors=None):
        self.id = drone_id
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.process_noise_dim = process_noise_dim
        self.Qi = Qi  # Process noise covariance matrix
        self.R = R  # Measurement noise covariance matrix
        self.neighbors = neighbors if neighbors else []  # List of neighboring drone IDs


        # Initialize state estimate and covariance
        self.state = np.zeros((state_dim, 1))  # Initial state estimate ˆ xi+(0)
        self.covariance = np.eye(state_dim)  # Initial covariance Pi+(0)
        self.Phi = np.eye(state_dim)  # Initial Φi(0)
        self.P_bar = {}  # Dictionary for cross-covariances ̄ Pi jl(0)
    
    def state_transition(self, control_input):
        # (f^i) state transition function describes how the state of the system evolves over time given the current state and control inputs
        # Example state transition function for a 2D robot
        x, y, theta = self.state.flatten()
        v, omega = control_input.flatten()
        dt = 1 # Time step, should be defined based on your system
        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = theta + omega * dt
        return np.array([[x_new], [y_new], [theta_new]])

    def calculate_Fi(self, control_input):
        # Example calculation of the Jacobian matrix Fi
        x, y, theta = self.state.flatten()
        v, omega = control_input.flatten()
        dt = 1 # Time step
        Fi = np.array([
        [1, 0, -v * np.sin(theta) * dt],
        [0, 1, v * np.cos(theta) * dt],
        [0, 0, 1]
        ])
        return Fi

    def calculate_Gi(self, state):
        # Example calculation of the process noise Jacobian matrix Gi(xi)
        # Assuming process noise affects the state directly
        Gi = np.eye(self.state_dim, self.process_noise_dim)
        return Gi

    ## KALMAN

    def propagate(self, control_input):
        # Calculate Fi and Gi based on the current state and control input
        Fi = self.calculate_Fi(control_input)
        Gi = self.calculate_Gi(self.state)

        # State propagation
        self.state = self.state_transition(control_input)
        # Evolution of Φi
        self.Phi = Fi @ self.Phi
        # Covariance propagation
        self.covariance = Fi @ self.covariance @ Fi.T + Gi @ self.Qi @ Gi.T
    
    def update_no_measurement(self):
        self.state = self.state
        self.covariance = self.covariance
        self.cross_covariances = self.cross_covariances

    def update_with_measurement(self, b, measurement, landmark_message):
        # Extract landmark message
        x_b_minus, Phi_b, P_b_minus = landmark_message

        # Compute residual and innovation covariance
        r_a = measurement - self.h_ab(self.state, x_b_minus)
        H_a, H_b = self.compute_jacobians(self.state, x_b_minus)
        # Extract relevant matrices from P_bar
        b_tuple = tuple(b.flatten())  # Convert numpy array to a tuple
        P_bar_a_b = self.P_bar.get((self.id, b_tuple), np.zeros((self.state_dim, self.state_dim)))
        P_bar_b_a = self.P_bar.get((b_tuple, self.id), np.zeros((self.state_dim, self.state_dim)))
        # Compute S_ab
        S_ab = (self.R + H_a @ self.covariance @ H_a.T + H_b @ P_b_minus @ H_b.T -
                H_a @ self.Phi @ P_bar_a_b @ Phi_b.T @ H_b.T -
                H_b @ Phi_b @ P_bar_b_a @ self.Phi.T @ H_a.T)
    
        S_ab_inv_sqrt = np.linalg.inv(sqrtm(S_ab)) # used in following calculations

        # Compute correction terms
        # key = (tuple(self.id.flatten()), tuple(b.flatten()))
        # P_bar_a_b = self.P_bar[key]
        embed()
        D_a = (np.linalg.inv(self.Phi) @ self.Phi @ self.P_bar[(self.id, b_tuple)] @ Phi_b.T @ H_b.T -
               np.linalg.inv(self.Phi) @ self.P_bar @ H_a.T) @ S_ab_inv_sqrt
        D_b = (np.linalg.inv(Phi_b) @ P_b_minus @ H_b.T -
               self.P_bar[(b_tuple, self.id)] @ self.Phi.T @ H_a.T) @ S_ab_inv_sqrt    #np.linalg.inv(S_ab)

        # Broadcast update message to the network (simplified here)
        update_message = (self.id, b, r_a, D_a, D_b,
                        Phi_b.T @ H_b.T @ S_ab_inv_sqrt, 
                        self.Phi.T @ H_a.T @ S_ab_inv_sqrt)

        return update_message
        

    def broadcast_update_message(self, update_message, drones):
        """
        Broadcasts the update message to all other drones in the network.

        :param update_message: Tuple containing the update information.
        :param drones: List of all drones in the network.
        """
        for drone in drones:
            if drone.id != self.id:  # Exclude self from broadcasting
                drone.receive_update_message(update_message)

    def receive_update_message(self, update_message):
        """
        Receives the update message from another drone and updates its own state accordingly.

        :param update_message: Tuple containing the update information.
        """
        a, b, r_a, D_a, D_b, update_term_b, update_term_a = update_message

        # Update own variables using the received information
        # Calculate Dj for each j in V\{a, b}
        for j in self.neighbors:
            if j != a and j != b:
                # Extract necessary variables for j
                P_j_jb = self.P_bar.get((j, b), np.zeros_like(self.P_bar))
                P_j_ja = self.P_bar.get((j, a), np.zeros_like(self.P_bar))
                Phi_j = self.Phi
                H_j = self.compute_jacobian(self.state, j)

                # Calculate Dj
                D_j = (P_j_jb @ Phi_b.T @ H_b.T @ S_ab_inv_sqrt - 
                    P_j_ja @ Phi_a.T @ H_a.T @ S_ab_inv_sqrt)

                # Update variables
                self.state += Phi_j @ D_j @ r_a
                self.P_bar += Phi_j @ D_j @ D_j.T @ Phi_j.T
                self.P_bar[(j, l)] -= D_j @ D_l.T
    

    def h_ab(self, state_a, state_b):
        """
        Compute the measurement model h_ab for the relative position of robot b with respect to robot a.

        state_a: State of robot a [x_a, y_a, theta_a]
        state_b: State of robot b [x_b, y_b, theta_b]
        Relative measurement [distance, bearing]
        """
        x_a, y_a, theta_a = state_a.flatten()
        x_b, y_b, theta_b = state_b.flatten()

        # Compute the relative position
        delta_x = x_b - x_a
        delta_y = y_b - y_a

        # Compute the distance and bearing
        distance = np.sqrt(delta_x**2 + delta_y**2)
        bearing = np.arctan2(delta_y, delta_x) - theta_a

        # Normalize bearing to the range [-pi, pi]
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi

        return np.array([[distance], [bearing]])

    def compute_jacobians(self, state_a, state_b):
        """
        Compute the Jacobians H_a and H_b of the measurement model (Equation 6).

        :param state_a: State of robot a [x_a, y_a, theta_a]
        :param state_b: State of robot b [x_b, y_b, theta_b]
        :return: Jacobians H_a and H_b
        """
        x_a, y_a, theta_a = state_a.flatten()
        x_b, y_b, theta_b = state_b.flatten()

        delta_x = x_b - x_a
        delta_y = y_b - y_a
        distance = np.sqrt(delta_x**2 + delta_y**2)

        H_a = np.array([
            [-delta_x / distance, -delta_y / distance, 0],
            [delta_y / (distance**2), -delta_x / (distance**2), -1]
        ])

        H_b = np.array([
            [delta_x / distance, delta_y / distance, 0],
            [-delta_y / (distance**2), delta_x / (distance**2), 0]
        ])

        return H_a, H_b
    


# Example simulation function
def main():
    num_drones = 5  # Number of drones in the network
    drones = []

    # Initialize drones with unique IDs and neighbors
    for i in range(num_drones):
        # Example setup for each drone
        state_dim = 3  # State dimension [x, y, theta] theta is orientation
        control_dim = 2  # Control input dimension [v, omega] (velocities)
        process_noise_dim = 3  # Dimension of process noise vector
        Qi = np.diag([0.1, 0.1, 0.01])  # Process noise covariance matrix
        R = np.eye(2) * 0.01  # Example measurement noise covariance matrix

        # Assuming some initial neighbors for each drone (for demonstration)
        neighbors = [j for j in range(num_drones) if j != i]

        drone = Drone(drone_id=i, state_dim=state_dim, control_dim=control_dim,
                      process_noise_dim=process_noise_dim, Qi=Qi, R=R, neighbors=neighbors)
        drones.append(drone)

    num_time_steps = 10  # Number of simulation time steps

    # Example simulation loop
    for t in range(num_time_steps):
        # Each drone propagates its state based on control inputs (not shown in this example)
        for drone in drones:
            # Example control input (v, omega)
            control_input = np.array([[0.1], [0.05]])  # Adjust as per your simulation scenario
            drone.propagate(control_input)

        # Assume one drone makes a measurement and broadcasts the update message
        # Example: Drone 0 makes a measurement towards 3 and broadcasts the update
        if t == 0:
            measurement = np.array([[5.0], [0.1]])  # Example measurement [distance, bearing]
            landmark_message = (np.array([[10.0], [0.05], [0.1]]), np.eye(3), np.eye(3))  # Example landmark message
            b = 3 
            update_message = drones[0].update_with_measurement(measurement, b, landmark_message)
            drones[0].broadcast_update_message(update_message, drones)

        # Each drone receives and processes update messages
        for drone in drones:
            # Example: All drones receive and process update messages
            for other_drone in drones:
                if other_drone.id != drone.id:
                    drone.receive_update_message(update_message)

# Run the main simulation
if __name__ == "__main__":
    main()