import numpy as np
import matplotlib.pyplot as plt

## INITIALIZATION FUNCTIONS 

def initialize_state_transition_matrix(num_agents, state):
    # Each drone has a state vector [x, y]
    state_dim = state[0].shape[0]

    # Initialize the block diagonal matrix F
    F = np.zeros((num_agents * state_dim, num_agents * state_dim))
    
    for i in range(num_agents):
        # The state transition matrix for a single drone
        # the position remains unchanged (identity matrix) if no control inputs are applied.
        Fi = np.array([
            [1, 0],  # x' = x
            [0, 1],  # y' = y
        ])
        
        # Place Fi in the block diagonal position for the i-th drone
        start_idx = i * state_dim
        end_idx = start_idx + state_dim
        F[start_idx:end_idx, start_idx:end_idx] = Fi

    return F

def initialize_process_noise_covariance_matrix(num_agents, state, default_variance=[0.1, 0.1], custom_variances=None):
    ''' 
    initializes Q as a block diagonal matrix with each block corresponding 
    to a drone's process noise covariance matrix.

    custom_variances: is an optional list where each element is a list of variances for each drone.
         Example: custom_variances = [
                        [0.1, 0.1],  # Drone 1
                        [0.2, 0.2],  # Drone 2
                        [0.3, 0.3],  # Drone 3
                        [0.1, 0.1],  # Drone 4
                        [0.2, 0.2]   # Drone 5
                    ]
    If not provided, default_variance is used assuming all drones have same variance
    '''
    # Each drone has a state vector [x, y]
    state_dim = state[0].shape[0]

    # Initialize the block diagonal matrix Q
    Q = np.zeros((num_agents * state_dim, num_agents * state_dim))
    
    for i in range(num_agents):
        # Use custom variance if provided, otherwise use default variance
        if custom_variances and i < len(custom_variances):
            variance_x, variance_y = custom_variances[i]
        else:
            variance_x, variance_y = default_variance
        
        # The process noise covariance matrix for a single drone
        Qi = np.array([
            [variance_x, 0],  # Variance in x
            [0, variance_y]   # Variance in y
        ])
        
        # Place Qi in the block diagonal position for the i-th drone
        start_idx = i * state_dim
        end_idx = start_idx + state_dim
        Q[start_idx:end_idx, start_idx:end_idx] = Qi

    return Q

def initialize_relative_observation_matrix(num_agents, state):
    state_dim = state[0].shape[0]  # Each robot has a state vector [x, y]
    measurement_dim = 2  # Each measurement has a dimension [delta_x, delta_y]
    
    # Total number of measurements (considering all pairs)
    num_measurements = num_agents * (num_agents - 1)  # Each pair of agents forms a measurement
    
    # Initialize the block matrix H_rel
    H_rel = np.zeros((num_measurements * measurement_dim, num_agents * state_dim * 2))
    
    measurement_index = 0
    
    for i in range(num_agents):
        for j in range(num_agents):
            if i != j:
                # Partial derivatives for the measurement between robot i and robot j
                Hij = np.array([
                    [-1, 0, 1, 0],  # d(delta_x) / d(x_i, y_i, x_j, y_j)
                    [0, -1, 0, 1]   # d(delta_y) / d(x_i, y_i, x_j, y_j)
                ])
                
                # Place -Hij and Hij in the correct block positions in H_rel
                start_row = measurement_index * measurement_dim
                start_col_i = i * state_dim
                start_col_j = j * state_dim + num_agents * state_dim  # Offset by num_agents * state_dim for second block
                
                # Fill in the block for -Hij (for robot i)
                H_rel[start_row:start_row + measurement_dim, start_col_i:start_col_i + state_dim] = -Hij[:, :state_dim]
                # Fill in the block for Hij (for robot j)
                H_rel[start_row:start_row + measurement_dim, start_col_j:start_col_j + state_dim] = Hij[:, state_dim:]
                
                measurement_index += 1

    return H_rel

def initialize_measurement_noise_covariance_matrix(num_agents, measurement_noise_variance):
    measurement_dim = 2  # Each measurement is 2-dimensional (delta_x, delta_y)
    R_i = measurement_noise_variance * np.eye(measurement_dim)
    R = np.kron(np.eye(num_agents * (num_agents - 1)), R_i)
    return R

def extract_submatrix(matrix, index, state_dim):
    ''' Function to extract submatrices for i-th drone from the block matrices created above '''
    start_idx = index * state_dim
    end_idx = start_idx + state_dim
    return matrix[start_idx:end_idx, start_idx:end_idx]



## GET IN FORMATION FUNCTIONS

def calculate_formation_offsets(formation, num_drones, radius):
    ''' Computes positions that the drones should aim to achieve in the formation '''
    if formation == 'circle':
        # Divide the circle into equal segments
        angles = np.linspace(0, 2 * np.pi, num_drones, endpoint=False)
        # Calculate the offset positions based on the angles
        offsets = np.array([
            radius * np.array([np.cos(angle), np.sin(angle)])
            for angle in angles
        ])
    else:
        raise ValueError(f"Unknown formation: {formation}")

    return offsets


def compute_control_input(drone, desired_centroid, formation_offsets, dt):
    ''' Computes drone control input based on relative positions, centroid and desired formation '''
    idx = drone.id - 1
    desired_position = desired_centroid + formation_offsets[idx]
    error = desired_position - drone.x[:2]

    u = np.zeros(drone.x.shape[0]) 
    
    # compute control input using PID
    u[:2] += drone.pid_controller.compute(error, dt)

    # Repulsion component to avoid collisions
    repulsion = np.zeros(2)
    for neighbor in drone.neighbors:
        distance = np.linalg.norm(drone.x[:2] - neighbor.x[:2])
        if distance < 2:  # Repulsion threshold
            repulsion += (drone.x[:2] - neighbor.x[:2]) / distance
    u[:2] += repulsion * 0.1  # Scale the repulsion

    # Log control inputs and error
    print(f"Drone {drone.id} - Error: {error}, Control Input: {u[:2]}")

    return u

def plot_formation_offsets(desired_centroid, formation_offsets):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Desired Formation Offsets')
    ax.grid(zorder=0)

    ax.scatter(desired_centroid[0], desired_centroid[1], marker='^', color='red', label='Fire', zorder=5)
    for i, offset in enumerate(formation_offsets):
        position = desired_centroid + offset
        ax.scatter(position[0], position[1], s=100, label=f'Drone {i+1}', zorder=5)

    ax.legend()
    plt.show()
