import numpy as np

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

def initialize_process_noise_covariance_matrix(num_agents, default_variance=[0.1, 0.1], custom_variances=None):
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
    state_dim = 2

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

    u = np.zeros(4)
    
    # for neighbor in drone.neighbors:
    #     relative_position = neighbor.x[:2] - drone.x[:2]
    #     u[:2] += drone.pid_controller.compute(relative_position, dt)
    
    u[:2] += drone.pid_controller.compute(error, dt)

    # Repulsion component to avoid collisions
    repulsion = np.zeros(2)
    for neighbor in drone.neighbors:
        distance = np.linalg.norm(drone.x[:2] - neighbor.x[:2])
        if distance < 5:  # Repulsion threshold
            repulsion += (drone.x[:2] - neighbor.x[:2]) / distance
    u[:2] += repulsion * 0.1  # Scale the repulsion

    return u
