import numpy as np

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
