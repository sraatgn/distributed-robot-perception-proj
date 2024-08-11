import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import networkx as nx


def plot_simulation(drones, fire_position):
    plt.figure()
    for drone in drones:
        positions = np.array(drone.positions_upt)
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
    fire_scatter = ax.scatter(fire_position[0], fire_position[1], color='red', label='Fire', marker='^')

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

    def init():
        for line in lines:
            line.set_data([], [])
        for final_pos in final_positions:
            final_pos.set_offsets(np.empty((0, 2)))
        fire_scatter.set_offsets(np.array([fire_position]))
        return lines + final_positions + [fire_scatter]

    def update(frame):
        for line, final_pos, drone in zip(lines, final_positions, drones):
            positions = np.array(drone.positions_upt[:frame])
            if positions.size > 0:
                line.set_data(positions[:, 0], positions[:, 1])
                final_pos.set_offsets(positions[-1].reshape(1, 2))

        return lines + final_positions + [fire_scatter]

    ani = animation.FuncAnimation(
        fig, update, frames=range(1, max(len(drone.positions_upt) for drone in drones) + 1),
        init_func=init, blit=True, repeat=True, interval=500
    )

    plt.tight_layout()
    plt.show()

def animate_simulation_expected_traj(drones, fire_position):
    """ Plot animated simulation with estimated and desired trajectories """

    fig, ax = plt.subplots()
    ax.set_xlim(0, 30)  # Adjust according to the simulation boundaries
    ax.set_ylim(0, 30)  # Adjust according to the simulation boundaries
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Drone Fire Monitoring Simulation')
    ax.grid()

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    lines_estimated = []
    lines_true = [] 
    
    for i, drone in enumerate(drones):
        color = color_cycle[i % len(color_cycle)]
        line_est, = ax.plot([], [], color=color, label=f'Drone {drone.id} Estimated')
        line_true, = ax.plot([], [], linestyle='--', color=color, label=f'Drone {drone.id} True')
        lines_estimated.append(line_est)
        lines_true.append(line_true)
    
    final_positions_estimated = [ax.scatter([], [], s=100, color=color_cycle[i % len(color_cycle)]) for i in range(len(drones))]
    fire_scatter = ax.scatter(fire_position[0], fire_position[1], color='red', label='Fire Detected')

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

    def init():
        for line in lines_estimated + lines_true:
            line.set_data([], [])
        for final_pos in final_positions_estimated:
            final_pos.set_offsets(np.empty((0, 2)))
        fire_scatter.set_offsets(np.array([fire_position]))
        return lines_estimated + lines_true + final_positions_estimated + [fire_scatter]

    def update(frame):
        for line_est, line_true, final_pos_est, drone in zip(lines_estimated, lines_true, final_positions_estimated, drones):
            estimated_positions = np.array(drone.positions_upt[:frame])
            true_positions = np.array(drone.true_positions[:frame])
            if estimated_positions.size > 0:
                line_est.set_data(estimated_positions[:, 0], estimated_positions[:, 1])
                final_pos_est.set_offsets(estimated_positions[-1].reshape(1, 2))
            if true_positions.size > 0:
                line_true.set_data(true_positions[:, 0], true_positions[:, 1])

        return lines_estimated + lines_true + final_positions_estimated + [fire_scatter] 

    ani = animation.FuncAnimation(
        fig, update, frames=range(1, max(len(drone.positions_upt) for drone in drones) + 1),
        init_func=init, blit=True, repeat=True, interval=500
    )

    plt.tight_layout()
    plt.show()

def plot_trajectories(drones, fire_position, plot_pred=True):
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i,drone in enumerate(drones):
        color = color_cycle[i % len(color_cycle)]
        true_positions = np.array(drone.true_positions)
        pred_positions = np.array(drone.positions_pred)
        updated_positions = np.array(drone.positions_upt)
        plt.plot(true_positions[:, 0], true_positions[:, 1], '--'  ,label=f'True Traj Drone {drone.id}', color=color)
        plt.plot(updated_positions[:, 0], updated_positions[:, 1], label=f'Updated Traj Drone {drone.id}', color=color)
        if plot_pred:
            plt.plot(pred_positions[:, 0], pred_positions[:, 1], ':', label=f'Pred Traj Drone {drone.id}', color=color)

        # cross at the start of each trajectory
        plt.scatter(true_positions[0, 0], true_positions[0, 1], marker='x', color=color, s=100, zorder=5)
        # dot at the end of each trajectory
        plt.scatter(true_positions[-1, 0], true_positions[-1, 1], marker='o', color=color, s=100, zorder=5)

    plt.scatter(fire_position[0], fire_position[1], marker='^', color='red', label='Fire', zorder=5)
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    plt.title('Trajectories')
    plt.grid(True, zorder=0)
    plt.xlim(0, 30)  
    plt.ylim(0, 30)  
    plt.tight_layout()
    plt.show()


## PLOT METRICS
def plot_error(time_steps, err_prediction, err_update, error_type='RMSE'):
    """
    Plots the RMSE for prediction and update steps over time.

    Parameters:
        time_steps (list or np.array): Array of time steps.
        err_prediction (np.array): Array of error values after prediction.
        err_update (np.array): Array of error values after update.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(time_steps), np.array(err_prediction), label=f'{error_type} after Prediction', color='blue')
    plt.plot(range(time_steps), np.array(err_update), label=f'{error_type} after Update', color='green')
    plt.xlabel('Iteration')
    plt.ylabel(error_type)
    plt.title(f'{error_type} Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_kalman_gain(drones):
    for drone in drones:
        gains = np.array(drone.kalmangains)
        plt.figure(figsize=(12, 6))
        for i in range(gains.shape[1]):
            for j in range(gains.shape[2]):
                plt.plot(gains[:, i, j], label=f'K[{i},{j}]')
        plt.title(f'Kalman Gain Evolution for Drone {drone.id}')
        plt.xlabel('Iteration')
        plt.ylabel('Kalman Gain Value')
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_x_variance_over_time(drones):
    plt.figure(figsize=(10, 5))
    for drone in drones:
        x_variances = drone.x_variances
        plt.plot(x_variances, label=f'Drone {drone.id}')
    plt.xlabel('Iteration')
    plt.ylabel('Variance of x Position')
    plt.title('Variance of x Position Over Time')
    plt.legend()
    plt.grid(True)
    #plt.ylim(0, 0.5)  # Adjust this limit based on the expected range of variances
    plt.show()

def plot_variances_over_time(drones, loc='x'):

    color_map = plt.get_cmap('tab10', len(drones))  # 'tab10' can generate up to 10 distinct colors
    colors = [color_map(i) for i in range(len(drones))]

    for i, drone in enumerate(drones):
        if loc == 'x':
            variances_pred = [P[0,0] for P in drone.covariances_pred]
            variances_upt = [P[0,0] for P in drone.covariances_upt]
        elif loc == 'y':
            variances_pred = [P[1,1] for P in drone.covariances_pred]
            variances_upt = [P[1,1] for P in drone.covariances_upt]

        time_steps_pred = range(len(variances_pred))
        time_steps_upt = range(len(variances_upt))
        
        color = colors[i % len(colors)]
        
        plt.plot(time_steps_pred, variances_pred, linestyle='--', color=color, label=f'Pred Variance Drone {drone.id}')
        plt.plot(time_steps_upt, variances_upt, linestyle='-', color=color, label=f'Upt Variance Drone {drone.id}')
    
    plt.xlabel('Iteration')
    plt.ylabel(f'Variance of {loc} Position')
    plt.title(f'Variance of {loc} Position Over Time (Prediction and Update)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_measurement_network(measurements_history, num_drones):
    # Create a directed graph
    G = nx.DiGraph()
    # Dictionary to count how many times each drone made a measurement
    a_count = {i+1: 0 for i in range(num_drones)}
    b_count = {i+1: 0 for i in range(num_drones)}

    # Add nodes for each drone
    for i in range(num_drones):
        G.add_node(i+1, label=f'Drone {i+1}')

    # Add edges based on the measurement history
    for (a, b) in measurements_history:
        if a <= num_drones and b <= num_drones:
            G.add_edge(a, b, label=f'{a} -> {b}')
            a_count[a] += 1
            b_count[b] += 1

    # Draw the graph
    pos = nx.spring_layout(G)  # Position nodes using Fruchterman-Reingold force-directed algorithm
    labels = {
        i: f'Drone {i}\nMade: {a_count[i]}\nReceived: {b_count[i]}' 
        for i in range(1, num_drones + 1)
    }
    nx.draw(G, pos, labels=labels, node_size=2000, node_color='skyblue', arrows=True, arrowsize=20, font_size=9, font_color='black')

    plt.title('Measurement Network')
    plt.show()


