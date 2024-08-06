import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
            positions = np.array(drone.positions_upt[:frame])
            if positions.size > 0:
                line.set_data(positions[:, 0], positions[:, 1])
                final_pos.set_offsets(positions[-1].reshape(1, 2))

        return lines + final_positions + [fire_scatter]

    ani = animation.FuncAnimation(
        fig, update, frames=range(1, max(len(drone.positions_upt) for drone in drones) + 1),
        init_func=init, blit=True, repeat=True, interval=500
    )

    plt.show()

def animate_simulation_expected_traj(drones, fire_position):
    """ Plot animated simulation with estimated and desired trajectories """

    fig, ax = plt.subplots()
    ax.set_xlim(0, 30)  # Adjust according to the simulation boundaries
    ax.set_ylim(0, 30)  # Adjust according to the simulation boundaries
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Drone Fire Detection Simulation')
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

    ax.legend()

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

    plt.show()

def plot_trajectories(drones, fire_position):
    for drone in drones:
        true_positions = np.array(drone.true_positions)
        pred_positions = np.array(drone.positions_pred)
        updated_positions = np.array(drone.positions_upt)
        plt.plot(true_positions[:, 0], true_positions[:, 1], label=f'True Traj Drone {drone.id}')
        plt.plot(pred_positions[:, 0], pred_positions[:, 1], '--', label=f'Pred Traj Drone {drone.id}')
        plt.plot(updated_positions[:, 0], updated_positions[:, 1], ':', label=f'Updated Traj Drone {drone.id}')
    plt.scatter(fire_position[0], fire_position[1], marker='x', color='red', label='Fire Position')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.legend()
    plt.title('Trajectories of Drones')
    plt.grid(True)
    plt.xlim(0, 30)  
    plt.ylim(0, 30)  
    plt.show()


## PLOT METRICS
def plot_rmse(time_steps, rmse_prediction, rmse_update):
    """
    Plots the RMSE for prediction and update steps over time.

    Parameters:
        time_steps (list or np.array): Array of time steps.
        rmse_prediction (np.array): Array of RMSE values after prediction.
        rmse_update (np.array): Array of RMSE values after update.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(time_steps), np.array(rmse_prediction), label='RMSE After Prediction', color='blue')
    plt.plot(range(time_steps), np.array(rmse_update), label='RMSE After Update', color='green')
    plt.xlabel('Time Step')
    plt.ylabel('RMSE')
    plt.title('RMSE Over Time')
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
        plt.xlabel('Time Step')
        plt.ylabel('Kalman Gain Value')
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_x_variance_over_time(drones):
    plt.figure(figsize=(10, 5))
    for drone in drones:
        x_variances = drone.x_variances
        plt.plot(x_variances, label=f'Drone {drone.id}')
    plt.xlabel('Time Step')
    plt.ylabel('Variance of x Position')
    plt.title('Variance of x Position Over Time')
    plt.legend()
    plt.grid(True)
    #plt.ylim(0, 0.5)  # Adjust this limit based on the expected range of variances
    plt.show()

def plot_x_variances_over_time(drones):
    color_map = plt.get_cmap('tab10', len(drones))  # 'tab10' can generate up to 10 distinct colors
    colors = [color_map(i) for i in range(len(drones))]

    for i, drone in enumerate(drones):
        x_variances_pred = drone.x_variances_pred
        x_variances_upt = drone.x_variances_upt
        
        time_steps_pred = range(len(x_variances_pred))
        time_steps_upt = range(len(x_variances_upt))
        
        color = colors[i % len(colors)]
        
        plt.plot(time_steps_pred, x_variances_pred, linestyle='--', color=color, label=f'Pred Variance Drone {drone.id}')
        plt.plot(time_steps_upt, x_variances_upt, linestyle='-', color=color, label=f'Upt Variance Drone {drone.id}')
    
    plt.xlabel('Time Step')
    plt.ylabel('Variance of x Position')
    plt.title('Variance of x Position Over Time (Prediction and Update)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

