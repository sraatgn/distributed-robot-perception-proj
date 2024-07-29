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
