import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def animate_simulation(drones, fire_position, interim_master):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Drone Fire Detection Simulation')
    ax.grid()

    lines = [ax.plot([], [], label=f'Drone {drone.id}')[0] for drone in drones]
    final_positions = [ax.scatter([], [], s=100) for _ in drones]
    fire_scatter = ax.scatter(fire_position[0], fire_position[1], color='red', label='Fire Detected')
    interim_master_marker = ax.scatter([], [], s=200, edgecolors='blue', facecolors='none', linewidths=2, label='Interim Master')

    ax.legend()

    def init():
        for line in lines:
            line.set_data([], [])
        for final_pos in final_positions:
            final_pos.set_offsets(np.empty((0, 2)))
        fire_scatter.set_offsets(np.array([fire_position]))
        interim_master_marker.set_offsets(np.empty((0, 2)))
        return lines + final_positions + [fire_scatter, interim_master_marker]

    def update(frame):
        for line, final_pos, drone in zip(lines, final_positions, drones):
            positions = np.array(drone.positions[:frame])
            if positions.size > 0:
                line.set_data(positions[:, 0], positions[:, 1])
                final_pos.set_offsets(positions[-1].reshape(1, 2))

        if interim_master:
            interim_master_marker.set_offsets(interim_master.positions[frame-1].reshape(1, 2))

        return lines + final_positions + [fire_scatter, interim_master_marker]

    ani = animation.FuncAnimation(
        fig, update, frames=range(1, max(len(drone.positions) for drone in drones) + 1),
        init_func=init, blit=True, repeat=True, interval=500
    )

    plt.show()
