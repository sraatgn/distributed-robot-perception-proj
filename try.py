import numpy as np
import matplotlib.pyplot as plt

class Drone:
    def __init__(self, id, position, sensor_range, comm_range):
        self.id = id
        self.position = np.array(position)
        self.sensor_range = sensor_range
        self.comm_range = comm_range
        self.map = {}  # Local map

    def move(self):
        # Random walk for simplicity
        self.position += np.random.uniform(-1, 1, 2)
    
    def sense(self, environment):
        ## Sense surroundings (e.g., distances to landmarks)
        # Agent position: simulate GPS signal with noise
        gps_signal = self.position + np.random.normal(0, 0.1, 2)  # GPS noise
        self.update_position(gps_signal)
    
    def communicate(self, other_drones):
        ## Share information with nearby drones
        for other in other_drones:
            if other.id != self.id and np.linalg.norm(self.position - other.position) <= self.comm_range:
                # Share positions (this is a simple example; more complex merging strategies can be used)
                self.map[other.id] = other.position

    
    def update_map(self):
        # Update the local map
        pass

    def update_position(self, gps_signal):
        self.position = gps_signal  # Directly update position using GPS signal


def simulate(drones, environment, steps):
    plt.ion()  # Turn on interactive mode
    for _ in range(steps):
        for drone in drones:
            drone.move()
            drone.sense(environment)
            drone.communicate(drones)
        visualize(drones)
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep the plot open after the simulation finishes

def visualize(drones):
    plt.clf()
    for drone in drones:
        plt.plot(drone.position[0], drone.position[1], 'ro')
        circle = plt.Circle(drone.position, drone.sensor_range, color='b', fill=False)
        plt.gca().add_artist(circle)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.pause(0.1)

if __name__ == "__main__":
    num_drones = 5
    drones = [Drone(id=i, position=np.random.uniform(-5, 5, 2), sensor_range=2.0, comm_range=5.0) for i in range(num_drones)]
    environment = {}  # Define the environment with landmarks
    simulate(drones, environment, 100)
