# Efficient Monitoring of Peat Fires Using EKF-Based Distributed Cooperative Localization with Multiple UAVs

This repository contains the code for the final exam of the course Distributed Robot Perception. It consists of a simulation of a **distributed Cooperative Localization algorithm** for a group of agents (drones) aiming at monitoring the area subject to a peat fire. 
In detail: an adjustable number of agents start at random positions and aim at converging around the target point in a circular formation, to monitor the area surrounding it. This is achieved by combining a distributed Cooperative Localization algorithm and PID control input in a simplified framework (for the sake of simulation). Results and more details are available in the project's report. 

## Contents
- `src/`: classes.
  - `drone.py`: Drone class (initialization of agents, predict & update steps of filter).
  - `pid_controller.py`: PIDController class.
- `utils/`
    - `formation.py`: (I) initialization functions for matrices (II) functions for computing formation and control input. 
	  - `measurements.py`: measurement model and simulation of relative measurements.
	  - `metrics.py`: functions for computing error metrics.
    - `pid_tuning.py`: functions for PID grid-search.
    - `plotting.py`: plotting functions (animations, trajectories, evaluation).
- `main.py`: main simulation file to run.

## Run the simulation
The simulation can be run by:
1. Cloning this repository in the desired directory
``` 
git clone https://github.com/sraatgn/distributed-robot-perception-proj.git
```
2. In that directory (`cd <dir/path>`), create a virtual environment:
``` 
python3.10 -m venv robotenv
```
3. Activate environment:
```
# On Windows
robotenv\Scripts\activate
# On macOS/Linux
source robotenv/bin/activate
```
4. Install required packages:
```
pip install -r requirements.txt
```
5. Run `main.py` with desired parameters. The simulation parameters (e.g. number of agents) can be changed in the final section of the file when calling the `simulate_fire_detection()` function. To compute metrics and plot results (un)comment desired lines in the same section and in the simulation function.

## Authors
- Sara Tegoni [@sraatgn](https://github.com/sraatgn)
- Juana Sofia Cruz Contento [@juanathebanana](https://github.com/juanathebanana)
