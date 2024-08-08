import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from IPython import embed

# def calculate_rmse(drones):
#     total_error = 0
#     count = 0
#     for drone in drones:
#         for est_pos, true_pos in zip(drone.positions, drone.true_positions):
#             total_error += np.linalg.norm(est_pos - true_pos)**2
#             count += 1
#     rmse = np.sqrt(total_error / count)
#     return rmse

def calculate_rmse(drones, num_timesteps, after_update=True):
    """
    Computes the RMSE between the true states and the estimated states.
    Average over all drones for each time step

    Returns:
        list of RMSE values for each time step.
    """
    rmse = []

    for t in range(num_timesteps):
        total_error = 0
        num_valid_drones = 0
        
        for drone in drones:
            true_states_len = len(drone.true_positions)
            estimated_states_len = len(drone.positions_upt if after_update else drone.positions_pred)
            
            if t < true_states_len and t < estimated_states_len:
                true_state = drone.true_positions[t]
                estimated_state = drone.positions_upt[t] if after_update else drone.positions_pred[t]
                total_error += np.linalg.norm(estimated_state - true_state) ** 2
                num_valid_drones += 1
            else:
                print(f"Warning: Drone {drone.id} does not have data for time step {t}")
        
        if num_valid_drones > 0:
            rmse.append(np.sqrt(total_error / num_valid_drones))
        else:
            rmse.append(None)  # No valid data for this time step
    
    return rmse

def compute_rmse(true_positions, estimated_positions):
    true_positions = np.array(true_positions)
    estimated_positions = np.array(estimated_positions)
    mse = np.mean((true_positions - estimated_positions) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def calculate_nmse(drones, num_timesteps, after_update=True):
    """
    Computes the NMSE (Normalized Mean Squared Error) between the true states and the estimated states.

    Returns:
        list of NMSE values for each time step.
    """
    nmse = []

    for t in range(num_timesteps):
        total_error = 0
        total_variance = 0
        num_valid_drones = 0
        
        for drone in drones:
            true_states_len = len(drone.true_positions)
            estimated_states_len = len(drone.positions_upt if after_update else drone.positions_pred)
            covariances_len = len(drone.covariances_upt if after_update else drone.covariances_pred)
            
            if t < true_states_len and t < estimated_states_len and t < covariances_len:
                true_state = drone.true_positions[t]
                estimated_state = drone.positions_upt[t] if after_update else drone.positions_pred[t]
                covariance_matrix = drone.covariances_upt[t] if after_update else drone.covariances_pred[t]
                
                # Calculate squared error
                squared_error = np.linalg.norm(estimated_state - true_state) ** 2
                total_error += squared_error
                
                # Use the trace of the covariance matrix as the total variance
                total_variance += np.trace(covariance_matrix)
                
                num_valid_drones += 1
            else:
                print(f"Warning: Drone {drone.id} does not have data for time step {t}")
        
        if num_valid_drones > 0 and total_variance > 0:
            # Calculate mean squared error
            mse = total_error / num_valid_drones
            # Normalize by total variance
            nmse_value = mse / (total_variance / num_valid_drones)
            nmse.append(nmse_value)
        else:
            nmse.append(None)  # No valid data for this time step or zero variance
    
    return nmse



def calculate_detection_metrics(drones, fire_position):
    y_true = []
    y_pred = []
    for drone in drones:
        for pos in drone.positions:
            y_true.append(np.linalg.norm(pos - fire_position) < 5)
            y_pred.append(drone.detect_fire(fire_position))
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1

