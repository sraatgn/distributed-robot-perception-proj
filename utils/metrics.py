import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_rmse(drones):
    total_error = 0
    count = 0
    for drone in drones:
        for est_pos, true_pos in zip(drone.positions, drone.true_positions):
            total_error += np.linalg.norm(est_pos - true_pos)**2
            count += 1
    rmse = np.sqrt(total_error / count)
    return rmse

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

